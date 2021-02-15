#!/usr/bin/env python

import numpy as np
from numpy.random import standard_normal
from numpy.linalg import inv,pinv,norm

import time,copy,extinction,pickle

from scipy.interpolate import splprep,splev,BSpline,griddata,bisplev,bisplrep,interp1d,interp2d
from scipy.integrate import trapz, simps
from scipy.optimize import minimize, least_squares,minimize_scalar,lsq_linear
from scipy.ndimage import gaussian_filter1d
from scipy.special import factorial
from scipy import linalg,sparse
from scipy.sparse import linalg as sprslinalg

import scipy.stats as ss

from sncosmo.salt2utils import SALT2ColorLaw
from sncosmo.models import StretchSource
from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
from sncosmo.utils import integration_grid

from salt3.util.synphot import synphot
from salt3.util.query import query_yes_no
from salt3.training import saltresids #_master as saltresids

from multiprocessing import Pool, get_context
from emcee.interruptible_pool import InterruptiblePool
from iminuit import Minuit
from datetime import datetime
from tqdm import tqdm
from itertools import starmap
from os import path

from scipy.ndimage import gaussian_filter1d
from scipy import sparse


import iminuit,warnings
import logging
log=logging.getLogger(__name__)

#Which integer code corresponds to which reason for LSMR terminating
stopReasons=['x=0 solution','atol approx. solution','atol+btol approx. solution','ill conditioned','machine precision limit','machine precision limit','machine precision limit','max # of iteration']

class SALTTrainingResult(object):
	 def __init__(self, **kwargs):
		 self.__dict__.update(kwargs)


def ensurepositivedefinite(matrix,maxiter=5):
	mineigenval=np.linalg.eigvalsh(matrix)[0]
	print(mineigenval)
	if mineigenval>0:
		return matrix
	else:
		if maxiter==0: 
			raise ValueError('Unable to make matrix positive semidefinite')
		return ensurepositivedefinite(matrix+np.diag(-mineigenval*2* np.ones(matrix.shape[0])),maxiter-1)
		

def getgaussianfilterdesignmatrix(shape,smoothing):
	windowsize=10+shape%2
	window=gaussian_filter1d(1.*(np.arange(windowsize)==windowsize//2),smoothing)
	while ~(np.any(window==0)):
		windowsize=2*windowsize+shape%2
		window=gaussian_filter1d(1.*(np.arange(windowsize)==windowsize//2),smoothing)
	window=window[window>0]

	diagonals=[]
	offsets=list(range(-(window.size//2),window.size//2+1))
	for i,offset in enumerate(offsets):
		diagonals+=[np.tile(window[i],shape-np.abs(offset))]
	design=sparse.diags(diagonals,offsets).tocsr()
	for i in range(window.size//2+1):
		design[i,:window.size//2+1]=gaussian_filter1d(1.*(np.arange(design.shape[0])== i ),5)[:window.size//2+1]
		design[-i-1,-(window.size//2+1) : ]=gaussian_filter1d(1.*(np.arange(design.shape[0])== i ),5)[:window.size//2+1][::-1]	  
	return design

def isDiag(M):
	i, j = M.shape
	assert i == j 
	test = M.reshape(-1)[:-1].reshape(i-1, j+1)
	return ~np.any(test[:, 1:])

def ridders(f,central,h,maxn,tol):
	"""Iterative method to evaluate the second derivative of a function f based on a stepsize h and a relative tolerance tol"""
	lookup={}
	def A(n,m):
		if (n,m) in lookup: return lookup[n,m]
		if n==1:
			result=(f((h/2**(m-1)))-2*central+f((-h/2**(m-1))))/(h/2**(m-1))**2
		elif n<1:
			return 0
		else:
			result =(4**(n-1)*A(n-1,m+1)-A(n-1,m))/(4**(n-1)-1)
		lookup[(n,m)]=result
		return result
	def AwithErr(n):
		diff=A(n,1)-A(n-1,1)
		return A(n,1),	norm(diff)/ min(norm(A(n,1)),norm(A(n-1,1)))
	best=AwithErr(2)
	result,err=best
	prev=best[1]
	diverging=0
	errs=[prev]
	for n in range(3,maxn+1):
		if err<tol:
			log.debug(f'Second directional derivative found to tolerance {tol} after {n} iterations')
			best= result,err
			break
		elif err>prev:
			diverging+=1
			if diverging>2:
				log.warning(f'Second directional derivative diverging after {n} iterations, tolerance is {best[1]}')
				break
		else:
			diverging=0
			if err<best[1]:
				best=result,err
				diverging=0
		result,err=AwithErr(n)
		errs+=[err]
		prev=err
	return best

class fitting:
	def __init__(self,n_components,n_colorpars,
				 n_phaseknots,n_waveknots,datadict):

		self.n_phaseknots = n_phaseknots
		self.n_waveknots = n_waveknots
		self.n_components = n_components
		self.n_colorpars = n_colorpars
		self.datadict = datadict

	def gaussnewton(self,gn,guess,
					gaussnewton_maxiter,
					only_data_errs=False):

		gn.debug = False
		convergenceresult = \
				gn.convergence_loop(
				guess,loop_niter=gaussnewton_maxiter)
			
		return convergenceresult,\
			'Gauss-Newton MCMC was successful'
		
	def mcmc(self,saltfitter,guess,
			 n_processes,n_mcmc_steps,
			 n_burnin_mcmc,stepsizes=None):
		
		saltfitter.debug = False
		if n_processes > 1:
			with InterruptiblePool(n_processes) as pool:
				x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
					modelerr,clpars,clerr,clscat,SNParams = \
					saltfitter.mcmcfit(
						guess,n_mcmc_steps,n_burnin_mcmc,pool=pool,stepsizes=stepsizes)
		else:
			x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
				modelerr,clpars,clerr,clscat,SNParams = \
				saltfitter.mcmcfit(
					guess,n_mcmc_steps,n_burnin_mcmc,pool=None,stepsizes=stepsizes)

		return x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams,'Adaptive MCMC was successful'


class mcmc(saltresids.SALTResids):
	def __init__(self,guess,datadict,parlist,chain=[],loglikes=[],**kwargs):
		self.loglikes=loglikes
		self.chain=chain

		super().__init__(guess,datadict,parlist,**kwargs)
		
		
	def get_proposal_cov(self, n, beta=0.25):
		d, _ = self.M2_recent.shape
		init_period = self.nsteps_before_adaptive
		s_0, s_opt, C_0 = self.AMpars['sigma_0'], self.AMpars['sigma_opt'], self.AMpars['C_0']
		if n<= init_period or np.random.rand()<=beta:
			return np.sqrt(C_0), False
		else:
			# We can always divide M2 by n-1 since n > init_period
			return np.sqrt((s_opt/(self.nsteps_adaptive_memory - 1))*self.M2_recent), True
	
	def generate_AM_candidate(self, current, n, steps_from_gn=False):
		prop_std,adjust_flag = self.get_proposal_cov(n)
		
		#tstart = time.time()
		candidate = np.zeros(self.npar)
		candidate = np.random.normal(loc=current,scale=np.diag(prop_std))
		for i,par in zip(range(self.npar),self.parlist):
			if self.adjust_snpars and (par == 'm0' or par == 'm1' or par == 'modelerr'):
				candidate[i] = current[i]
			elif self.adjust_modelpars and par != 'm0' and par != 'm1' and par != 'modelerr':
				candidate[i] = current[i]
			else:
				if not steps_from_gn and (par.startswith('modelerr') or par.startswith('x0') or par == 'm0' or par == 'clscat'):
					candidate[i] = current[i]*10**(0.4*np.random.normal(scale=prop_std[i,i]))
				else:
					pass
		return candidate
		
	def get_propcov_init(self,x,stepsizes=None):
		C_0 = np.zeros([len(x),len(x)])
		if stepsizes is not None:
			for i,par in zip(range(self.npar),self.parlist):
				C_0[i,i] = stepsizes[i]**2.
		else:
			for i,par in zip(range(self.npar),self.parlist):
				if par == 'm0':
					C_0[i,i] = self.stepsize_magscale_M0**2.
				elif par.startswith('modelerr'):
					C_0[i,i] = (self.stepsize_magscale_err)**2.
				elif par == 'm1':
					C_0[i,i] = (self.stepsize_magadd_M1)**2.
				elif par.startswith('x0'):
					C_0[i,i] = self.stepsize_x0**2.
				elif par.startswith('x1'):
					C_0[i,i] = self.stepsize_x1**2.
				elif par == 'clscat':
					C_0[i,i] = (self.stepsize_magscale_clscat)**2.
				elif par.startswith('c'): C_0[i,i] = (self.stepsize_c)**2.
				elif par.startswith('specrecal'): C_0[i,i] = self.stepsize_specrecal**2.
				elif par.startswith('tpkoff'):
					C_0[i,i] = self.stepsize_tpk**2.
				elif par.startswith('modelcorr'):
					C_0[i,i]= self.stepsize_errcorr**2
		self.AMpars = {'C_0':C_0,
					   'sigma_0':0.1/np.sqrt(self.npar),
					   'sigma_opt':2.38*self.adaptive_sigma_opt_scale/np.sqrt(self.npar)}
	
	def update_moments(self, sample, n):
		next_n = (n + 1)
		w = 1/next_n
		new_mean = self.mean + w*(sample - self.mean)
		delta_bf, delta_af = sample - self.mean, sample - new_mean
		self.M2 += np.outer(delta_bf, delta_af)
		self.mean = new_mean

		return
	
	def mcmcfit(self,x,nsteps,nburn,pool=None,debug=False,thin=1,stepsizes=None,SpecErrScale=0.01):
		npar = len(x)
		self.npar = npar
		self.chain,self.loglikes = [],[]
		# initial log likelihood
		if self.chain==[]:
			self.chain+=[x]
		if self.loglikes==[]:
			self.loglikes += [self.maxlikefit(x,pool=pool,debug=debug,SpecErrScale=SpecErrScale)]
		self.M0stddev = np.std(x[self.parlist == 'm0'])
		self.M1stddev = np.std(x[self.parlist == 'm1'])
		self.errstddev = self.stepsize_magscale_err
		self.M2 = np.zeros([len(x),len(x)])
		self.M2_recent = np.empty_like(self.M2)
		self.mean = x[:], 

		if stepsizes is not None:
			steps_from_gn = True
			stepsizes[stepsizes > 0.1] = 0.1
			stepsizes *= 1e-14
		else: steps_from_gn = False
		self.get_propcov_init(x,stepsizes=stepsizes)
		accept = 0
		nstep = 0
		accept_frac = 0.5
		accept_frac_recent = 0.5
		accepted_history = np.array([])
		n_adaptive = 0
		self.adjust_snpars,self.adjust_modelpars = False,False
		while nstep < nsteps:
			nstep += 1
			n_adaptive += 1
			
			if not nstep % 50 and nstep > 250:
				accept_frac_recent = len(accepted_history[-100:][accepted_history[-100:] == True])/100.
			if self.modelpar_snpar_tradeoff_nstep:
				if not nstep % self.modelpar_snpar_tradeoff_nstep and nstep > self.nsteps_before_modelpar_tradeoff:
					if self.adjust_snpars: self.adjust_modelpars = True; self.adjust_snpars = False
					else: self.adjust_modelpars = False; self.adjust_snpars = True

			if self.use_lsqfit:
				if not (nstep+1) % self.nsteps_between_lsqfit:
					X = self.lsqguess(current=self.chain[-1],snpars=True)
				if not (nstep) % self.nsteps_between_lsqfit:
					X = self.lsqguess(current=self.chain[-1],doMangle=True)
				else:
					X = self.generate_AM_candidate(current=self.chain[-1], n=nstep, steps_from_gn=steps_from_gn)
			else:
				X = self.generate_AM_candidate(current=self.chain[-1], n=nstep, steps_from_gn=steps_from_gn)
			self.__components_time_stamp__ = time.time()
			
			# loglike
			this_loglike = self.maxlikefit(X,pool=pool,debug=debug,SpecErrScale=SpecErrScale)
			accept_bool = self.accept(self.loglikes[-1],this_loglike)
			if accept_bool:
				if not nstep % thin:

					self.chain+=[X]
				self.loglikes+=[this_loglike]

				accept += 1
				log.info('step = %i, accepted = %i, acceptance = %.3f, recent acceptance = %.3f'%(
					nstep,accept,accept/float(nstep),accept_frac_recent))
			else:
				if not nstep % thin:

					self.chain+=[self.chain[-1]]
				self.loglikes += [self.loglikes[-1]]

			accepted_history = np.append(accepted_history,accept_bool)
			if not (nstep) % self.nsteps_between_lsqfit:
				self.updateEffectivePoints(self.chain[-1])
			self.update_moments(self.chain[-1], n_adaptive)
			if not n_adaptive % self.nsteps_adaptive_memory:
				n_adaptive = 0
				#ix,iy = np.where(self.M2 < 1e-5)
				#iLow = np.where(ix == iy)[0]
				#self.M2[ix[iLow],iy[iLow]] = 1e-5
				#if self.adjust_snpars and 'M2_snpars' in self.__dict__.keys(): M2_recent = copy.deepcopy(self.M2_snpars)
				#elif self.adjust_snpars and 'M2_snpars' not in self.__dict__.keys(): M2_recent = copy.deepcopy(self.M2_allpars)
				#elif self.adjust_modelpars and 'M2_modelpars' in self.__dict__.keys(): M2_recent = copy.deepcopy(self.M2_modelpars)
				#elif self.adjust_modelpars and 'M2_modelpars' not in self.__dict__.keys(): M2_recent = copy.deepcopy(self.M2_allpars)
				#else:
				
				self.M2_recent = np.empty_like(self.M2)
				self.M2_recent[:] = self.M2
				self.mean = self.chain[-1][:]
				self.M2 = np.empty_like(self.M2)
				#if self.adjust_snpars: self.M2_snpars = copy.deepcopy(M2_recent)
				#elif self.adjust_modelpars: self.M2_modelpars = copy.deepcopy(M2_recent)

		log.info('acceptance = %.3f'%(accept/float(nstep)))
		if nstep < nburn:
			raise RuntimeError('Not enough steps to wait %i before burn-in'%nburn)
		xfinal,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams = \
			self.getPars(self.loglikes,np.array(self.chain).T,nburn=int(nburn/thin))
		
		return xfinal,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams
		
	def accept(self, last_loglike, this_loglike):
		alpha = np.exp(this_loglike - last_loglike)
		return_bool = False
		if alpha >= 1:
			return_bool = True
		else:
			if np.random.rand() < alpha:
				return_bool = True
		return return_bool



class GaussNewton(saltresids.SALTResids):
	def __init__(self,guess,datadict,parlist,**kwargs):
		self.debug=False
		super().__init__(guess,datadict,parlist,**kwargs)
		self.lsqfit = False
		
		self.GN_iter = {}
		self.damping={}
		self.directionaloptimization=True
		self.geodesiccorrection=False
		self.updatejacobian=True
		self._robustify = False
		self._writetmp = False
		self.chi2_diff_cutoff = .1
		self.fitOptions={}
		self.iModelParam=np.ones(self.npar,dtype=bool)
		self.iModelParam[self.imodelerr]=False
		self.iModelParam[self.imodelcorr]=False
		self.iModelParam[self.iclscat]=False
		self.uncertaintyKeys=set(['photvariances_' +sn for sn in self.datadict]+['specvariances_' +sn for sn in self.datadict]+sum([[f'photCholesky_{sn}_{flt}' for flt in self.datadict[sn].filt] for sn in self.datadict],[]))
		
		self.Xhistory=[]
		self.tryFittingAllParams=True
		fitlist = [('all parameters','all'),('all parameters grouped','all-grouped'),('supernova params','sn'),
				   (" x0",'x0'),('both components','components'),('component 0 piecewise','piecewisecomponent0'),('principal component 0','component0'),('x1','x1'),
				   ('component 1 piecewise','piecewisecomponent1'),('principal component 1','component1'),('color','color'),('color law','colorlaw'),
				   ('spectral recalibration const.','spectralrecalibration_norm'),('all spectral recalibration','spectralrecalibration'),
				   ('time of max','tpk'),('error model','modelerr'),('params with largest residuals','highestresids'),('pca parameters only','pcaparams'), 
				   ('noncomponent parameters','snparams+colorlaw+recal'),('components and recalibration','components+recal')]+ [(f'sn {sn}',sn) for sn in self.datadict.keys()]

		for message,fit in fitlist:
			self.GN_iter[fit]=1
			self.damping[fit]=0.1
			if 'all' in fit or fit=='highestresids':
				includePars=np.ones(self.npar,dtype=bool)
				if not self.fitTpkOff:
					includePars[self.itpk]=False
			else:
				includePars=np.zeros(self.npar,dtype=bool)
				if fit in self.datadict:
					sn=fit
					includePars=np.array([ sn in name.split('_') for name in self.parlist])
				elif fit=='components+recal':
					includePars[self.im0]=True
					includePars[self.im1]=True
					includePars[self.ispcrcl]=True
				elif 'pcaparams' == fit:
					self.GN_iter[fit]=2
					includePars[self.im0]=True
					includePars[self.im1]=True		
					includePars[self.ix0]=True
					includePars[self.ix1]=True
				elif 'components' in fit:
					includePars[self.im0]=True
					includePars[self.im1]=True		
				elif 'component0' in fit :
					self.damping[fit]=1e-3
					includePars[self.im0]=True
				elif 'component1' in fit:
					self.damping[fit]=1e-3
					includePars[self.im1]=True		
				elif fit=='sn':
					self.damping[fit]=1e-3
					includePars[self.ix0]=True
					includePars[self.ix1]=True
				elif fit=='snparams+colorlaw+recal':
					includePars[self.ix0]=True
					includePars[self.ix1]=True
					includePars[self.ic]=True
					includePars[self.iCL]=True
					includePars[self.ispcrcl]=True
				
				elif fit=='x0':
					self.damping[fit]=0
					includePars[self.ix0]=True
				elif fit=='x1':
					self.damping[fit]=1e-3
					includePars[self.ix1]=True
				elif fit=='color':
					self.damping[fit]=1e-3
					includePars[self.ic]=True
				elif fit=='colorlaw':
					includePars[self.iCL]=True
				elif fit=='tpk':
					includePars[self.itpk]=True
				elif fit=='spectralrecalibration':
					if len(self.ispcrcl):
						includePars[self.ispcrcl]=True
					else:
						self.ispcrcl = []
				elif fit=='spectralrecalibration_norm':
					if len(self.ispcrcl_norm):
						includePars[self.ispcrcl_norm]=True
					else:
						self.ispcrcl = []
				elif fit=='modelerr':
					includePars[self.imodelerr]=True
					includePars[self.imodelcorr]=True
					includePars[self.parlist=='clscat']=True
				else:
					raise NotImplementedError("""This option for a Gauss-Newton fit with a 
	restricted parameter set has not been implemented: {}""".format(fit))
			if self.fix_salt2modelpars:
				includePars[self.im0]=False
				includePars[self.im1]=False		
				includePars[self.im0new]=True
				includePars[self.im1new]=True
			self.fitOptions[fit]=(message,includePars)

		if kwargs['fitting_sequence'].lower() == 'default' or not kwargs['fitting_sequence']:
			self.fitlist = [('all'),
							('pcaparams'),
							('color'),('colorlaw'),
							('spectralrecalibration'),		
							('sn'),
							('tpk')]
		else:
			self.fitlist = [f for f in kwargs['fitting_sequence'].split(',')]

	def datauncertaintiesfromhessianapprox(self,X,suppressregularization=True,smoothingfactor=150):
		"""Approximate Hessian by jacobian times own transpose to determine uncertainties in flux surfaces"""
		log.info("determining M0/M1 errors by approximated Hessian")
		itpk=np.zeros(X.size,dtype=bool)
		itpk[self.itpk]=True
		varyingParams=self.fitOptions['components'][1]|self.fitOptions['color'][1]|self.fitOptions['spectralrecalibration'][1]|self.fitOptions['colorlaw'][1]
		logging.debug('Allowing parameters {np.unique(self.parlist[varyingParams])} in calculation of inverse Hessian')
		if suppressregularization:
			self.neff[self.neff<self.neffMax]=10
		residuals,jac=self.lsqwrap(X,{},varyingParams,True,doSpecResids=True)
		self.updateEffectivePoints(X)
		#Simple preconditioning of the jacobian before attempting to invert
		precondition=sparse.diags(1/np.sqrt(np.asarray((jac.power(2)).sum(axis=0))),[0])
		precondjac=(jac*precondition)
		
		#Define the matrix sigma^-1=J^T J
		square=ensurepositivedefinite(precondjac.T*precondjac.toarray())
		#Inverting cholesky matrix for speed
		L=linalg.cholesky(square,lower=True)
		invL=(linalg.solve_triangular(L,np.diag(np.ones(L.shape[0])),lower=True))
		#Turning spline_derivs into a sparse matrix for speed
		spline_derivs = np.zeros([len(self.phaseout),len(self.waveout),self.im0.size])
		for i in range(self.im0.size):
			if self.bsorder == 0: continue
			spline_derivs[:,:,i]=bisplev(self.phaseout,self.waveout,(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder))
		spline2d=sparse.csr_matrix(spline_derivs.reshape(-1,self.im0.size))
		
		#Smooth things a bit, since this is supposed to be for broadband photometry
		if smoothingfactor>0:
			smoothingmatrix=getgaussianfilterdesignmatrix(spline2d.shape[0],smoothingfactor/self.waveoutres)
			spline2d=smoothingmatrix*spline2d
		#Uncorrelated effect of parameter uncertainties on M0 and M1
		varyparlist= self.parlist[varyingParams]
		m0pulls=invL*precondition.tocsr()[:,varyparlist=='m0']*spline2d.T
		m1pulls=invL*precondition.tocsr()[:,varyparlist=='m1']*spline2d.T 
		M0dataerr = np.sqrt((m0pulls**2).sum(axis=0).reshape((self.phaseout.size,self.waveout.size)))
		cov_M0_M1_data = (m0pulls*m1pulls).sum(axis=0).reshape((self.phaseout.size,self.waveout.size))
		del m0pulls
		M1dataerr = np.sqrt((m1pulls**2).sum(axis=0).reshape((self.phaseout.size,self.waveout.size)))
		del m1pulls
		correlation=cov_M0_M1_data/(M0dataerr*M1dataerr)
		correlation[np.isnan(correlation)]=0
		M0,M1=self.SALTModel(X)
		M0dataerr=np.clip(M0dataerr,0,np.abs(M0).max()*2)
		M1dataerr=np.clip(M1dataerr,0,np.abs(M1).max()*2)
		correlation=np.clip(correlation,-1,1)
		cov_M0_M1_data=correlation*(M0dataerr*M1dataerr)
		
		return M0dataerr, M1dataerr,cov_M0_M1_data

	def datauncertaintiesfromjackknife(self,X,max_iter,n_bootstrapsamples):
		"""Determine uncertainties in flux surfaces by bootstrapping"""
		log.info("determining M0/M1 errors by bootstrapping")
		snlist=list(self.datadict.keys())
		removesnindices=np.random.choice(len(snlist),n_bootstrapsamples)
		samples=[self.convergence_loop(X,max_iter, snlist[:i]+snlist[i+1:], getdatauncertainties=False).X for i in	removesnindices]
		deviation=(samples-X)
		sigma=ensurepositivedefinite(np.dot(deviation.T,deviation)/(deviation.shape[0]-1))
		L=linalg.cholesky(sigma,lower=True)
		
		#Turning spline_derivs into a sparse matrix for speed
		spline2d=sparse.csr_matrix(self.spline_derivs.reshape(-1,self.im0.size))
		m0pulls=L[varyparlist=='m0',:].T*spline2d.T
		m1pulls=L[varyparlist=='m1',:].T*spline2d.T 
		
		M0dataerr = np.sqrt((m0pulls**2).sum(axis=0).reshape((self.phase.size,self.wave.size)))
		M1dataerr = np.sqrt((m1pulls**2).sum(axis=0).reshape((self.phase.size,self.wave.size)))
		cov_M0_M1_data = (m0pulls*m1pulls).sum(axis=0).reshape((self.phase.size,self.wave.size))

		return M0dataerr, M1dataerr,cov_M0_M1_data
	
	def convergence_loop(self,guess,loop_niter=3,usesns=None,getdatauncertainties=True):
		lastResid = 1e20
		log.info('Initializing')
		start=datetime.now()

		#if self.fit_model_err: fixUncertainty = False
		#else: fixUncertainty = True
		
		if len(self.usePriors) != len(self.priorWidths):
			raise RuntimeError('length of priors does not equal length of prior widths!')
		stepsizes=None
		#residuals = self.lsqwrap(guess,uncertainties,False,False,doPriors=False)
	
		X = copy.deepcopy(guess[:])
		clscatzeropoint=X[self.iclscat[-1]]
		nocolorscatter=clscatzeropoint==-np.inf
		if not nocolorscatter: log.debug('Turning off color scatter for convergence_loop')
		X[self.iclscat[-1]]=-np.inf


		Xlast = copy.deepcopy(guess[:])
		if np.all(X[self.ix1]==0) or np.all(X[self.ic]==0):
			#If snparams are totally uninitialized
			log.info('Estimating supernova parameters x0,x1,c and spectral normalization')
			for fit in ['x0','color','x0','color','x1']:
				X,chi2_init,chi2=self.process_fit(
					X,self.fitOptions[fit][1],{},fit=fit,doPriors=False,
					doSpecResids=  (fit=='x0'),allowjacupdate=False)
		else:
			chi2_init=(self.lsqwrap(X,{},usesns=usesns)**2).sum()
		log.info(f'starting loop; {loop_niter} iterations')
		chi2results=self.getChi2Contributions(X,{})
		for name,chi2component,dof in chi2results:
			if name.lower()=='photometric':
				photochi2perdof=chi2component/dof
		
		for superloop in range(loop_niter):
			tstartloop = time.time()
			try:
				if not superloop % 5 and  self.fit_model_err and not self.fit_cdisp_only and photochi2perdof<3 :# and not superloop == 0:
					X=self.iterativelyfiterrmodel(X)
					storedResults={}
					chi2results=self.getChi2Contributions(X,storedResults)
					uncertainties={key:storedResults[key] for key in self.uncertaintyKeys}
				else:
					log.info('Reevaluted model error')
					storedResults={}
					chi2results=self.getChi2Contributions(X,storedResults)
					uncertainties={key:storedResults[key] for key in self.uncertaintyKeys}
				
				for name,chi2component,dof in chi2results:
					log.info('{} chi2/dof is {:.1f} ({:.2f}% of total chi2)'.format(name,chi2component/dof,chi2component/sum([x[1] for x in chi2results])*100))
					if name.lower()=='photometric':
						photochi2perdof=chi2component/dof

				X,chi2,converged = self.robust_process_fit(X,uncertainties.copy(),chi2_init,superloop,usesns=usesns)
				if chi2_init-chi2 < -1.e-6:
					log.warning("MESSAGE WARNING chi2 has increased")
				elif np.abs(chi2_init-chi2) < self.chi2_diff_cutoff:

					log.info(f'chi2 difference less than cutoff {self.chi2_diff_cutoff}, exiting loop')
					break

				log.info(f'finished iteration {superloop+1}, chi2 improved by {chi2_init-chi2:.1f}')
				log.info(f'iteration {superloop+1} took {time.time()-tstartloop:.3f} seconds')

				if converged:
					log.info('Gauss-Newton optimizer could not further improve chi2')
					break
				chi2_init = chi2
				stepsizes = self.getstepsizes(X,Xlast)
				Xlast = copy.deepcopy(X)

			except KeyboardInterrupt as e:
				if query_yes_no("Terminate optimization loop and begin writing output?"):
					break
				else:
					if query_yes_no("Enter pdb?"):
						import pdb;pdb.set_trace()
					else:
						raise e
			except Exception as e:
				logging.exception('Error encountered in convergence_loop, exiting')
				raise e
		X[self.iclscat[-1]]=clscatzeropoint
		try:
			if self.fit_model_err: X= self.fitcolorscatter(X)
		except Exception as e:
			logging.critical('Color scatter crashed during fitting, finishing writing output')
			logging.critical(e, exc_info=True)

		Xredefined=self.priors.satisfyDefinitions(X,self.SALTModel(X))
		logging.info('Checking that rescaling components to satisfy definitions did not modify photometry')
		
		try:
			unscaledresults={}
			scaledresults={}
			for sn in self.datadict:
				photresidsunscaled=self.ResidsForSN(X,sn,unscaledresults)[0]
				photresidsrescaled=self.ResidsForSN(Xredefined,sn,scaledresults)[0]
				for flt in photresidsunscaled:
					assert(np.allclose(photresidsunscaled[flt]['resid'],photresidsrescaled[flt]['resid']))
		except AssertionError:
			logging.critical('Rescaling components failed; photometric residuals have changed. Will finish writing output using unscaled quantities')
			Xredefined=X.copy()
		
		if getdatauncertainties:
			try:
				M0dataerr, M1dataerr,cov_M0_M1_data=self.datauncertaintiesfromhessianapprox(Xredefined)
			except:
				print('uncertainties failed!!')
				M0dataerr, M1dataerr,cov_M0_M1_data=None,None,None
		else:
			M0dataerr, M1dataerr,cov_M0_M1_data=None,None,None
		# M0/M1 errors
		xfinal,phase,wave,M0,M0modelerr,M1,M1modelerr,cov_M0_M1_model,\
			modelerr,clpars,clerr,clscat,SNParams = \
			self.getParsGN(Xredefined)
		if M0dataerr is None:
			M0dataerr, M1dataerr,cov_M0_M1_data = np.zeros(len(M0)),np.zeros(len(M0)),np.zeros(len(M0))
		#log.info("using Minuit to determine the uncertainties on M0 and M1")
		#if self.fit_model_err:
		#	log.info('Optimizing model error')
		#	m0var,m1var,m0m1cov=self.iterativelyfitdataerrmodel(X)
		#else:
		#	m0var = m1var = m0m1cov = np.zeros(len(M0))
		#M0dataerr,M1dataerr,cov_M0_M1_data = self.getErrsGN(m0var,m1var,m0m1cov)

		log.info('Total time spent in convergence loop: {}'.format(datetime.now()-start))
		

		
		return SALTTrainingResult(num_lightcurves=self.num_lc,num_spectra=self.num_spectra,num_sne=len(self.datadict),
				parlist=self.parlist,X=xfinal,X_raw=X,phase=phase,wave=wave,M0=M0,M0modelerr=M0modelerr,M0dataerr=M0dataerr,
				M1=M1,M1modelerr=M1modelerr,M1dataerr=M1dataerr,cov_M0_M1_model=cov_M0_M1_model,cov_M0_M1_data=cov_M0_M1_data,
				modelerr=modelerr,clpars=clpars,clerr=clerr,clscat=clscat,SNParams=SNParams,stepsizes=stepsizes)
		
		#raise RuntimeError("convergence_loop reached 100000 iterations without convergence")
	def fitOneSN(self,X,sn):
		X=X.copy()
		includePars=self.fitOptions[sn][1] 
		includePars[self.itpk]=False
		#Estimate parameters first using GN method
		for par in self.parlist[includePars]:
			if 'specrecal' in par: continue
			if 'specx0' in par:
				result=self.ResidsForSN(X,sn,{},varyParams=self.parlist==par,fixUncertainty=True)[1]
			else:
				result=self.ResidsForSN(X,sn,{},varyParams=self.parlist==par,fixUncertainty=True)[0]
			resid,grad=result['resid'],result['resid_jacobian'][:,0]
			X[self.parlist==par]-=np.dot(grad,resid)/np.dot(grad,grad)

		def fn(Y):
			if len(Y[Y != Y]):
				import pdb; pdb.set_trace()
			Xnew=X.copy()
			Xnew[includePars]=Y
			return - self.loglikeforSN(Xnew,sn,{},varyParams=None)
		def grad(Y):
			if len(Y[Y != Y]):
				import pdb; pdb.set_trace()
			Xnew=X.copy()
			Xnew[includePars]=Y
			#log.info(self.maxlikefit(Xnew,computeDerivatives=True)[1])
			#import pdb; pdb.set_trace()
			return - self.loglikeforSN(Xnew,sn,{},varyParams=includePars)[1]
		log.info('Initialized log likelihood: {:.2f}'.format(self.loglikeforSN(X,sn,{},varyParams=None)))
		params=['x'+str(i) for i in range(includePars.sum())]
		

		initVals=X[includePars].copy()
		kwargs={}
		for i,parname in enumerate(self.parlist[includePars]):
			if 'x1' in parname:
				kwargs['limit_'+params[i]] = (-5,5)
			elif 'x0' in parname:
				kwargs['limit_'+params[i]] = (0,2)
			elif 'c_' in parname:
				kwargs['limit_'+params[i]] = (-0.5,1)
			elif 'specrecal' in parname:
				kwargs['limit_'+params[i]] = (-0.5,0.5)
			else:
				kwargs['limit_'+params[i]] = (-5,5)

		kwargs.update({params[i]: initVals[i] for i in range(includePars.sum())})
		m=Minuit(fn,use_array_call=True,forced_parameters=params,grad=grad,errordef=1,**kwargs)
		result,paramResults=m.migrad()#includePars.sum()*6)
		#if np.allclose(np.array([x.value for x	 in paramResults]),X[includePars]):
		X=X.copy()
		X[includePars]=np.array([x.value for x	in paramResults])

		#		if np.allclose(X[includePars],initVals):

		log.info('Final log likelihood: {:.2f}'.format( -result.fval))
		
		return X,-result.fval
	
	def fitcolorscatter(self,X,fitcolorlaw=False,rescaleerrs=True,maxiter=2000):
		message='Optimizing color scatter'
		if rescaleerrs:
			message+=' and scaling model uncertainties'
		if fitcolorlaw:
			message+=', with color law varying'
		log.info(message)
		if X[self.iclscat[-1]]==-np.inf:
			X[self.iclscat[-1]]=-8
		includePars=np.zeros(self.parlist.size,dtype=bool)
		includePars[self.iclscat]=True
		includePars[self.iCL]=fitcolorlaw
		
		log.info('Initialized log likelihood: {:.2f}'.format(self.maxlikefit(X,{})))
		storedResults={}
		storedResults['components'] =self.SALTModel(X)
		if self.bsorder != 0: storedResults['componentderivs'] = self.SALTModelDeriv(X,1,0,self.phase,self.wave)		
		if not rescaleerrs :
			if 'saltErr' not in storedResults:
				storedResults['saltErr']=self.ErrModel(X)
			if 'saltCorr' not in storedResults:
				storedResults['saltCorr']=self.CorrelationModel(X)
		log.debug(str(storedResults.keys()))
		X,minuitresult=self.minuitoptimize(X,includePars,{},rescaleerrs=rescaleerrs,fixFluxes=not fitcolorlaw,dospec=False,maxiter=maxiter)
		log.info('Finished optimizing color scatter')
		log.debug(str(minuitresult))
		return X
		 
	def iterativelyfiterrmodel(self,X):
		log.info('Optimizing model error')
		X=X.copy()
		imodelerr=np.zeros(self.parlist.size,dtype=bool)
		imodelerr[self.imodelerr]=True
		problemerrvals=(X<0)&imodelerr
		X[problemerrvals]=1e-3

		X0=X.copy()
		mapFun= starmap
		
		log.debug('turning off model priors')
		store_priors = self.usePriors.copy()
		self.usePriors = ()
		
		storedResults={}
		log.info('Initialized log likelihood: {:.2f}'.format(self.maxlikefit(X,storedResults)))
		fluxes={key:storedResults[key] for key in storedResults if 'fluxes' in key }
		storedResults=fluxes.copy()
		args=[(X0,sn,storedResults,None,False,1,True,False) for sn in self.datadict.keys()]
		result0=np.array(list(mapFun(self.loglikeforSN,args)))
		partriplets= list(zip(np.where(self.parlist=='modelerr_0')[0],np.where(self.parlist=='modelerr_1')[0],np.where(self.parlist=='modelcorr_01')[0]))

		for i,parindices in tqdm(enumerate(partriplets)):
			includePars=np.zeros(self.parlist.size,dtype=bool)
			includePars[list(parindices)]=True
			storedResults=fluxes.copy()
			args=[(X0+includePars*.5,sn,storedResults,None,False,1,True,False) for sn in self.datadict.keys()]
			result=np.array(list(mapFun(self.loglikeforSN,args)))
			usesns=np.array(list(self.datadict.keys()))[result!=result0]
			logging.debug(f'{usesns.size} SNe constraining {i}th error bin')
			X,minuitfitresult=self.minuitoptimize(X,includePars,fluxes,fixFluxes=True,dospec=False,usesns=usesns)
		log.debug('turning on model priors')
		self.usePriors = store_priors
		log.info('Finished model error optimization')
		return X

	def minuitoptimize(self,X,includePars,storedResults=None,rescaleerrs=False,maxiter=100,**kwargs):
		X=X.copy()
		if not self.fitTpkOff: includePars[self.itpk]=False
		if storedResults is None: storedResults={}
		if	not rescaleerrs:
			def fn(Y):
				Xnew=X.copy()
				Xnew[includePars]=Y
				result=-self.maxlikefit(Xnew,storedResults.copy(),**kwargs)
				return 1e10 if np.isnan(result) else result
		else:
			def fn(Y):
				try:
					Xnew=X.copy()
					Xnew[includePars]=Y[:-1]
					Xnew[self.imodelerr]*=Y[-1]
					result=-self.maxlikefit(Xnew,storedResults.copy(),**kwargs)
					return 1e10 if np.isnan(result) else result
				except KeyboardInterrupt as e:
					raise e
				except:
					logging.warning('Error caught in minuit loop')
					return 1e10
		params=['x'+str(i) for i in range(includePars.sum())]
		initVals=X[includePars].copy()

		#kwargs={'limit_'+params[i] : self.bounds[np.where(includePars)[0][i]] for i in range(includePars.sum()) if }
		minuitkwargs=({params[i]: initVals[i] for i in range(includePars.sum())})
		minuitkwargs.update({'error_'+params[i]: 1e-2 for i in range(includePars.sum())})
		clscatindices=np.where(self.parlist[includePars] == 'clscat')[0]
		if clscatindices.size>0:
			minuitkwargs.update({'limit_'+params[i]: (-1e-4,1e-4) for i in [clscatindices[0]]})
			minuitkwargs.update({'limit_'+params[i]: (-1,1) for i in clscatindices[1:-1]})
			minuitkwargs.update({'limit_'+params[i]: (-10,2) for i in [clscatindices[-1]]})
		minuitkwargs.update({'limit_'+params[i]: (-1,1) for i in np.where(self.parlist[includePars] == 'modelerr_0')[0]})
		minuitkwargs.update({'limit_'+params[i]: (-1,1) for i in np.where(self.parlist[includePars] == 'modelerr_1')[0]})
		minuitkwargs.update({'limit_'+params[i]: (-1,1) for i in np.where(self.parlist[includePars] == 'modelcorr_01')[0]})
		
		minuitkwargs.update({'limit_'+params[i]: (-100,100) for i in np.where(self.parlist[includePars] == 'cl')[0]})

		
		if rescaleerrs:
			extrapar= 'x'+str(includePars.sum())
			params+=[extrapar]
			minuitkwargs[extrapar]=1
			minuitkwargs['error_'+extrapar]=1e-2
			minuitkwargs['limit_'+extrapar]=(0,2)
		
		m=Minuit(fn,use_array_call=True,forced_parameters=params,errordef=.5,**minuitkwargs)
		try:
			result,paramResults=m.migrad(ncall=maxiter)
		except KeyboardInterrupt:
			logging.info('Keyboard interrupt, exiting minuit loop')
			return X,None
		X=X.copy()
		
		paramresults=np.array([x.value for x  in paramResults])
		if rescaleerrs:
			X[includePars]=paramresults[:-1]
			X[self.imodelerr]*=paramresults[-1]
		else:
			X[includePars]=paramresults
		
		return X,result
	
	#def iterativelyfitdataerrmodel(self,X):
	#
	#	X0=X.copy()
	#	mapFun= starmap
	#
	#	storedResults={}
	#	print('Initialized log likelihood: {:.2f}'.format(self.maxlikefit(X,storedResults)))
	#	fluxes={key:storedResults[key] for key in storedResults if 'fluxes' in key }
	#	storedResults=fluxes.copy()
	#	args=[(X0,sn,storedResults,None,False,1,True,False) for sn in self.datadict.keys()]
	#	result0=np.array(list(mapFun(self.loglikeforSN,args)))
	#	pardoublets= list(zip(np.where(self.parlist=='m0')[0],np.where(self.parlist=='m1')[0]))
	#
	#	m0var,m1var,m01cov = np.zeros(len(pardoublets)),np.zeros(len(pardoublets)),np.zeros(len(pardoublets))
	#	for i,parindices in enumerate(tqdm(pardoublets)):
	#		waverange=self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
	#		phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-self.bsorder-1),i//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
	#		if phaserange[0] < 0 and phaserange[1] > 0 and waverange[0] < 5000 and waverange[1] > 5000:
	#			includePars=np.zeros(self.parlist.size,dtype=bool)
	#			includePars[list(parindices)]=True
	#			
	#			storedResults=fluxes.copy()
	#			args=[(X0+includePars*.5,sn,storedResults,None,False,1,True,False) for sn in self.datadict.keys()]
	#			result=np.array(list(mapFun(self.loglikeforSN,args)))
	#
	#			m0var[i],m1var[i],m01cov[i] = self.minuitoptimize_components(X,includePars,fluxes,fixFluxes=False,dospec=True)
	#
	#	return m0var,m1var,m01cov
			 
	#def minuitoptimize_components(self,X,includePars,storedResults=None,**kwargs):
	#
	#	X=X.copy()
	#	if storedResults is None: storedResults={}
	#	def fn(Y):
	#		Xnew=X.copy()
	#		Xnew[includePars]=Y
	#		result=-self.maxlikefit(Xnew,{},**kwargs) #storedResults.copy(),**kwargs)
	#		return 1e10 if np.isnan(result) else result
	#
	#	params=['x'+str(i) for i in range(includePars.sum())]
	#	initVals=X[includePars].copy()
	#
	#	minuitkwargs=({params[i]: initVals[i] for i in range(includePars.sum())})
	#	minuitkwargs.update({'error_'+params[i]: 1e-2 for i in range(includePars.sum())})
	#
	#	m=Minuit(fn,use_array_call=True,forced_parameters=params,errordef=.5,**minuitkwargs)
	#	result,paramResults=m.migrad()
	#	paramerr=np.array([x.value for x  in paramResults])
	#
	#	if m.covariance:
	#		return paramResults[0].error**2.,paramResults[1].error**2.,m.covariance[('x0', 'x1')]
	#	else:
	#		return paramResults[0].error**2.,paramResults[1].error**2.,0.0

	def iterativelyfitdataerrmodel(self,X):

		X0=X.copy()
		mapFun= starmap

		storedResults={}
		log.info('Initialized log likelihood: {:.2f}'.format(self.maxlikefit(X,storedResults)))
		fluxes={key:storedResults[key] for key in storedResults if 'fluxes' in key }
		storedResults=fluxes.copy()
		args=[(X0,sn,storedResults,None,False,1,True,False) for sn in self.datadict.keys()]
		result0=np.array(list(mapFun(self.loglikeforSN,args)))
		pars = np.where(self.parlist=='modelerr_0')[0]
		
		m0var,m1var,m01cov = np.zeros(self.im0.size),np.zeros(self.im0.size),np.zeros(self.im0.size)
		for i,parindices in enumerate(tqdm(pars)):
			waverange=self.errwaveknotloc[[i%(self.errwaveknotloc.size-self.errbsorder-1),i%(self.errwaveknotloc.size-self.errbsorder-1)+self.errbsorder+1]]
			phaserange=self.errphaseknotloc[[i//(self.errwaveknotloc.size-self.errbsorder-1),i//(self.errwaveknotloc.size-self.errbsorder-1)+self.errbsorder+1]]
			#if phaserange[0] < 0 and phaserange[1] > 0 and waverange[0] <= 5000 and waverange[1] > 5000:
			includeM0Pars=np.zeros(self.parlist.size,dtype=bool)
			includeM1Pars=np.zeros(self.parlist.size,dtype=bool)
			# could be slow (and ugly), but should work.....
			m0idx = np.array([],dtype=int)
			for j in range(self.im0.size):
				m0waverange=self.waveknotloc[[j%(self.waveknotloc.size-self.bsorder-1),j%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
				m0phaserange=self.phaseknotloc[[j//(self.waveknotloc.size-self.bsorder-1),j//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
				if (waverange[0] <= m0waverange[0] and waverange[1] >= m0waverange[0] and \
				   phaserange[0] <= m0phaserange[0] and phaserange[1] >= m0phaserange[0]) or \
				(waverange[0] <= m0waverange[1] and waverange[1] >= m0waverange[1] and \
				 phaserange[0] <= m0phaserange[1] and phaserange[1] >= m0phaserange[1]):
					includeM0Pars[self.im0[j]] = True
					includeM1Pars[self.im1[j]] = True
					m0idx = np.append(m0idx,j)
			varyParams = includeM0Pars | includeM1Pars
			#storedResults=fluxes.copy()

			m0var_single,m1var_single,m01cov_single = self.minuitoptimize_components(
				X,includeM0Pars,includeM1Pars,fluxes,fixFluxes=False,dospec=True,varyParams=varyParams)
			m0var[m0idx] = m0var_single*len(m0idx)
			m1var[m0idx] = m1var_single*len(m0idx)
			m01cov[m0idx] = m01cov_single*len(m0idx)
			#if m0var > 0: 
		return m0var,m1var,m01cov

	def minuitoptimize_components(self,X,includeM0Pars,includeM1Pars,storedResults=None,varyParams=None,**kwargs):

		X=X.copy()
		if storedResults is None: storedResults={}
		def fn(Y):
			Xnew=X.copy()
			Xnew[includeM0Pars] += Y[0]
			Xnew[includeM1Pars] += Y[1]
			storedCopy = storedResults.copy()
			if 'components' in storedCopy.keys():
				storedCopy.pop('components')
			result=-self.maxlikefit(Xnew,{})#storedCopy)#,**kwargs)
			return 1e10 if np.isnan(result) else result
			
		params=['x0','x1']
		minuitkwargs=({'x0':0,'x1':1})
		minuitkwargs.update({'error_x0': 1e-2,'error_x1': 1e-2})
		minuitkwargs.update({'limit_x0': (-3,3),'limit_x1': (-3,3)})

		
		m=Minuit(fn,use_array_call=True,forced_parameters=params,errordef=.5,**minuitkwargs)
		result,paramResults=m.migrad()

		if m.covariance:
			return paramResults[0].error**2.,paramResults[1].error**2.,m.covariance[('x0', 'x1')]
		else:
			return paramResults[0].error**2.,paramResults[1].error**2.,0.0

	
	def getstepsizes(self,X,Xlast):
		stepsizes = X-Xlast
		#stepsizes[self.parlist == 'm0'] = \
		#	(X[self.parlist == 'm0']-Xlast[self.parlist == 'm0'])/Xlast[self.parlist == 'm0']
		#stepsizes[self.parlist == 'modelerr'] = \
		#	(X[self.parlist == 'modelerr']-Xlast[self.parlist == 'modelerr'])/Xlast[self.parlist == 'modelerr']
		#stepsizes[self.parlist == 'clscat'] = \
		#	(X[self.parlist == 'clscat']-Xlast[self.parlist == 'clscat'])/Xlast[self.parlist == 'clscat']
		#import pdb; pdb.set_trace()
		return stepsizes
		
	def getChi2Contributions(self,guess,storedResults):
		if self.n_colorpars:
			if not 'colorLaw' in storedResults:
				storedResults['colorLaw'] = -0.4 * SALT2ColorLaw(self.colorwaverange, guess[self.parlist == 'cl'])(self.wave)
				storedResults['colorLawInterp']= interp1d(self.wave,storedResults['colorLaw'],kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
		else: storedResults['colorLaw'] = 1
				
		if not 'components' in storedResults:
			storedResults['components'] =self.SALTModel(guess)
		if not 'componentderivs' in storedResults:
			storedResults['componentderivs'] = self.SALTModelDeriv(guess,1,0,self.phase,self.wave)
		
		if not all([('specvariances_{}'.format(sn) in storedResults) and ('photvariances_{}'.format(sn) in storedResults) for sn in self.datadict]):
			storedResults['saltErr']=self.ErrModel(guess)
			storedResults['saltCorr']=self.CorrelationModel(guess)
		varyParams=np.zeros(guess.size,dtype=bool)
		photresids = []
		specresids = [] # Jacobian matrix from r

		for sn in self.datadict.keys():
			photresidsdict,specresidsdict=self.ResidsForSN(
				guess,sn,storedResults,varyParams,fixUncertainty=True)
			photresids+=[photresidsdict[k]['resid'] for k in photresidsdict]
			specresids+=[specresidsdict[k]['resid'] for k in specresidsdict]

		priorResids,priorVals,priorJac=self.priors.priorResids(self.usePriors,self.priorWidths,guess)
		priorResids=[priorResids]

		BoundedPriorResids,BoundedPriorVals,BoundedPriorJac = \
			self.priors.BoundedPriorResids(self.bounds,self.boundedParams,guess)
		priorResids+=[BoundedPriorResids]
		
		regResids=[]
		if self.regularize:
			for regularization, weight,regKey in [(self.phaseGradientRegularization, self.regulargradientphase,'regresult_phase'),
										   (self.waveGradientRegularization,self.regulargradientwave,'regresult_wave' ),
										   (self.dyadicRegularization,self.regulardyad,'regresult_dyad')]:
				if weight ==0:
					continue
				if regKey in storedResults:
					regResids += storedResults[regKey]
				else:
					for resids,regJac in zip( *regularization(guess,storedResults,varyParams)):
						regResids += [resids*np.sqrt(weight)]
					storedResults[regKey]=priorResids[-self.n_components:]
		chi2Results=[]
		for name,x in [('Photometric',photresids),('Spectroscopic',specresids),('Prior',priorResids),('Regularization',regResids)]:
			if ((len(regResids) and name == 'Regularization') or name != 'Regularization') and len(x):
				x=np.concatenate(x)
			else: x=np.array([0.0])
			chi2Results+=[(name,(x**2).sum(),x.size)]
		return chi2Results
	
	def robust_process_fit(self,X_init,uncertainties,chi2_init,niter,usesns=None):
		X,chi2=X_init.copy(),chi2_init
		storedResults=uncertainties.copy()
		for fit in self.fitlist:
			if 'all-grouped' in fit :continue #
			if 'modelerr' in fit: continue
			if 'tpk'==fit and not self.fitTpkOff: continue
			log.info('fitting '+self.fitOptions[fit][0])
		
		
			Xprop = X.copy()
			if (fit=='all'):
				if self.tryFittingAllParams:
					Xprop,chi2prop,chi2 = self.process_fit(Xprop,self.fitOptions[fit][1],storedResults,fit=fit,usesns=usesns)
					if (chi2prop/chi2 < 0.95):
						log.info('Terminating iteration {}, continuing with all parameter fit'.format(niter+1))
						return Xprop,chi2prop,False
					elif (chi2prop<chi2):
						X,chi2=Xprop,chi2prop
						storedResults= {key:storedResults[key] for key in storedResults if (key in self.uncertaintyKeys)}
					else:
						retainReg=True
						retainPCDerivs=True
						storedResults= {key:storedResults[key] for key in storedResults if (key in self.uncertaintyKeys) or
								(retainReg and key.startswith('regresult' )) or
							   (retainPCDerivs and key.startswith('pcDeriv_'   )) }
			elif fit.startswith('piecewisecomponent'):
				for i in range(self.GN_iter[fit]):
					for i,p in enumerate(self.phaseBinCenters):
						log.info(f'fitting phase {p:.1f}')
						indices=np.arange((self.waveknotloc.size-4)*(self.phaseknotloc.size-4))
						iFit= (((i-1)*(self.waveknotloc.size-4)) <= indices) & (indices <((i+2)*(self.waveknotloc.size-4)))
						includeParsPhase=np.zeros(self.npar,dtype=bool)

						if fit== 'piecewisecomponents':
							includeParsPhase[self.im0[iFit]]=True
							includeParsPhase[self.im1[iFit]]=True
						else:
							includeParsPhase[self.__dict__[f'im{fit[-1]}'][iFit]] = True
						Xprop,chi2prop,chi2 = self.process_fit(Xprop,includeParsPhase,storedResults,fit=fit,usesns=usesns)

						#includeParsPhase[self.__dict__['im%s'%fit[-1]][iFit]] = True
						#Xprop,chi2prop = self.process_fit(X,includeParsPhase,storedResults,fit=fit)

						if np.isnan(Xprop).any() or np.isnan(chi2prop) or ~np.isfinite(chi2prop):
							log.error('NaN detected, breaking out of loop')
							break;
						if chi2prop<chi2 :
							X,chi2=Xprop[:],chi2prop
						retainPCDerivs=True
						storedResults= {key:storedResults[key] for key in storedResults if (key in self.uncertaintyKeys) or
							   (retainPCDerivs and key.startswith('pcDeriv_'   )) }
					else:
						continue
			else:
				for i in range(self.GN_iter[fit]):
					Xprop,chi2prop,chi2 = self.process_fit(Xprop,self.fitOptions[fit][1],storedResults,fit=fit,usesns=usesns)
					if np.isnan(Xprop).any()  or np.isnan(chi2prop) or ~np.isfinite(chi2prop):
						log.error('NaN detected, breaking out of loop')
						break;
					if chi2prop<chi2 :
						X,chi2=Xprop,chi2prop
					varyParams=self.fitOptions[fit][1]
					retainReg=( ~ np.any(varyParams[self.im0]) and ~ np.any(varyParams[self.im1]) )
					retainPCDerivs=( ~ np.any(varyParams[self.ic]) and ~ np.any(varyParams[self.iCL]) and ~ np.any(varyParams[self.ispcrcl]) and ~ np.any(varyParams[self.itpk])  ) 
					storedResults= {key:storedResults[key] for key in storedResults if (key in self.uncertaintyKeys) or
							(retainReg and key.startswith('regresult' )) or
						   (retainPCDerivs and key.startswith('pcDeriv_'   )) }

					if chi2 != chi2 or chi2 == np.inf:
						break
				else:
					continue
		#In this case GN optimizer can do no better
		return X,chi2,(X is X_init)
		 #_init
	def linesearch(self,X,searchdir,uncertainties,**kwargs):
		def opFunc(x):
			return ((self.lsqwrap(X-(x*searchdir),uncertainties.copy(),None,**kwargs))**2).sum()
		result,step,stepType=minimize_scalar(opFunc),searchdir,'Gauss-Newton'
		log.info('Linear optimization factor is {:.2f} x {} step'.format(result.x,stepType))
		log.info('Linear optimized chi2 is {:.2f}'.format(result.fun))
		return result.x*searchdir,result.fun
		
		
	def process_fit(self,X,iFit,storedResults,fit='all',allowjacupdate=True,**kwargs):
		
		X=X.copy()
		varyingParams=iFit&self.iModelParam
		if 'usesns' in kwargs :
			if kwargs['usesns' ] is None:
				kwargs.pop('usesns')
			else:
				snnotinset=[sn for sn in self.datadict if sn not in kwargs['usesns']]
				sndependentparams=np.prod([self.fitOptions[sn][1] for sn in snnotinset],axis=0).astype(bool)
				log.debug(f'Removing {sndependentparams.sum()} params because {len(snnotinset)} SNe are not included in this iteration')
				varyingParams=varyingParams&~sndependentparams
		if not self.fitTpkOff: varyingParams[self.itpk]=False
		residuals,jacobian=self.lsqwrap(X,storedResults,varyingParams,**kwargs)
		oldChi=(residuals**2).sum()
		jacobian=jacobian.tocsc()
		
		if fit=='highestresids':
			#Fit only the parameters affecting the highest residual points
			includePars=np.diff(jacobian[ (residuals**2 > np.percentile(residuals**2,100-1e-3)) ,:].indptr) != 0
		else:
			#Exclude any parameters that are not currently affecting the fit (column in jacobian zeroed for that index)
			includePars= np.diff(jacobian.indptr) != 0
		if not includePars.all():
			varyingParams=varyingParams & includePars
			jacobian=jacobian[:,includePars]		
		
		#Exclude any residuals unaffected by the current fit (column in jacobian zeroed for that index)
		jacobian=jacobian.tocsr()

		log.info('Number of parameters fit this round: {}'.format(includePars.sum()))
		log.info('Initial chi2: {:.2f} '.format(oldChi))
		#Simple diagonal preconditioning matrix designed to make the jacobian better conditioned. Seems to do well enough! If output is showing significant condition number in J, consider improving this
		preconditoningMatrix=sparse.diags(1/np.sqrt(np.asarray((jacobian.power(2)).sum(axis=0))),[0])
		#If this is a sparse matrix, treat it as such, otherwise convert to a standard array
		isJacobianSparse=jacobian.nnz/(jacobian.shape[0]*jacobian.shape[1])<0.5
		if isJacobianSparse:
			precondjac=jacobian.dot(preconditoningMatrix)
		else:
			jacobian=jacobian.toarray()
			precondjac=jacobian*preconditoningMatrix

		uncertainties={key:storedResults[key] for key in storedResults if key in self.uncertaintyKeys}
		scale=1.5
		
		tol=1e-8
		#Convenience function to encapsulate the code to calculate LSMR results with varying damping
		def gaussNewtonFit(damping):
			result=sprslinalg.lsmr(precondjac,residuals,damp=damping,maxiter=2*min(jacobian.shape),atol=tol,btol=tol)
			gaussNewtonStep=np.zeros(X.size)
			gaussNewtonStep[varyingParams]=preconditoningMatrix*result[0]
			postGN=(self.lsqwrap(X-gaussNewtonStep,uncertainties.copy(),None,**kwargs)**2).sum() #
			if fit=='all': log.debug(f'Attempting fit with damping {damping} gave chi2 {postGN}')
			return result,postGN,gaussNewtonStep,damping
			
				
		if self.damping[fit]!=0 :
			results=[]
			#Try increasing, decreasing, or keeping the same damping to see which does best
			for dampingFactor in [1/scale,1] :
					results+=[(gaussNewtonFit(self.damping[fit]*dampingFactor))]
			result,postGN,gaussNewtonStep,damping=min(results,key=lambda x:x[1])
			precondstep,stopsignal,itn,normr,normar,norma,conda,normx=result
			#If the new position is worse, need to increase the damping until the new position is better
			increasedDamping=False
			while oldChi<postGN:
				increasedDamping=True
				self.damping[fit]*=scale*11/9
				result,postGN,gaussNewtonStep,damping=gaussNewtonFit(self.damping[fit])
				precondstep,stopsignal,itn,normr,normar,norma,conda,normx=result
			#Ratio of actual improvement in chi2 to how well the optimizer thinks it did
			reductionratio= (oldChi-postGN)/(oldChi-(normr**2))
			#If the chi^2 is improving less than expected, check whether damping needs to be increased
			if reductionratio<0.33 and not increasedDamping: 
				increaseddampingresult=gaussNewtonFit(self.damping[fit]*scale*11/9)
				if increaseddampingresult[1]<postGN:
					result,postGN,gaussNewtonStep,damping=increaseddampingresult
					precondstep,stopsignal,itn,normr,normar,norma,conda,normx=result
			self.damping[fit]=damping
		else:
			result,postGN,gaussNewtonStep,damping=gaussNewtonFit(0)
			precondstep,stopsignal,itn,normr,normar,norma,conda,normx=result
			reductionratio= (oldChi-postGN)/(oldChi-(normr**2))
		log.debug('First Gauss-Newton step: LSMR results with damping factor {:.2e}: {}, norm r {:.2f}, norm J^T r {:.2f}, norm J {:.2f}, cond J {:.2f}, norm step {:.2f}, reduction ratio {:.2f} required {} iterations'.format(self.damping[fit],stopReasons[stopsignal],normr,normar,norma,conda,normx,reductionratio,itn ))
		if stopsignal==7: log.warning('Gauss-Newton solver reached max # of iterations')

		if np.any(np.isnan(gaussNewtonStep)):
			log.error('NaN detected in stepsize; exitting to debugger')
			import pdb;pdb.set_trace()
		
		log.info('After Gauss-Newton chi2 is {:.2f}'.format(postGN))
		
		
		if self.updatejacobian and allowjacupdate:	
			prevresult=result
			currentresids=self.lsqwrap(X-gaussNewtonStep,uncertainties.copy(),None,**kwargs)
			prevresids=residuals
			prevjac=precondjac.copy()
			prevstep=gaussNewtonStep.copy()
			currentchi2=(currentresids**2).sum()
			for i in range(30):
				self.Xhistory+=[(X-prevstep,currentchi2,prevresult)]
				residpreddiff=((currentresids-prevresids)-prevjac.dot(-prevresult[0]))
				sparsestructure=((prevjac!=0)*sparse.diags(prevresult[0]))
				rowproduct=np.asarray((sparsestructure.power(2)).sum(axis=1)).T[0]
				rowproduct[rowproduct==0]=np.inf
				sparsestructure=sparse.diags(1/rowproduct)* sparsestructure
				currentjac= prevjac-(sparse.diags(residpreddiff)*sparsestructure)
				currentresult=sprslinalg.lsmr(currentjac,currentresids,damp=damping,maxiter=2*min(jacobian.shape),atol=tol,btol=tol)
				precondstep,stopsignal,itn,normr,normar,norma,conda,normx=currentresult

				currentstep=prevstep.copy()
				currentstep[varyingParams]+=preconditoningMatrix*currentresult[0]
				
				nextresids=self.lsqwrap(X-currentstep,uncertainties.copy(),None,**kwargs)
				nextchi2=(nextresids**2).sum()
				chi2improvement=currentchi2-nextchi2
				reductionratio= chi2improvement/(currentchi2-(normr**2))
				log.info(f'Reiterating with updated jacobian gives improvement {chi2improvement}')
				log.debug('On reiteration: LSMR results are {}, norm r {:.2f}, norm J^T r {:.2f}, norm J {:.2f}, cond J {:.2f}, norm step {:.2f}, reduction ratio {:.2f} required {} iterations'.format(stopReasons[stopsignal],normr,normar,norma,conda,normx,reductionratio,itn ))
				if chi2improvement<0:
					log.info('Negative improvement, finishing process_fit')
					break
				else:
					prevstep=currentstep
					prevresids=currentresids
					currentresids=nextresids
					currentchi2=nextchi2
					prevjac=currentjac
					prevresult=currentresult
			self.Xhistory+=[(X-prevstep,currentchi2,prevresult)]
			
			postGN=(currentresids**2).sum()
			gaussNewtonStep=prevstep
		else:
			if self.geodesiccorrection:
				#Fancypants cubic correction to the Gauss-Newton step based on fun differential geometry! See https://arxiv.org/abs/1207.4999 for derivation
				# Right now, doesn't seem worth it compared to directional fit; maybe worth trying both?
						
				directionalSecondDeriv,tol=	ridders(lambda dx: self.lsqwrap(X+dx*gaussNewtonStep,uncertainties.copy(),None,**kwargs) ,residuals,.5,5,1e-8)
				accelerationdir,stopsignal,itn,normr,normar,norma,conda,normx=sprslinalg.lsmr(precondjac,directionalSecondDeriv,damp=self.damping[fit],maxiter=2*min(jacobian.shape))
				secondStep=np.zeros(X.size)
				secondStep[varyingParams]=0.5*preconditoningMatrix*accelerationdir
			
				postgeodesic=(self.lsqwrap(X-gaussNewtonStep-secondStep,uncertainties.copy(),None,**kwargs)**2).sum() #doSpecResids
				log.info('After geodesic acceleration correction chi2 is {:.2f}'.format(postgeodesic))
				if postgeodesic<postGN :
					chi2=postgeodesic
					gaussNewtonStep=gaussNewtonStep+secondStep
				else:
					chi2=postGN
					gaussNewtonStep=gaussNewtonStep
		if self.directionaloptimization:
			linearStep,chi2=self.linesearch(X,gaussNewtonStep,uncertainties,**kwargs)	
			X-=linearStep		
		else:
			X-=gaussNewtonStep
			chi2=postGN
		
		with open(path.join(self.outputdir,'gaussnewtonhistory.pickle'),'wb') as file: pickle.dump(self.Xhistory,file)
		log.info('Chi2 diff, % diff')
		log.info(' '.join(['{:.2f}'.format(x) for x in [oldChi-chi2,(100*(oldChi-chi2)/oldChi)] ]))
		print('')
		return X,chi2,oldChi

