#!/usr/bin/env python

import numpy as np
from numpy.random import standard_normal
from numpy.linalg import inv,pinv

import time,copy,extinction

from scipy.interpolate import splprep,splev,BSpline,griddata,bisplev,bisplrep,interp1d,interp2d
from scipy.integrate import trapz, simps
from scipy.optimize import minimize, least_squares,minimize_scalar,lsq_linear
from scipy.stats import norm
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
import iminuit,warnings

import logging
log=logging.getLogger(__name__)

class fitting:
	def __init__(self,n_components,n_colorpars,
				 n_phaseknots,n_waveknots,datadict):

		self.n_phaseknots = n_phaseknots
		self.n_waveknots = n_waveknots
		self.n_components = n_components
		self.n_colorpars = n_colorpars
		self.datadict = datadict

	def gaussnewton(self,gn,guess,
					n_processes,n_mcmc_steps,
					n_burnin_mcmc,
					gaussnewton_maxiter):

		gn.debug = False
		xfinal,xunscaled,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams,stepsizes = \
			gn.convergence_loop(
				guess,loop_niter=gaussnewton_maxiter)

		return xfinal,xunscaled,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams,stepsizes,\
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
		
	def lsqguess(self, current, snpars=False, M0=False, M1=False, doMangle=False):

		candidate = copy.deepcopy(current)
		
		#salterr = self.ErrModel(candidate)
		if self.n_colorpars:
			colorLaw = SALT2ColorLaw(self.colorwaverange, candidate[self.parlist == 'cl'])
		else: colorLaw = None
		if self.n_colorscatpars:
			colorScat = True
		else: colorScat = None

		if snpars:
			log.info('using scipy minimizer to find SN params...')		
			components = self.SALTModel(candidate)
			log.info('error hack!')
			for sn in self.datadict.keys():
			
				def lsqwrap(guess):

					candidate[self.parlist == 'x0_%s'%sn] = guess[0]
					candidate[self.parlist == 'x1_%s'%sn] = guess[1]
					candidate[self.parlist == 'c_%s'%sn] = guess[2]
					candidate[self.parlist == 'tpkoff_%s'%sn] = guess[3]

					args = (None,sn,candidate,components,self.salterr,colorLaw,colorScat,False)
					return -self.loglikeforSN(args)


				guess = np.array([candidate[self.parlist == 'x0_%s'%sn][0],candidate[self.parlist == 'x1_%s'%sn][0],
								  candidate[self.parlist == 'c_%s'%sn][0],candidate[self.parlist == 'tpkoff_%s'%sn][0]])
			
				result = minimize(lsqwrap,guess)
				#self.blah = True
				candidate[self.parlist == 'x0_%s'%sn] = result.x[0]
				candidate[self.parlist == 'x1_%s'%sn] = result.x[1]
				candidate[self.parlist == 'c_%s'%sn] = result.x[2]
				candidate[self.parlist == 'tpkoff_%s'%sn] = result.x[3]

		elif M0:
			log.info('using scipy minimizer to find M0...')
				
			self.lsqfit = True
			def lsqwrap(guess):
				
				candidate[self.parlist == 'm0'] = guess
				components = self.SALTModel(candidate)

				logmin = np.array([])
				for sn in self.datadict.keys():
					args = (None,sn,candidate,components,salterr,colorLaw,colorScat,False)
					logmin = np.append(logmin,self.loglikeforSN(args))
				logmin *= -1
				logmin -= self.m0prior(components) + self.m1prior(candidate[self.ix1]) + self.endprior(components)
				if colorLaw:
					logmin -= self.EBVprior(colorLaw)
				
				log.info(np.sum(logmin*2))
				return logmin

			guess = candidate[self.parlist == 'm0']
			result = least_squares(lsqwrap,guess,max_nfev=6)
			candidate[self.parlist == 'm0'] = result.x
			self.lsqfit = False
			
		elif M1:
			log.info('using scipy minimizer to find M1...')
			
			self.lsqfit = True
			def lsqwrap(guess):

				candidate[self.parlist == 'm0'] = guess
				components = self.SALTModel(candidate)

				logmin = np.array([])
				for sn in self.datadict.keys():
					args = (None,sn,candidate,components,salterr,colorLaw,colorScat,False)
					logmin = np.append(logmin,self.loglikeforSN(args))
				logmin *= -1
				logmin -= self.m0prior(components) + self.m1prior(candidate[self.ix1]) + self.endprior(components)
				if colorLaw:
					logmin -= self.EBVprior(colorLaw)
				
				log.info(np.sum(logmin*2))
				return logmin

			guess = candidate[self.parlist == 'm0']
			result = least_squares(lsqwrap,guess,max_nfev=6)
			candidate[self.parlist == 'm0'] = result.x
			self.lsqfit = False

		elif doMangle:
			log.info('mangling!')
			if self.n_colorpars:
				colorLaw = SALT2ColorLaw(self.colorwaverange, current[self.parlist == 'cl'])
			else: colorLaw = None
				
			from mangle import mangle
			mgl = mangle(self.phase,self.wave,self.kcordict,self.datadict,
						 self.n_components,colorLaw,self.fluxfactor,
						 self.phaseknotloc,self.waveknotloc)
			components = self.SALTModel(candidate)

			guess = np.array([1.,1.,0.,1.,0.,0.,1.,0.])
			guess = np.append(guess,np.ones(16))
			
			def manglewrap(guess,returnRat=False):
				loglike = 0
				rat,swff,spff = np.array([]),np.array([]),np.array([])
				for sn in self.datadict.keys():
					x0,x1,c,tpkoff = \
						current[self.parlist == 'x0_%s'%sn][0],current[self.parlist == 'x1_%s'%sn][0],\
						current[self.parlist == 'c_%s'%sn][0],current[self.parlist == 'tpkoff_%s'%sn][0]

					if returnRat:
						loglikesingle,ratsingle,swf,spf = \
							mgl.mangle(guess,components,sn,(x0,x1,c,tpkoff),returnRat=returnRat)
						loglike += loglikesingle
						rat = np.append(rat,ratsingle)
						swff = np.append(swff,swf)
						spff = np.append(spff,spf)
					else:
						loglike += mgl.mangle(guess,components,sn,(x0,x1,c,tpkoff),returnRat=returnRat)

				if returnRat:
					log.info(-2*loglike)
					return rat,swff,spff
				else:
					log.info(-2*loglike)
					return -1*loglike


			result = minimize(manglewrap,guess,options={'maxiter':15})
			rat,swf,spf = manglewrap(result.x,returnRat=True)
			M0,M1 = mgl.getmodel(result.x,components,
								 #current[self.parlist == 'm0'],
								 #current[self.parlist == 'm1'],
								 np.array(rat),np.array(swf),np.array(spf))

			fullphase = np.zeros(len(self.phase)*len(self.wave))
			fullwave = np.zeros(len(self.phase)*len(self.wave))
			nwave = len(self.wave)
			count = 0
			for p in range(len(self.phase)):
				for w in range(nwave):
					fullphase[count] = self.phase[p]
					fullwave[count] = self.wave[w]
					count += 1
					
			bsplM0 = bisplrep(fullphase,fullwave,M0.flatten(),kx=3,ky=3,
							  tx=self.phaseknotloc,ty=self.waveknotloc,task=-1)
			bsplM1 = bisplrep(fullphase,fullwave,M1.flatten(),kx=3,ky=3,
							  tx=self.phaseknotloc,ty=self.waveknotloc,task=-1)
			candidate[self.parlist == 'm0'] = bsplM0[2]
			candidate[self.parlist == 'm1'] = bsplM1[2]

			
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
					#import pdb; pdb.set_trace()
				
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
		
		self.GN_iter = {'all':1,'components':1,'all-grouped':1,'x0':1,'x1':1,'component0':1,'piecewisecomponents':1,'piecewisecomponent0':1,'piecewisecomponent1':1,
						'component1':1,'color':1,'colorlaw':1,'spectralrecalibration':1,
						'spectralrecalibration_norm':1,'spectralrecalibration_poly':1,
					    'tpk':1,'modelerr':1}

		self._robustify = False
		self._writetmp = False
		self.chi2_diff_cutoff = 1
		self.fitOptions={}
		self.iModelParam=np.ones(self.npar,dtype=bool)
		self.iModelParam[self.imodelerr]=False
		self.iModelParam[self.imodelcorr]=False
		self.iModelParam[self.iclscat]=False

		self.tryFittingAllParams=True
		fitlist = [('all parameters','all'),('all parameters grouped','all-grouped'),
				   (" x0",'x0'),('both components','components'),('component 0 piecewise','piecewisecomponent0'),('principal component 0','component0'),('x1','x1'),
				   ('component 1 piecewise','piecewisecomponent1'),('principal component 1','component1'),('color','color'),('color law','colorlaw'),
				   ('spectral recalibration const.','spectralrecalibration_norm'),('all spectral recalibration','spectralrecalibration'),
				   ('spectral recalibration higher orders','spectralrecalibration_poly'),
				   ('time of max','tpk'),('error model','modelerr')]

		for message,fit in fitlist:
			if 'all' in fit:
				includePars=np.ones(self.npar,dtype=bool)
			else:
				includePars=np.zeros(self.npar,dtype=bool)
				if 'components' in fit:
					includePars[self.im0]=True
					includePars[self.im1]=True
				elif 'component0' in fit :
					includePars[self.im0]=True
				elif 'component1' in fit:
					includePars[self.im1]=True
				elif fit=='sn':
					includePars[self.ix0]=True
					includePars[self.ix1]=True
				elif fit=='x0':
					includePars[self.ix0]=True
				elif fit=='x1':
					includePars[self.ix1]=True
				elif fit=='color':
					includePars[self.ic]=True
				elif fit=='colorlaw':
					includePars[self.iCL]=True
				elif fit=='tpk':
					includePars[self.itpk]=True
				elif fit=='spectralrecalibration':
					includePars[self.ispcrcl]=True
				elif fit=='spectralrecalibration_norm':
					includePars[self.ispcrcl_norm]=True
				elif fit=='spectralrecalibration_poly':
					includePars[self.ispcrcl_poly]=True
				elif fit=='modelerr':
					includePars[self.imodelerr]=True
					includePars[self.imodelcorr]=True
					includePars[self.parlist=='clscat']=True
				else:
					raise NotImplementedError("""This option for a Gauss-Newton fit with a 
	restricted parameter set has not been implemented: {}""".format(fit))
			self.fitOptions[fit]=(message,includePars)
		self.fitlist = [('all'),#('all parameters grouped','all-grouped'),
			#('piecewise both components','piecewisecomponents'),
			('x0'),('component0'),
			('component1'),('x1'),
			('color'),('colorlaw'),
			('spectralrecalibration'),
			
			('tpk')]

	def convergence_loop(self,guess,loop_niter=3):
		lastResid = 1e20
		log.info('Initializing')
		

		#if self.fit_model_err: fixUncertainty = False
		#else: fixUncertainty = True
		
		if len(self.usePriors) != len(self.priorWidths):
			raise RuntimeError('length of priors does not equal length of prior widths!')
		stepsizes=None
		#residuals = self.lsqwrap(guess,uncertainties,False,False,doPriors=False)
		storedResults={}
		residuals = self.lsqwrap(guess,storedResults)
		self.uncertaintyKeys={key for key in storedResults if key.startswith('photvariances_') or key.startswith('specvariances_') or key.startswith('photCholesky_') }
		uncertainties={key:storedResults[key] for key in self.uncertaintyKeys}
		chi2_init = (residuals**2.).sum()
		X = copy.deepcopy(guess[:])
		Xlast = copy.deepcopy(guess[:])
		log.info('Estimating supernova parameters x0,x1,c and spectral normalization')
		for fit in ['x0','spectralrecalibration_norm','color','x1']:
			X,chi2_init=self.process_fit(X,self.fitOptions[fit][1],uncertainties.copy(),fit=fit)
		chi2results=self.getChi2Contributions(X,uncertainties.copy())
		for name,chi2component,dof in chi2results:
			log.info('{} chi2/dof is {:.1f} ({:.2f}% of total chi2)'.format(name,chi2component/dof,chi2component/chi2_init*100))

		log.info('starting loop; %i iterations'%loop_niter)
		for superloop in range(loop_niter):
			try:
				X,chi2,converged = self.robust_process_fit(X,uncertainties.copy(),chi2_init,superloop)
				chi2results=self.getChi2Contributions(X,uncertainties.copy())
				for name,chi2component,dof in chi2results:
					log.info('{} chi2/dof is {:.1f} ({:.2f}% of total chi2)'.format(name,chi2component/dof,chi2component/chi2*100))
					if name.lower()=='photometric':
						photochi2perdof=chi2component/dof
				if chi2_init-chi2 < -1.e-6:
					log.warning("MESSAGE WARNING chi2 has increased")
				elif np.abs(chi2_init-chi2) < self.chi2_diff_cutoff:
					xfinal,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
						modelerr,clpars,clerr,clscat,SNParams = \
						self.getParsGN(X)
					stepsizes = self.getstepsizes(X,Xlast)
					return xfinal,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
						modelerr,clpars,clerr,clscat,SNParams,stepsizes
				if self.fit_model_err and photochi2perdof<60:
					log.info('Optimizing model error')
					X,loglike=self.minuitOptimize(X,'modelerr')
					residuals = self.lsqwrap(X,storedResults)
					uncertainties={key:storedResults[key] for key in self.uncertaintyKeys}

				log.info('finished iteration %i, chi2 improved by %.1f'%(superloop+1,chi2_init-chi2))
				if converged:
					log.info('Gauss-Newton optimizer could not further improve chi2')
					break
			except KeyboardInterrupt:
				if query_yes_no("Terminate optimization loop and begin writing output?"):
					break
			chi2_init = chi2
			stepsizes = self.getstepsizes(X,Xlast)
			Xlast = copy.deepcopy(X)
		#Retranslate x1, M1, x0, M0 to obey definitions
		Xredefined=self.priors.satisfyDefinitions(X,self.SALTModel(X))
		
		xfinal,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams = \
			self.getParsGN(Xredefined)


		return xfinal,X,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams,stepsizes
		
		#raise RuntimeError("convergence_loop reached 100000 iterations without convergence")
	
	def minuitOptimize(self,X,fit='all'):
		includePars=self.fitOptions[fit][1] 
		def fn(Y):
			if len(Y[Y != Y]):
				import pdb; pdb.set_trace()
			Xnew=X.copy()
			Xnew[includePars]=Y
			return - self.maxlikefit(Xnew)
		def grad(Y):
			if len(Y[Y != Y]):
				import pdb; pdb.set_trace()
			Xnew=X.copy()
			Xnew[includePars]=Y
			#log.info(self.maxlikefit(Xnew,computeDerivatives=True)[1])
			#import pdb; pdb.set_trace()
			return - self.maxlikefit(Xnew,computeDerivatives=True)[1]
		log.info('Initialized log likelihood: {:.2f}'.format(self.maxlikefit(X)))
		params=['x'+str(i) for i in range(includePars.sum())]
		initVals=X[includePars].copy()
		#import pdb;pdb.set_trace()
		#kwargs={'limit_'+params[i] : self.bounds[np.where(includePars)[0][i]] for i in range(includePars.sum()) if }
		kwargs=({params[i]: initVals[i] for i in range(includePars.sum())})
		kwargs.update({'error_'+params[i]: np.abs(X[includePars][i])/10 for i in range(includePars.sum())})
		kwargs.update({'limit_'+params[i]: (-1,1) for i in np.where(self.parlist[includePars] == 'clscat')[0]})
		kwargs.update({'limit_'+params[i]: (0,100) for i in np.where(self.parlist[includePars] == 'modelerr_0')[0]})
		kwargs.update({'limit_'+params[i]: (0,100) for i in np.where(self.parlist[includePars] == 'modelerr_1')[0]})
		m=Minuit(fn,use_array_call=True,forced_parameters=params,grad=grad,errordef=1,**kwargs)
		result,paramResults=m.migrad(includePars.sum()*6)
		X=X.copy()
		
		X[includePars]=np.array([x.value for x  in paramResults])

		# 		if np.allclose(X[includePars],initVals):
# 			import pdb;pdb.set_trace()
		log.info('Final log likelihood: {:.2f}'.format( -result.fval))
		
		return X,-result.fval


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
			photresids+=[photresidsdict['resid']]
			specresids+=[specresidsdict['resid']]


		priorResids,priorVals,priorJac=self.priors.priorResids(self.usePriors,self.priorWidths,guess)
		priorResids=[priorResids]

		BoundedPriorResids,BoundedPriorVals,BoundedPriorJac = \
			self.priors.BoundedPriorResids(self.bounds,self.boundedParams,guess)
		priorResids+=[BoundedPriorResids]

		if self.regularize:
			for regularization, weight,regKey in [(self.phaseGradientRegularization, self.regulargradientphase,'regresult_phase'),
										   (self.waveGradientRegularization,self.regulargradientwave,'regresult_wave' ),
										   (self.dyadicRegularization,self.regulardyad,'regresult_dyad')]:
				if weight ==0:
					continue
				if regKey in storedResults:
					priorResids += storedResults[regKey]
				else:
					for regResids,regJac in zip( *regularization(guess,storedResults,varyParams)):
						priorResids += [regResids*np.sqrt(weight)]
					storedResults[regKey]=priorResids[-self.n_components:]
		chi2Results=[]
		for name,x in [('Photometric',photresids),('Spectroscopic',specresids),('Prior',priorResids)]:
			x=np.concatenate(x)
			chi2Results+=[(name,(x**2).sum(),x.size)]
		return chi2Results
	
	def lsqwrap(self,guess,storedResults,varyParams=None,doPriors=True):
		if varyParams is None:
			varyParams=np.zeros(self.npar,dtype=bool)
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
		if varyParams[self.imodelerr].any() or varyParams[self.imodelcorr].any() or varyParams[self.iclscat].any():
			raise ValueError('lsqwrap not allowed to handle varying model uncertainties')
		
		residuals = []
		jacobian = [] # Jacobian matrix from r

		for sn in self.datadict.keys():
			photresidsdict,specresidsdict=self.ResidsForSN(
				guess,sn,storedResults,varyParams,fixUncertainty=True)
			residuals+=[photresidsdict['resid'],specresidsdict['resid']]
			jacobian+=[sparse.coo_matrix(photresidsdict['resid_jacobian']),sparse.coo_matrix(specresidsdict['resid_jacobian'])]
			

		if doPriors:

			priorResids,priorVals,priorJac=self.priors.priorResids(self.usePriors,self.priorWidths,guess)
			residuals+=[priorResids]
			jacobian+=[priorJac[:,varyParams]]

			BoundedPriorResids,BoundedPriorVals,BoundedPriorJac = \
				self.priors.BoundedPriorResids(self.bounds,self.boundedParams,guess)
			residuals+=[BoundedPriorResids]
			jacobian+=[sparse.coo_matrix(BoundedPriorJac[:,varyParams])]

		if self.regularize:
			for regularization, weight,regKey in [(self.phaseGradientRegularization, self.regulargradientphase,'regresult_phase'),
										   (self.waveGradientRegularization,self.regulargradientwave,'regresult_wave' ),
										   (self.dyadicRegularization,self.regulardyad,'regresult_dyad')]:
				if weight ==0:
					continue
				if regKey in storedResults and not (varyParams[self.im0].any() or varyParams[self.im1].any()):
					residuals += storedResults[regKey]
					jacobian +=  [sparse.coo_matrix(np.zeros((r.size,varyParams.sum())) )for r in storedResults[regKey]]
				else:
					for regResids,regJac in zip( *regularization(guess,storedResults,varyParams)):
						residuals += [regResids*np.sqrt(weight)]
						jacobian+=[sparse.coo_matrix(regJac)*np.sqrt(weight)]
					storedResults[regKey]=residuals[-self.n_components:]

		if varyParams.any():
			return np.concatenate(residuals),sparse.vstack(jacobian)
		else:
			return  np.concatenate(residuals)

	def robust_process_fit(self,X_init,uncertainties,chi2_init,niter):
		X,chi2=X_init,chi2_init
		storedResults=uncertainties.copy()
		for fit in self.fitlist:
			if 'all-grouped' in fit :continue #
			if 'modelerr' in fit: continue
			if 'tpk'==fit and not self.fitTpkOff: continue
			log.info('fitting '+self.fitOptions[fit][0])
			

			Xprop = X.copy()

			if (fit=='all'):
				if self.tryFittingAllParams:
					Xprop,chi2prop = self.process_fit(Xprop,self.fitOptions[fit][1],storedResults,fit=fit)
					if (chi2prop/chi2 < 0.9):
						log.info('Terminating iteration {}, continuing with all parameter fit'.format(niter+1))
						return Xprop,chi2prop,False
					elif (chi2prop<chi2):
						if chi2prop>chi2*(1-1e-3):
							self.tryFittingAllParams=False
							log.info('Discontinuing all parameter fit')
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
						log.info('fitting phase %.1f'%p)
						indices=np.arange((self.waveknotloc.size-4)*(self.phaseknotloc.size-4))
						iFit= (((i-1)*(self.waveknotloc.size-4)) <= indices) & (indices <((i+2)*(self.waveknotloc.size-4)))
						includeParsPhase=np.zeros(self.npar,dtype=bool)

						if fit== 'piecewisecomponents':
							includeParsPhase[self.im0[iFit]]=True
							includeParsPhase[self.im1[iFit]]=True
						else:
							includeParsPhase[self.__dict__['im%s'%fit[-1]][iFit]] = True
						Xprop,chi2prop = self.process_fit(Xprop,includeParsPhase,storedResults,fit=fit)

						#includeParsPhase[self.__dict__['im%s'%fit[-1]][iFit]] = True
						#Xprop,chi2prop = self.process_fit(X,includeParsPhase,storedResults,fit=fit)

						if np.isnan(Xprop).any() or (~np.isfinite(Xprop)).any() or np.isnan(chi2prop) or ~np.isfinite(chi2prop):
							log.error('NaN detected, breaking out of loop')
							break;
						if chi2prop<chi2 :
							X,chi2=Xprop,chi2prop
						retainPCDerivs=True
						storedResults= {key:storedResults[key] for key in storedResults if (key in self.uncertaintyKeys) or
							   (retainPCDerivs and key.startswith('pcDeriv_'   )) }

			else:
				for i in range(self.GN_iter[fit]):
					Xprop,chi2prop = self.process_fit(Xprop,self.fitOptions[fit][1],storedResults,fit=fit)
					if np.isnan(Xprop).any() or (~np.isfinite(Xprop)).any() or np.isnan(chi2prop) or ~np.isfinite(chi2prop):
						log.error('NaN detected, breaking out of loop')
						break;
					if chi2prop<chi2 :
						X,chi2=Xprop,chi2prop
					retainReg=(not ('all' in fit or 'component' in fit))
					retainPCDerivs=('component' in fit)  or fit.startswith('x')
					storedResults= {key:storedResults[key] for key in storedResults if (key in self.uncertaintyKeys) or
							(retainReg and key.startswith('regresult' )) or
						   (retainPCDerivs and key.startswith('pcDeriv_'   )) }

					if chi2 != chi2 or chi2 == np.inf:
						break
						


		#In this case GN optimizer can do no better
		return X,chi2,(X is X_init)
		 #_init
	def linear_fit(self,X,gaussnewtonstep,uncertainties):
		def opFunc(x,stepdir):
			return ((self.lsqwrap(X-(x*stepdir),uncertainties.copy(),None,True))**2).sum()
		result,step,stepType=minimize_scalar(opFunc,args=(gaussnewtonstep,)),gaussnewtonstep,'Gauss-Newton'
		log.info('Linear optimization factor is {:.2f} x {} step'.format(result.x,stepType))
		log.info('Linear optimized chi2 is {:.2f}'.format(result.fun))
		return result.x*gaussnewtonstep,result.fun
		
		
	def process_fit(self,X,iFit,storedResults,fit='all',doPriors=True):
		X=X.copy()
		varyingParams=iFit&self.iModelParam
		if not self.fitTpkOff: varyingParams[self.itpk]=False
		
		residuals,jacobian=self.lsqwrap(X,storedResults,varyingParams,doPriors)
		oldChi=(residuals**2).sum()
		
		#Exclude any parameters that are not currently affecting the fit (column in jacobian zeroed for that index)
		jacobian=jacobian.tocsc()
		includePars= np.diff(jacobian.indptr) != 0
		if not includePars.all():
			varyingParams[varyingParams]=varyingParams[varyingParams] & includePars
			jacobian=jacobian[:,includePars]		
		
		#Exclude any residuals unaffected by the current fit (column in jacobian zeroed for that index)
		jacobian=jacobian.tocsr()
		includeResids=np.diff(jacobian.indptr) != 0
		if not includeResids.all():
			jacobian=jacobian[includeResids,:]		

		log.info('Number of parameters fit this round: {}'.format(includePars.sum()))
		log.info('Initial chi2: {:.2f} '.format(oldChi))
		isJacobianSparse=jacobian.nnz/(jacobian.shape[0]*jacobian.shape[1])<0.5
		#If this is a sparse matrix, treat it as such, otherwise use standard linear algebra solver
		if isJacobianSparse:
		
			log.info('Using sparse linear algebra')
			result=sprslinalg.lsmr(jacobian,residuals[includeResids],atol=1e-6,btol=1e-6,maxiter=2*min(jacobian.shape))
			if result[1]==7: log.warning('Gauss-Newton solver reached max # of iterations')
			stepsize=result[0]
		else:
			log.info('Using dense linear algebra')
			jacobian=jacobian.toarray()
			stepsize=linalg.lstsq(jacobian,residuals[includeResids],cond=self.conditionNumber)[0]
			
		if np.any(np.isnan(stepsize)):
			log.error('NaN detected in stepsize; exitting to debugger')
			import pdb;pdb.set_trace()
		
		gaussNewtonStep=np.zeros(X.size)
		gaussNewtonStep[varyingParams]=stepsize
		uncertainties={key:storedResults[key] for key in self.uncertaintyKeys}
		#Was trying a clip in linear_fit; may not be worth it with new, more stable algorithm
		preclip=((self.lsqwrap(X-gaussNewtonStep,uncertainties.copy(),None,True))**2).sum()
		log.info('After Gauss-Newton chi2 is {:.2f}'.format(preclip))
	
		linearStep,linearChi2=self.linear_fit(X,gaussNewtonStep,uncertainties)
		if linearChi2>preclip:
			X-=gaussNewtonStep
			chi2=preclip
		else:
			X-=linearStep
			chi2=linearChi2
		log.info('Chi2 diff, % diff')
		log.info(' '.join(['{:.2f}'.format(x) for x in [oldChi-chi2,(100*(oldChi-chi2)/oldChi)] ]))
		print('')
		#import pdb; pdb.set_trace()
		return X,chi2

