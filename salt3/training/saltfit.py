#!/usr/bin/env python

import numpy as np
from scipy.interpolate import splprep,splev,BSpline,griddata,bisplev,bisplrep,interp1d,interp2d
from scipy.integrate import trapz, simps
from salt3.util.synphot import synphot
from sncosmo.salt2utils import SALT2ColorLaw
import time
from itertools import starmap
from salt3.training import init_hsiao,saltresids #_master as saltresids
from sncosmo.models import StretchSource
from scipy.optimize import minimize, least_squares
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
import pylab as plt
from scipy.special import factorial
from astropy.cosmology import Planck15 as cosmo
from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
import extinction
from multiprocessing import Pool, get_context
import copy
import scipy.stats as ss
from numpy.random import standard_normal
from scipy.linalg import cholesky
from emcee.interruptible_pool import InterruptiblePool
from sncosmo.utils import integration_grid
from numpy.linalg import inv,pinv


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
		x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams = \
			gn.convergence_loop(
				guess,loop_niter=gaussnewton_maxiter)

		return x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams,'Gauss-Newton MCMC was successful'

		
	def mcmc(self,saltfitter,guess,
			 n_processes,n_mcmc_steps,
			 n_burnin_mcmc):

		saltfitter.debug = False
		if n_processes > 1:
			with InterruptiblePool(n_processes) as pool:
		#	with multiprocessing.Pool(n_processes) as pool:
				x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
					modelerr,clpars,clerr,clscat,SNParams = \
					saltfitter.mcmcfit(guess,n_mcmc_steps,n_burnin_mcmc,pool=pool)
		else:
			x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
				modelerr,clpars,clerr,clscat,SNParams = \
				saltfitter.mcmcfit(guess,n_mcmc_steps,n_burnin_mcmc,pool=None)

		return x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams,'Adaptive MCMC was successful'


class mcmc(saltresids.SALTResids):
	def __init__(self,guess,datadict,parlist,chain=[],loglikes=[],**kwargs):
		self.loglikes=loglikes
		self.chain=chain

		super().__init__(guess,datadict,parlist,**kwargs)
		
		
	def get_proposal_cov(self, M2, n, beta=0.25):
		d, _ = M2.shape
		init_period = self.nsteps_before_adaptive
		s_0, s_opt, C_0 = self.AMpars['sigma_0'], self.AMpars['sigma_opt'], self.AMpars['C_0']
		if n<= init_period or np.random.rand()<=beta:
			return np.sqrt(C_0), False
		else:
			# We can always divide M2 by n-1 since n > init_period
			return np.sqrt((s_opt/(self.nsteps_adaptive_memory - 1))*M2), True
	
	def generate_AM_candidate(self, current, M2, n):
		prop_std,adjust_flag = self.get_proposal_cov(M2, n)
		
		#tstart = time.time()
		candidate = np.zeros(self.npar)
		candidate = np.random.normal(loc=current,scale=np.diag(prop_std))
		for i,par in zip(range(self.npar),self.parlist):
			if self.adjust_snpars and (par == 'm0' or par == 'm1' or par == 'modelerr'):
				candidate[i] = current[i]
			elif self.adjust_modelpars and par != 'm0' and par != 'm1' and par != 'modelerr':
				candidate[i] = current[i]
			else:
				if par == 'modelerr' or par.startswith('x0') or par == 'm0' or par == 'clscat':
					candidate[i] = current[i]*10**(0.4*np.random.normal(scale=prop_std[i,i]))
				#else:
				#	candidate[i] = np.random.normal(loc=current[i],scale=prop_std[i,i])
		#print(time.time()-tstart)
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
			print('using scipy minimizer to find SN params...')		
			components = self.SALTModel(candidate)
			print('error hack!')
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
			print('using scipy minimizer to find M0...')
				
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
				
				print(np.sum(logmin*2))
				return logmin

			guess = candidate[self.parlist == 'm0']
			result = least_squares(lsqwrap,guess,max_nfev=6)
			candidate[self.parlist == 'm0'] = result.x
			self.lsqfit = False
			
		elif M1:
			print('using scipy minimizer to find M1...')
			
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
				
				print(np.sum(logmin*2))
				return logmin

			guess = candidate[self.parlist == 'm0']
			result = least_squares(lsqwrap,guess,max_nfev=6)
			candidate[self.parlist == 'm0'] = result.x
			self.lsqfit = False

		elif doMangle:
			print('mangling!')
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
					print(-2*loglike)
					return rat,swff,spff
				else:
					print(-2*loglike)
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

	def get_propcov_init(self,x):
		C_0 = np.zeros([len(x),len(x)])
		for i,par in zip(range(self.npar),self.parlist):
			if par == 'm0':
				C_0[i,i] = self.stepsize_magscale_M0**2.
			if par == 'modelerr':
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

		self.AMpars = {'C_0':C_0,
					   'sigma_0':0.1/np.sqrt(self.npar),
					   'sigma_opt':2.38*self.adaptive_sigma_opt_scale/np.sqrt(self.npar)}
	
	def update_moments(self,mean, M2, sample, n):
		next_n = (n + 1)
		w = 1/next_n
		new_mean = mean + w*(sample - mean)
		delta_bf, delta_af = sample - mean, sample - new_mean
		new_M2 = M2 + np.outer(delta_bf, delta_af)
		
		return new_mean, new_M2
	
	def mcmcfit(self,x,nsteps,nburn,pool=None,debug=False,thin=1):
		npar = len(x)
		self.npar = npar
		
		# initial log likelihood
		if self.chain==[]:
			self.chain+=[x]
		if self.loglikes==[]:
			self.loglikes += [self.maxlikefit(x,pool=pool,debug=debug)]
		self.M0stddev = np.std(x[self.parlist == 'm0'])
		self.M1stddev = np.std(x[self.parlist == 'm1'])
		self.errstddev = self.stepsize_magscale_err
		mean, M2 = x[:], np.zeros([len(x),len(x)])
		mean_recent, M2_recent = x[:], np.zeros([len(x),len(x)])
		
		self.get_propcov_init(x)

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

			#X = self.lsqguess(current=self.chain[-1],doMangle=True)
			if self.use_lsqfit:
				if not (nstep+1) % self.nsteps_between_lsqfit:
					X = self.lsqguess(current=self.chain[-1],snpars=True)
				if not (nstep) % self.nsteps_between_lsqfit:
					X = self.lsqguess(current=self.chain[-1],doMangle=True)
				else:
					X = self.generate_AM_candidate(current=self.chain[-1], M2=M2_recent, n=nstep)
				#elif not (nstep-3) % self.nsteps_between_lsqfit:
				#	X = self.lsqguess(current=Xlast,M1=True)
			else:
				X = self.generate_AM_candidate(current=self.chain[-1], M2=M2_recent, n=nstep)

			# loglike
			this_loglike = self.maxlikefit(X,pool=pool,debug=debug)

			# accepted?
			accept_bool = self.accept(self.loglikes[-1],this_loglike)
			if accept_bool:
				if not nstep % thin:

					self.chain+=[X]
				self.loglikes+=[this_loglike]

				accept += 1
				print('step = %i, accepted = %i, acceptance = %.3f, recent acceptance = %.3f'%(
					nstep,accept,accept/float(nstep),accept_frac_recent))
			else:
				if not nstep % thin:

					self.chain+=[self.chain[-1]]
				self.loglikes += [self.loglikes[-1]]

			accepted_history = np.append(accepted_history,accept_bool)
			if not (nstep) % self.nsteps_between_lsqfit:
				self.updateEffectivePoints(self.chain[-1])
			mean, M2 = self.update_moments(mean, M2, self.chain[-1], n_adaptive)
			if not n_adaptive % self.nsteps_adaptive_memory:
				n_adaptive = 0
				ix,iy = np.where(M2 < 1e-5)
				iLow = np.where(ix == iy)[0]
				M2[ix[iLow],iy[iLow]] = 1e-5
				# maybe too hacky
				
				if self.adjust_snpars and 'M2_snpars' in self.__dict__.keys(): M2_recent = copy.deepcopy(self.M2_snpars)
				elif self.adjust_snpars and 'M2_snpars' not in self.__dict__.keys(): M2_recent = copy.deepcopy(self.M2_allpars)
				elif self.adjust_modelpars and 'M2_modelpars' in self.__dict__.keys(): M2_recent = copy.deepcopy(self.M2_modelpars)
				elif self.adjust_modelpars and 'M2_modelpars' not in self.__dict__.keys(): M2_recent = copy.deepcopy(self.M2_allpars)
				else:
					M2_recent = copy.deepcopy(M2)
					self.M2_allpars = copy.deepcopy(M2)
					
				mean_recent = mean[:]
				mean, M2 = self.chain[-1][:], np.zeros([len(x),len(x)])
				if self.adjust_snpars: self.M2_snpars = copy.deepcopy(M2_recent)
				elif self.adjust_modelpars: self.M2_modelpars = copy.deepcopy(M2_recent)

		print('acceptance = %.3f'%(accept/float(nstep)))
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
	def __init__(self,guess,datadict,parlist,chain=[],loglikes=[],**kwargs):
		self.loglikes=loglikes
		self.chain=chain
		self.warnings = []
		self.debug=False
		super().__init__(guess,datadict,parlist,**kwargs)
		self.lsqfit = False
		
		self._robustify = False
		self._writetmp = False
		self.chi2_diff_cutoff = 1

	def addwarning(self,warning):
		print(warning)
		self.warnings.append(warning)
		
	def convergence_loop(self,guess,loop_niter=3):
		lastResid = 1e20
		print('Initializing')

		residuals = self.lsqwrap(guess,False,False,doPriors=True)
		chi2_init = (residuals**2.).sum()
		X = copy.deepcopy(guess[:])

		print('starting loop')
		for superloop in range(loop_niter):
			X,chi2 = self.robust_process_fit(X,chi2_init,superloop,chi2_init)
			
			if chi2_init-chi2 < -1.e-6:
				self.addwarning("MESSAGE WARNING chi2 has increased")
			elif np.abs(chi2_init-chi2) < self.chi2_diff_cutoff:
				xfinal,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
					modelerr,clpars,clerr,clscat,SNParams = \
					self.getParsGN(X)
				return xfinal,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
					modelerr,clpars,clerr,clscat,SNParams

			
			print('finished iteration %i, chi2 improved by %.1f'%(superloop+1,chi2_init-chi2))
			chi2_init = chi2
			
		xfinal,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams = \
			self.getParsGN(X)
		return xfinal,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams
		
		#raise RuntimeError("convergence_loop reached 100000 iterations without convergence")

	def lsqwrap(self,guess,computeDerivatives,computePCDerivs=True,doPriors=True,fit='all'):
		tstart = time.time()

		if self.n_colorscatpars:
			colorScat = True
		else: colorScat = None

		if self.n_colorpars:
			colorLaw = SALT2ColorLaw(self.colorwaverange, guess[self.parlist == 'cl'])
		else: colorLaw = None
		
		components = self.SALTModel(guess)

		numResids=self.num_phot+self.num_spec + (len(self.usePriors) if doPriors else 0)
		if self.regularize:
			numRegResids=sum([ self.n_components*(self.phasebins.size-1) * (self.wavebins.size -1) for weight in [self.regulargradientphase,self.regulargradientwave ,self.regulardyad] if not weight == 0])
			numResids+=numRegResids
			
		residuals = np.zeros( numResids)
		jacobian =	np.zeros((numResids,self.npar)) # Jacobian matrix from r
		
		idx = 0
		for sn in self.datadict.keys():
			photresidsdict,specresidsdict=self.ResidsForSN(guess,sn,components,colorLaw,computeDerivatives,computePCDerivs)
			
			idxp = photresidsdict['photresid'].size

			residuals[idx:idx+idxp] = photresidsdict['photresid']
			if computeDerivatives:
				jacobian[idx:idx+idxp,self.parlist=='cl'] = photresidsdict['dphotresid_dcl']
				jacobian[idx:idx+idxp,self.parlist=='m0'] = photresidsdict['dphotresid_dM0']
				jacobian[idx:idx+idxp,self.parlist=='m1'] = photresidsdict['dphotresid_dM1']

				for snparam in ('x0','x1','c'): #tpkoff should go here
					jacobian[idx:idx+idxp,self.parlist == '{}_{}'.format(snparam,sn)] = photresidsdict['dphotresid_d{}'.format(snparam)]
			idx += idxp

			idxp = specresidsdict['specresid'].size

			residuals[idx:idx+idxp] = specresidsdict['specresid']
			if computeDerivatives:
				jacobian[idx:idx+idxp,self.parlist=='cl'] = specresidsdict['dspecresid_dcl']
				jacobian[idx:idx+idxp,self.parlist=='m0'] = specresidsdict['dspecresid_dM0']
				jacobian[idx:idx+idxp,self.parlist=='m1'] = specresidsdict['dspecresid_dM1']

				for snparam in ('x0','x1','c'): #tpkoff should go here
					jacobian[idx:idx+idxp,self.parlist == '{}_{}'.format(snparam,sn)] = specresidsdict['dspecresid_d{}'.format(snparam)]
			idx += idxp
		print('priors: ',*self.usePriors)

		# priors
		print('hack')
		if not 'hi':
			priorResids,priorVals,priorJac=self.priorResids(self.usePriors,self.priorWidths,guess)
			print(*priorVals)
			#doPriors = False
			if doPriors:
				residuals[idx:idx+priorResids.size]=priorResids
				jacobian[idx:idx+priorResids.size,:]=priorJac
				idx+=priorResids.size
			
		if self.regularize:
			for regularization, weight in [(self.phaseGradientRegularization, self.regulargradientphase),(self.waveGradientRegularization,self.regulargradientwave ),(self.dyadicRegularization,self.regulardyad)]:
				if weight ==0:
					continue
				for regResids,regJac,indices in zip( *regularization(guess,computeDerivatives), [self.im0,self.im1]):
					residuals[idx:idx+regResids.size]=regResids*np.sqrt(weight)
					if computeDerivatives:
						jacobian[idx:idx+regResids.size,indices]=regJac*np.sqrt(weight)
					idx+=regResids.size

		if computeDerivatives:
			print('loop took %i seconds'%(time.time()-tstart))
			return residuals,jacobian
		else:
			return residuals
	
	def robust_process_fit(self,X,chi2_init,niter,chi2_last):

		#print('hack!')
		Xtmp = copy.deepcopy(X)
		if niter == 0: computePCDerivs = True
		else: computePCDerivs = False
		Xtmp,chi2_all = self.process_fit(Xtmp,fit='all',computePCDerivs=True)

		if chi2_init - chi2_all > 1 and chi2_all/chi2_last < 0.9:
			return Xtmp,chi2_all
		else:
			# "basic flipflop"??!?!
			print("fitting x0")
			X,chi2 = self.process_fit(X,fit='x0')
			print("fitting x1")
			X,chi2 = self.process_fit(X,fit='x1')
			print("fitting principal component 0")
			X,chi2 = self.process_fit(X,fit='component0')
			print("fitting principal component 1")
			X,chi2 = self.process_fit(X,fit='component1')
			print("fitting color")
			X,chi2 = self.process_fit(X,fit='color')
			print("fitting color law")
			X,chi2 = self.process_fit(X,fit='colorlaw')

		return X,chi2 #_init
	
	def process_fit(self,X,fit='all',snid=None,doPriors=True,computePCDerivs=False):
		
		#if fit == 'all' or fit == 'components': computePCDerivs = 3
		#elif fit == 'component0': computePCDerivs=1
		#elif fit == 'component1': computePCDerivs=2
		#else: computePCDerivs = 0
		#computePCDerivs = True
		#doPriors=False
		residuals,jacobian=self.lsqwrap(X,True,computePCDerivs,doPriors,fit)
		if fit == 'all':
			includePars=np.ones(self.npar,dtype=bool)
		else:
			includePars=np.zeros(self.npar,dtype=bool)
			if fit=='components':
				includePars[self.im0]=True
				includePars[self.im1]=True
			elif fit=='component0':
				includePars[self.im0]=True
			elif fit=='component1':
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
			else:
				raise NotImplementedError("""This option for a Gaussian Process fit with a 
restricted parameter set has not been implemented: {}""".format(fit))

		#Exclude any parameters that are not currently affecting the fit (column in jacobian zeroed for that index)
		includePars=includePars & ~(np.all(0==jacobian,axis=0))
		
		print('Number of parameters fit this round: {}'.format(includePars.sum()))
		jacobian=jacobian[:,includePars]
		stepsize = np.dot(np.dot(pinv(np.dot(jacobian.T,jacobian)),jacobian.T),
						  residuals.reshape(residuals.size,1)).reshape(includePars.sum())
		#if self.i>4 and fit=='all' and not self.debug: 
		#	import pdb;pdb.set_trace()
		#	self.debug=True
		if np.any(np.isnan(stepsize)):
			print('NaN detected in stepsize; exitting to debugger')
			import pdb;pdb.set_trace()

		X[includePars] -= stepsize

		# quick eval

		chi2 = np.sum(self.lsqwrap(X,False,False,doPriors=doPriors)**2.)
		print("chi2: old, new, diff")
		print((residuals**2).sum(),chi2,(residuals**2).sum()-chi2)
		#if chi2 != chi2:
		#	import pdb; pdb.set_trace()
		
		return X,chi2
	
