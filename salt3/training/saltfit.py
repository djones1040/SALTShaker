#!/usr/bin/env python

import numpy as np
from scipy.interpolate import splprep,splev,BSpline,griddata,bisplev,bisplrep,interp1d,interp2d
from scipy.integrate import trapz, simps
from salt3.util.synphot import synphot
from sncosmo.salt2utils import SALT2ColorLaw
import time
from itertools import starmap
from salt3.training import init_hsiao
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

_SCALE_FACTOR = 1e-12
_B_LAMBDA_EFF = np.array([4302.57])	 # B-band-ish wavelength
_V_LAMBDA_EFF = np.array([5428.55])	 # V-band-ish wavelength

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
					n_burnin_mcmc):

		gn.debug = False
		x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams = \
			gn.convergence_loop(guess)

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

class loglike:
	def __init__(self,guess,datadict,parlist,**kwargs):
		
		assert type(parlist) == np.ndarray
		self.blah = False
		self.debug = False
		self.nstep = 0
		self.parlist = parlist
		self.npar = len(parlist)
		self.datadict = datadict
		#ndata = 0
		#for sn in self.datadict.keys():
		#	photdata = self.datadict[sn]['photdata']
		#	for flt in np.unique(photdata['filt']):
		#		ndata += len(photdata['filt'][photdata['filt'] == flt])
		#self.n_data = ndata

		
		# initialize derivatives for the priors
		self.priorderivdict = {'Bmax':np.zeros(self.npar),
							   'x1mean':np.zeros(self.npar),
							   'x1std':np.zeros(self.npar),
							   'colorlaw':np.zeros(self.npar),
							   'M0end':np.zeros(self.npar),
							   'M1end':np.zeros(self.npar),
							   'tBmax':np.zeros(self.npar)}
		self.npriors = 7
		self.priors = np.array(self.priorderivdict.keys())
		
		self.guess = guess
		self.lsqfit = False
		self.nsn = len(self.datadict.keys())
		
		for key, value in kwargs.items(): 
			self.__dict__[key] = value

		# pre-set some indices
		self.m0min = np.min(np.where(self.parlist == 'm0')[0])
		self.m0max = np.max(np.where(self.parlist == 'm0')[0])
		self.errmin = np.min(np.where(self.parlist == 'modelerr')[0])
		self.errmax = np.max(np.where(self.parlist == 'modelerr')[0])
		self.ix1 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x1')])
		self.ix0 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x0')])
		self.ic	 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('c')])
		self.im0 = np.where(self.parlist == 'm0')[0]
		self.im1 = np.where(self.parlist == 'm1')[0]
		self.iCL = np.where(self.parlist == 'cl')[0]
		
		# set some phase/wavelength arrays
		self.splinecolorwave = np.linspace(self.colorwaverange[0],self.colorwaverange[1],self.n_colorpars)
		self.phasebins = np.linspace(self.phaserange[0],self.phaserange[1],
							 1+ (self.phaserange[1]-self.phaserange[0])/self.phaseres)
		self.wavebins = np.linspace(self.waverange[0],self.waverange[1],
							 1+(self.waverange[1]-self.waverange[0])/self.waveres)

		self.phase = np.linspace(self.phaserange[0],self.phaserange[1],
								 (self.phaserange[1]-self.phaserange[0])/self.phaseoutres,False)
		self.wave = np.linspace(self.waverange[0],self.waverange[1],
								(self.waverange[1]-self.waverange[0])/self.waveoutres,False)
		self.maxPhase=np.where(abs(self.phase) == np.min(abs(self.phase)))[0]
		
		self.splinephase = np.linspace(self.phaserange[0]-self.phaseres*0,
									   self.phaserange[1]+self.phaseres*0,
									   (self.phaserange[1]-self.phaserange[0])/self.phaseres+0*2,False)
		self.splinewave = np.linspace(self.waverange[0]-self.waveres*0,
									  self.waverange[1]+self.waveres*0,
									  (self.waverange[1]-self.waverange[0])/self.waveres+0*2,False)

		
		self.hsiaoflux = init_hsiao.get_hsiao(hsiaofile=self.initmodelfile,Bfilt=self.initbfilt,
											  phaserange=self.phaserange,waverange=self.waverange,
											  phaseinterpres=self.phaseoutres,waveinterpres=self.waveoutres,
											  phasesplineres=self.phaseres,wavesplineres=self.waveres,
											  days_interp=0)

		self.neff=0
		self.updateEffectivePoints(guess)

		# initialize the model
		self.components = self.SALTModel(guess)
		self.salterr = self.ErrModel(guess)
		self.salterr[:] = 1e-10
		
		self.m0guess = -19.49 #10**(-0.4*(-19.49-27.5))
		self.m1guess = 1
		self.extrapolateDecline=0.015
		# set up the filters
		self.stdmag = {}
		self.fluxfactor = {}
		for survey in self.kcordict.keys():
			if survey == 'default': 
				self.stdmag[survey] = {}
				self.bbandoverlap = (self.wave>=self.kcordict['default']['Bwave'].min())&(self.wave<=self.kcordict['default']['Bwave'].max())
				self.bbandpbspl = np.interp(self.wave[self.bbandoverlap],self.kcordict['default']['Bwave'],self.kcordict['default']['Bwave'])
				self.bbandpbspl *= self.wave[self.bbandoverlap]
				self.bbandpbspl /= np.trapz(self.bbandpbspl,self.wave[self.bbandoverlap])*HC_ERG_AA
				self.stdmag[survey]['B']=synphot(
					self.kcordict[survey]['primarywave'],self.kcordict[survey]['AB'],
					filtwave=self.kcordict['default']['Bwave'],filttp=self.kcordict[survey]['Btp'],
					zpoff=0)
				self.kcordict['default']['minlam'] = np.min(self.kcordict['default']['Bwave'][self.kcordict['default']['Btp'] > 0.01])
				self.kcordict['default']['maxlam'] = np.max(self.kcordict['default']['Bwave'][self.kcordict['default']['Btp'] > 0.01])
				self.kcordict['default']['fluxfactor'] = 10**(0.4*(self.stdmag[survey]['B']+27.5))
				
				continue
			self.stdmag[survey] = {}
			self.fluxfactor[survey] = {}
			primarywave = self.kcordict[survey]['primarywave']
			for flt in self.kcordict[survey].keys():
				if flt == 'filtwave' or flt == 'primarywave' or flt == 'snflux' or flt == 'AB' or flt == 'BD17' or flt == 'Vega': continue
				if self.kcordict[survey][flt]['magsys'] == 'AB': primarykey = 'AB'
				elif self.kcordict[survey][flt]['magsys'].upper() == 'VEGA': primarykey = 'Vega'
				elif self.kcordict[survey][flt]['magsys'] == 'BD17': primarykey = 'BD17'
				self.stdmag[survey][flt] = synphot(
					primarywave,self.kcordict[survey][primarykey],
					filtwave=self.kcordict[survey]['filtwave'],
					filttp=self.kcordict[survey][flt]['filttrans'],
					zpoff=0) - self.kcordict[survey][flt]['primarymag']
				self.fluxfactor[survey][flt] = 10**(0.4*(self.stdmag[survey][flt]+27.5))
				self.kcordict[survey][flt]['minlam'] = np.min(self.kcordict[survey]['filtwave'][self.kcordict[survey][flt]['filttrans'] > 0.01])
				self.kcordict[survey][flt]['maxlam'] = np.max(self.kcordict[survey]['filtwave'][self.kcordict[survey][flt]['filttrans'] > 0.01])
				
		#Count number of photometric and spectroscopic points
		self.num_spec=0
		self.num_phot=0
		for sn in self.datadict.keys():
			photdata = self.datadict[sn]['photdata']
			specdata = self.datadict[sn]['specdata']
			survey = self.datadict[sn]['survey']
			filtwave = self.kcordict[survey]['filtwave']
			z = self.datadict[sn]['zHelio']
			
			self.datadict[sn]['mwextcurve'] = np.zeros([len(self.phase),len(self.wave)])
			for i in range(len(self.phase)):
				self.datadict[sn]['mwextcurve'][i,:] = 10**(-0.4*extinction.fitzpatrick99(self.wave*(1+z),self.datadict[sn]['MWEBV']*3.1))

			self.num_spec += sum([specdata[key]['flux'].size for key in specdata])

			for flt in np.unique(photdata['filt']):
				self.num_phot+=(photdata['filt']==flt).sum()

		#Store derivatives of a spline with fixed knot locations with respect to each knot value
		starttime=time.time()
		self.spline_derivs=[ bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,3,3)) for i in range(self.im0.size)]
		print('Time to calculate spline_derivs: %.2f'%(time.time()-starttime))
		
		self.getobswave()

							
	def getobswave(self):
		"for each filter, setting up some things needed for synthetic photometry"
		
		for sn in self.datadict.keys():
			z = self.datadict[sn]['zHelio']
			survey = self.datadict[sn]['survey']
			filtwave = self.kcordict[survey]['filtwave']

			self.datadict[sn]['obswave'] = self.wave*(1+z)
			self.datadict[sn]['obsphase'] = self.phase*(1+z)
			self.datadict[sn]['pbspl'] = {}
			self.datadict[sn]['denom'] = {}
			self.datadict[sn]['idx'] = {}
			self.datadict[sn]['dwave'] = self.wave[1]*(1+z) - self.wave[0]*(1+z)
			for flt in np.unique(self.datadict[sn]['photdata']['filt']):

				filttrans = self.kcordict[survey][flt]['filttrans']

				g = (self.datadict[sn]['obswave'] >= filtwave[0]) & (self.datadict[sn]['obswave'] <= filtwave[-1])	# overlap range
				self.datadict[sn]['idx'][flt] = g
			
				pbspl = np.interp(self.datadict[sn]['obswave'][g],filtwave,filttrans)
				pbspl *= self.datadict[sn]['obswave'][g]
				denom = np.trapz(pbspl,self.datadict[sn]['obswave'][g])
				pbspl /= denom*HC_ERG_AA

				self.datadict[sn]['pbspl'][flt] = pbspl[np.newaxis,:]
				self.datadict[sn]['denom'][flt] = denom

		# rest-frame B
		filttrans = self.kcordict['default']['Btp']
		filtwave = self.kcordict['default']['Bwave']
			
		pbspl = np.interp(self.wave,filtwave,filttrans)
		pbspl *= self.wave
		denom = np.trapz(pbspl,self.wave)
		pbspl /= denom*HC_ERG_AA
		self.kcordict['default']['Bpbspl'] = pbspl
		self.kcordict['default']['Bdwave'] = self.wave[1] - self.wave[0]
				
	def maxlikefit(self,x,pool=None,debug=False,timeit=False):
		"""
		Calculates the likelihood of given SALT model to photometric and spectroscopic data given during initialization
		
		Parameters
		----------
		x : array
			SALT model parameters
			
		pool :	multiprocessing.pool.Pool, optional
			Optional worker pool to be used for calculating chi2 values for each SN. If not provided, all work is done in root process
		
		debug : boolean, optional
			Debug flag
		
		Returns
		-------
		
		chi2: float
			Goodness of fit of model to training data	
		"""
		
		#Set up SALT model
		# HACK
		components = self.SALTModel(x)
		
		salterr = self.ErrModel(x)
		if self.n_colorpars:
			colorLaw = SALT2ColorLaw(self.colorwaverange, x[self.parlist == 'cl'])
		else: colorLaw = None
		if self.n_colorscatpars:
			colorScat = True
		else: colorScat = None

		
		# timing stuff
		if timeit:
			self.tdelt0,self.tdelt1,self.tdelt2,self.tdelt3,self.tdelt4,self.tdelt5 = 0,0,0,0,0,0
		
		chi2 = 0
		#Construct arguments for maxlikeforSN method
		#If worker pool available, use it to calculate chi2 for each SN; otherwise, do it in this process
		args=[(None,sn,x,components,self.salterr,colorLaw,colorScat,debug,timeit) for sn in self.datadict.keys()]
		if pool:
			loglike=sum(pool.map(self.loglikeforSN,args))
		else:
			loglike=sum(starmap(self.loglikeforSN,args))

		loglike -= self.regularizationChi2(x,self.regulargradientphase,self.regulargradientwave,self.regulardyad)

		
		logp = loglike + self.m0prior(components)[0] + self.m1prior(x[self.ix1]) + self.endprior(components)+self.peakprior(x,components)

		if colorLaw:
			logp += self.EBVprior(colorLaw)
			
		self.nstep += 1
		if not self.lsqfit: print(logp*-2)
		else: print(logp.sum()*-2)

		if timeit:
			print('%.3f %.3f %.3f %.3f %.3f'%(self.tdelt0,self.tdelt1,self.tdelt2,self.tdelt3,self.tdelt4))

		return logp

	def m0prior(self,components,m0=False):
		int1d = interp1d(self.phase,components[0],axis=0)
		m0Bflux = np.sum(self.kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
			self.kcordict['default']['Bdwave']*self.kcordict['default']['fluxfactor']

		if m0Bflux > 0:
			m0B = -2.5*np.log10(m0Bflux) + 27.5
		else: m0B = 99
		if m0: return -2.5*np.log10(m0Bflux) + 27.5
		else:
			logprior = norm.logpdf(m0B,self.m0guess,0.1)
			return logprior
	
	def m1prior(self,x1pars):

		logprior = norm.logpdf(np.std(x1pars),self.m1guess,0.02)

		return logprior

	def m1priormean(self,x1pars):

		logprior = norm.logpdf(np.mean(x1pars),0,0.02)

		return logprior

	def m1priorstd(self,x1pars):

		logprior = norm.logpdf(np.std(x1pars),self.m1guess,0.02)

		return logprior
		
	def peakprior(self,x,components):
		wave=self.wave[self.bbandoverlap]
		lightcurve=np.sum(self.bbandpbspl[np.newaxis,:]*components[0][:,self.bbandoverlap],axis=1)
		# from D'Arcy - disabled for now!!	(barfs if model goes crazy w/o regularization)
		#maxPhase=np.argmax(lightcurve)
		#finePhase=np.arange(self.phase[maxPhase-1],self.phase[maxPhase+1],0.1)
		finePhase=np.arange(self.phase[self.maxPhase-1],self.phase[self.maxPhase+1],0.1)
		fineGrid=self.SALTModel(x,evaluatePhase=finePhase,evaluateWave=wave)[0]
		lightcurve=np.sum(self.bbandpbspl[np.newaxis,:]*fineGrid,axis=1)
		logprior = norm.logpdf(finePhase[np.argmax(lightcurve)],0,0.5)
		
		return logprior

	def endprior(self,components):
		
		logprior = norm.logpdf(np.sum(components[0][0,:]),0,0.1)
		logprior += norm.logpdf(np.sum(components[1][0,:]),0,0.1)
		
		return logprior

	def m0endprior(self,components):
		
		logprior = norm.logpdf(np.sum(components[0][0,:]),0,0.1)
		
		return logprior

	def m1endprior(self,components):
		
		logprior = norm.logpdf(np.sum(components[1][0,:]),0,0.1)
		
		return logprior
	
	def EBVprior(self,colorLaw):
		# 0.4*np.log(10) = 0.921
		logpriorB = norm.logpdf(colorLaw(_B_LAMBDA_EFF)[0], 0.0, 0.02)
		logpriorV = norm.logpdf(colorLaw(_V_LAMBDA_EFF)[0], 0.921, 0.02)
		return logpriorB + logpriorV
	
	def specmodelvalsforSN(self,x,sn,components,colorLaw,computeDerivatives,bsorder=3):
		#self,x,sn,specdata,intspecerr1d,int1dM0part,int1dM1part,
		#				   obswave,colorexp,colorlaw,snpars,z,obsphase,M0,components):

		if self.n_components == 1:
			M0 = copy.deepcopy(components[0])
		elif self.n_components == 2:
			M0,M1 = copy.deepcopy(components)
		
		resultsdict={}
		z = self.datadict[sn]['zHelio']
		survey = self.datadict[sn]['survey']
		filtwave = self.kcordict[survey]['filtwave']
		obswave = self.datadict[sn]['obswave']
		obsphase = self.datadict[sn]['obsphase']
		specdata = self.datadict[sn]['specdata']
		pbspl = self.datadict[sn]['pbspl']
		dwave = self.datadict[sn]['dwave']
		idx = self.datadict[sn]['idx']
		nspecdata = sum([specdata[key]['flux'].size for key in specdata])
		
		intspecerr1d = interp1d(obsphase,self.salterr,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
		
		x0,x1,c,tpkoff = x[self.parlist == 'x0_%s'%sn],x[self.parlist == 'x1_%s'%sn],\
						 x[self.parlist == 'c_%s'%sn],x[self.parlist == 'tpkoff_%s'%sn]
		clpars = x[self.parlist == 'cl']
		#x1 = x1 - np.mean(x[self.ix1])
		
		#Calculate spectral model
		M0 *= self.datadict[sn]['mwextcurve']
		M1 *= self.datadict[sn]['mwextcurve']
		
		if colorLaw:
			colorlaw = -0.4 * colorLaw(self.wave)
			colorexp = 10. ** (colorlaw * c)
			M0 *= colorexp; M1 *= colorexp
			
		M0 *= _SCALE_FACTOR/(1+z); M1 *= _SCALE_FACTOR/(1+z)
		int1dM0 = interp1d(obsphase,M0,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
		int1dM1 = interp1d(obsphase,M1,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")

		resultsdict={}
		resultsdict['modelflux'] = np.zeros(nspecdata)
		resultsdict['dataflux'] = np.zeros(nspecdata)
		resultsdict['uncertainty'] =  np.zeros(nspecdata)
		if computeDerivatives:
			resultsdict['dmodelflux_dx0'] = np.zeros((nspecdata,1))
			resultsdict['dmodelflux_dx1'] = np.zeros((nspecdata,1))
			resultsdict['dmodelflux_dc']  = np.zeros((nspecdata,1))
			resultsdict['dmodelflux_dM0'] = np.zeros([nspecdata,len(self.im0)])
			resultsdict['dmodelflux_dM1'] = np.zeros([nspecdata,len(self.im1)])
			resultsdict['dmodelflux_dcl'] = np.zeros([nspecdata,self.n_colorpars])


		iSpecStart = 0
		for k in specdata.keys():
			#import pdb; pdb.set_trace()
			SpecLen = specdata[k]['flux'].size
			phase=specdata[k]['tobs']+tpkoff
			#saltfluxinterp = int1d(phase)
			M0interp = int1dM0(phase)
			M1interp = int1dM1(phase)

			#Check spectrum is inside proper phase range, extrapolate decline if necessary
			if phase < obsphase.min():
				pass
			elif phase > obsphase.max():
				saltfluxinterp*=10**(-0.4* self.extrapolateDecline* (phase-obsphase.max()))

			#Define recalibration factor
			coeffs=x[self.parlist=='specrecal_{}_{}'.format(sn,k)]
			coeffs/=factorial(np.arange(len(coeffs)))
			recalexp = np.exp(np.poly1d(coeffs)((specdata[k]['wavelength']-np.mean(specdata[k]['wavelength']))/self.specrange_wavescale_specrecal))


			M0interp = np.interp(specdata[k]['wavelength'],obswave,M0interp[0])*recalexp
			M1interp = np.interp(specdata[k]['wavelength'],obswave,M1interp[0])*recalexp
			colorexpinterp = np.interp(specdata[k]['wavelength'],obswave,colorexp)
			colorlawinterp = np.interp(specdata[k]['wavelength'],obswave,colorlaw)

			#modulatedFlux = np.interp(specdata[k]['wavelength'],obswave,modulatedFlux[0])*recalexp
			modulatedFlux = x0*(M0interp + x1*M1interp)*recalexp
			
			resultsdict['modelflux'][iSpecStart:iSpecStart+SpecLen] = modulatedFlux
			resultsdict['dataflux'][iSpecStart:iSpecStart+SpecLen] = specdata[k]['flux']

			
			modelerr = np.interp( specdata[k]['wavelength'],obswave,intspecerr1d(phase)[0])
			specvar=(specdata[k]['fluxerr']**2.) # + (1e-3*saltfluxinterp)**2.)
			resultsdict['uncertainty'][iSpecStart:iSpecStart+SpecLen] = specvar + (modelerr*resultsdict['modelflux'][iSpecStart:iSpecStart+SpecLen])**2
			#resultsdict['resids']
			#self.datadict[sn]['specdata'][k]['specvar']
			
			# derivatives....
			if computeDerivatives:
				#self.datadict[sn]['specdata'][k]['colorderiv'] = np.zeros(len(specdata[k]['flux']))
				#self.datadict[sn]['specdata'][k]['colorlawderiv'] = np.zeros([len(specdata[k]['flux']),self.n_colorpars])
				#self.datadict[sn]['specdata'][k]['m0deriv'] = np.zeros([len(specdata[k]['flux']),len(self.im0)])
				#self.datadict[sn]['specdata'][k]['m1deriv'] = np.zeros([len(specdata[k]['flux']),len(self.im1)])
				
				# color
				#ctmp = c + 1e-3
				#modelfluxtmp = (M0interp+M1interp)/colorexpinterp*10.**(colorlawinterp * ctmp)
				#colorderivflux = (modelfluxtmp-saltfluxinterp)/1e-3
				#self.datadict[sn]['specdata'][k]['colorderiv'] = colorderivflux

				#import pdb; pdb.set_trace()
				resultsdict['dmodelflux_dc'][iSpecStart:iSpecStart+SpecLen,0] = x0*(M0interp + x1*M1interp)*np.log(10)*colorlawinterp
				resultsdict['dmodelflux_dx0'][iSpecStart:iSpecStart+SpecLen,0] = (M0interp + x1*M1interp)
				resultsdict['dmodelflux_dx1'][iSpecStart:iSpecStart+SpecLen,0] = x0*M1interp

				
				# color law
				for i in range(self.n_colorpars):
					#x[self.iCL[i]] += 1e-3
					#colorlawtmp = SALT2ColorLaw(self.colorwaverange, x[self.iCL])
					#modelfluxtmp = (M0interp+M1interp)/colorexpinterp*10.**(-0.4 * colorlawtmp(specdata[k]['wavelength']) * c)*recalexp
					#self.datadict[sn]['specdata'][k]['colorlawderiv'][:,i] = (modelfluxtmp-saltfluxinterp)/1e-3/np.sqrt(specvar)
					#x[self.iCL[i]] -= 1e-3
					dcolorlaw_dcli = SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)
					resultsdict['dmodelflux_dcl'][iSpecStart:iSpecStart+SpecLen,i] = modulatedFlux*-0.4*np.log(10)*c*dcolorlaw_dcli(specdata[k]['wavelength']/(1+z))

					
				# M0, M1
				if self.computePCs:
					for j in range(len(self.im0)):
						#Range of wavelength and phase values affected by changes in knot i
						waverange=self.waveknotloc[[i%(self.waveknotloc.size-bsorder-1),i%(self.waveknotloc.size-bsorder-1)+bsorder+1]]
						phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-bsorder-1),i//(self.waveknotloc.size-bsorder-1)+bsorder+1]]
						#Check if this spectrum is inside values affected by changes in knot i
						if waverange[0]*(1+z) > specdata[k]['wavelength'].min() or waverange[1]*(1+z) < specdata[k]['wavelength'].max():
							pass
						#Check which phases are affected by knot i
						inPhase=(phase>phaserange[0]*(1+z) ) & (phase<phaserange[1]*(1+z) )
						if inPhase.any():
							#Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
							#Integrate only over wavelengths within the relevant range
							inbounds=(self.wave[idx[flt]]>waverange[0])	 & (self.wave[idx[flt]]<waverange[1])
							derivInterp = interp1d(phase/(1+z),self.spline_derivs[i][:,inbounds],axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
							derivInterp2 = np.interp(specdata[k]['wavelength'],obswave,derivInterp)*_SCALE_FACTOR/(1+z)*x0*recalexp*colorexpinterp
							
							resultsdict['dmodelflux_dM0'][iSpecStart:iSpecStart+SpecLen,j] = derivInterp2 #*intmult
							
							#Dependence is the same for dM1, except with an extra factor of x1
							resultsdict['dmodelflux_dM1'][iSpecStart:iSpecStart+SpecLen,i] =  resultsdict['dmodelflux_dM0'][iSpecStart:iSpecStart+speclen,i]*x1
					
					if ( (phase>obsphase.max())).any():
						if phase > obsphase.max():
							resultsdict['dmodelflux_dM0'][iSpecStart:iSpecStart+SpecLen,:] *= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))
							resultsdict['dmodelflux_dM1'][iSpecStart:iSpecStart+SpecLen,:] *= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))

						
						#M0deriv = M0derivinterp[j](phase/(1+z))[0]
						# warning : ignoring changes in tpkoff for derivatives
						#M0derivinterp2 = np.interp(specdata[k]['wavelength'],obswave,M0deriv)*_SCALE_FACTOR/(1+z)*x0*recalexp*colorexpinterp
						#M1deriv = M1derivinterp[j](phase/(1+z))[0]
						#M1derivinterp2 = np.interp(specdata[k]['wavelength'],obswave,M1deriv)*_SCALE_FACTOR/(1+z)*x0*x1*recalexp*colorexpinterp

						#resultsdict['dmodelflux_dM0'][iSpecStart:iSpecStart+speclen,j] = M0derivinterp2/np.sqrt(specvar)
						#resultsdict['dmodelflux_dM1'][iSpecStart:iSpecStart+speclen,j] = M1derivinterp2/np.sqrt(specvar)
						#self.datadict[sn]['specdata'][k]['m0deriv'][:,j] = M0derivinterp2/np.sqrt(specvar)
						#self.datadict[sn]['specdata'][k]['m1deriv'][:,j] = M1derivinterp2/np.sqrt(specvar)

			#intm01d = interp1d(obsphase,M0,axis=0,kind='nearest')
			#self.datadict[sn]['specdata'][k]['m1flux'] = M1interp
			#self.datadict[sn]['specdata'][k]['modelflux'] = saltfluxinterp #np.interp(specdata[k]['wavelength'],obswave,intm01d(phase/(1+z))[0])
			
			#self.datadict[sn]['specdata'][k]['residuals'] = (specdata[k]['modelflux']-specdata[k]['flux'])/np.sqrt(specvar)*self.num_phot/self.num_spec
			#if self.compute_derivatives: import pdb; pdb.set_trace()

			#self.datadict[sn]['specdata'][k]['dmodelflux_dx0'] = specdata[k]['modelflux']/np.sqrt(specvar)/x0*self.num_phot/self.num_spec
			#self.datadict[sn]['specdata'][k]['dmodelflux_dx1'] = specdata[k]['modelflux']/np.sqrt(specvar)/x1*(1-1/self.nsn)*self.num_phot/self.num_spec
			#self.datadict[sn]['specdata'][k]['dmodelflux_dc'] = specdata[k]['colorderiv']/np.sqrt(specvar)*self.num_phot/self.num_spec
			#self.datadict[sn]['specdata'][k]['dmodelflux_dcl'] = specdata[k]['colorlawderiv']*self.num_phot/self.num_spec
			#self.datadict[sn]['specdata'][k]['dmodelflux_dM0'] = specdata[k]['m0deriv']*self.num_phot/self.num_spec
			#self.datadict[sn]['specdata'][k]['dmodelflux_dM1'] = specdata[k]['m1deriv']*self.num_phot/self.num_spec
			iSpecStart += SpecLen
		return resultsdict
			
	def photResidsForSN(self,x,sn,components,colorLaw,computeDerivatives,bsorder=3):
		modeldict=self.photmodelvalsforSN(x,sn,components,colorLaw,computeDerivatives,bsorder)
		residsdict={key.replace('dmodelflux','dphotresid'):modeldict[key]/(modeldict['uncertainty'][:,np.newaxis]) for key in modeldict if 'modelflux' in key}
		residsdict['photresid']=(modeldict['modelflux']-self.datadict[sn]['photdata']['fluxcal'])/modeldict['uncertainty']
		residsdict['weights']=1/modeldict['uncertainty']
		return residsdict

	def specResidsForSN(self,x,sn,components,colorLaw,computeDerivatives,bsorder=3):
		modeldict=self.specmodelvalsforSN(x,sn,components,colorLaw,computeDerivatives,bsorder)

		#residsdict = {}
		#for key in modeldict:
		#	if 'modelflux' in key:
		#		import pdb; pdb.set_trace()
		#		residsdict[key.replace('dmodelflux','dspecresid')] = modeldict[key]/(modeldict['uncertainty'][:,np.newaxis])
		
		residsdict={key.replace('dmodelflux','dspecresid'):modeldict[key]/(modeldict['uncertainty'][:,np.newaxis])*self.num_phot/self.num_spec for key in modeldict if 'modelflux' in key}

		residsdict['specresid']=(modeldict['modelflux']-modeldict['dataflux'])/modeldict['uncertainty']*self.num_phot/self.num_spec
		residsdict['weights']=1/modeldict['uncertainty']
		return residsdict
	
		
	def photmodelvalsforSN(self,x,sn,components,colorLaw,computeDerivatives,bsorder=3):
		if self.n_components == 1:
			M0 = copy.deepcopy(components[0])
		elif self.n_components == 2:
			M0,M1 = copy.deepcopy(components)
		
		z = self.datadict[sn]['zHelio']
		survey = self.datadict[sn]['survey']
		filtwave = self.kcordict[survey]['filtwave']
		obswave = self.datadict[sn]['obswave'] #self.wave*(1+z)
		obsphase = self.datadict[sn]['obsphase'] #self.phase*(1+z)
		photdata = self.datadict[sn]['photdata']
		pbspl = self.datadict[sn]['pbspl']
		dwave = self.datadict[sn]['dwave']
		idx = self.datadict[sn]['idx']

		interr1d = interp1d(obsphase,self.salterr,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
		
		x0,x1,c,tpkoff = x[self.parlist == 'x0_%s'%sn],x[self.parlist == 'x1_%s'%sn],\
						 x[self.parlist == 'c_%s'%sn],x[self.parlist == 'tpkoff_%s'%sn]
		clpars = x[self.parlist == 'cl']
		#x1 = x1 - np.mean(x[self.ix1])
		
		#Calculate spectral model
		M0 *= self.datadict[sn]['mwextcurve']
		M1 *= self.datadict[sn]['mwextcurve']
		
		if colorLaw:
			colorlaw = -0.4 * colorLaw(self.wave)
			colorexp = 10. ** (colorlaw * c)
			M0 *= colorexp; M1 *= colorexp
			
		M0 *= _SCALE_FACTOR/(1+z); M1 *= _SCALE_FACTOR/(1+z)
		int1dM0 = interp1d(obsphase,M0,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
		int1dM1 = interp1d(obsphase,M1,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")

		# spectra
		#intspecerr1d=interr1d
		#self.specmodelvalsforSN(x,sn,self.datadict[sn]['specdata'],intspecerr1d,int1dM0,int1dM1,
		#						obswave,colorexp,colorlaw,(x0,x1,c,tpkoff),z,obsphase,M0,components)

		resultsdict={}
		resultsdict['modelflux'] = np.zeros(len(photdata['filt']))
		resultsdict['uncertainty'] =  np.zeros(len(photdata['filt']))
		if computeDerivatives:
			resultsdict['dmodelflux_dx0'] = np.zeros((photdata['filt'].size,1))
			resultsdict['dmodelflux_dx1'] = np.zeros((photdata['filt'].size,1))
			resultsdict['dmodelflux_dc']  = np.zeros((photdata['filt'].size,1))
			resultsdict['dmodelflux_dM0'] = np.zeros([photdata['filt'].size,len(self.im0)])#*1e-6 #+ 1e-5
			resultsdict['dmodelflux_dM1'] = np.zeros([photdata['filt'].size,len(self.im1)])#*1e-6 #+ 1e-5
			resultsdict['dmodelflux_dcl'] = np.zeros([photdata['filt'].size,self.n_colorpars])

		for flt in np.unique(photdata['filt']):
			phase=photdata['tobs']+tpkoff
			#Select data from the appropriate filter filter
			selectFilter=(photdata['filt']==flt)
			
			filtPhot={key:photdata[key][selectFilter] for key in photdata}
			phase=phase[selectFilter]
			nphase = len(phase)

			salterrinterp = interr1d(phase)
			modelerr = np.sum(pbspl[flt]*salterrinterp[:,idx[flt]], axis=1) * dwave*HC_ERG_AA
			colorerr = 0
			
			
			#Array output indices match time along 0th axis, wavelength along 1st axis
			M0interp = int1dM0(phase)
			M1interp = int1dM1(phase)
			if computeDerivatives:
				modulatedM0= pbspl[flt]*M0interp[:,idx[flt]]
				modulatedM1=pbspl[flt]*M1interp[:,idx[flt]]
				
				if ( (phase>obsphase.max())).any():
					modulatedM0[(phase>obsphase.max())]*= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))[(phase>obsphase.max())]
					modulatedM1[(phase>obsphase.max())]*= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))[(phase>obsphase.max())]
				
				modelsynM0flux=np.sum(modulatedM0, axis=1)*dwave*self.fluxfactor[survey][flt]
				modelsynM1flux=np.sum(modulatedM1, axis=1)*dwave*self.fluxfactor[survey][flt]
				
				resultsdict['dmodelflux_dx0'][selectFilter,0] = modelsynM0flux+ x1*modelsynM1flux
				resultsdict['dmodelflux_dx1'][selectFilter,0] = modelsynM1flux*x0
				
				modulatedFlux= x0*(modulatedM0 +modulatedM1*x1)
				modelflux = x0* (modelsynM0flux+ x1*modelsynM1flux)
			else:
				modelflux = x0* np.sum(pbspl[flt]*(M0interp[:,idx[flt]]+x1*M1interp[:,idx[flt]]), axis=1)*dwave*self.fluxfactor[survey][flt]
				if ( (phase>obsphase.max())).any():
					modelflux[(phase>obsphase.max())]*= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))[(phase>obsphase.max())]
			
			resultsdict['modelflux'][selectFilter] = modelflux
			resultsdict['uncertainty'][selectFilter] = np.sqrt(photdata['fluxcalerr'][selectFilter]**2. + (modelflux*modelerr)**2. + colorerr**2.)

			
			if computeDerivatives:
				#d model / dc is total flux (M0 and M1 components (already modulated with passband)) times the color law and a factor of ln(10)
				#import pdb; pdb.set_trace()
				resultsdict['dmodelflux_dc'][selectFilter,0]=np.sum((modulatedFlux)*np.log(10)*colorlaw[np.newaxis,idx[flt]], axis=1)*dwave*self.fluxfactor[survey][flt]
				#empderiv = np.sum((modulatedFlux/(10. ** (colorlaw * c))*10. ** (colorlaw * (c+0.001)) - modulatedFlux)/1.0e-3, axis=1)*dwave*self.fluxfactor[survey][flt]
				#import pdb; pdb.set_trace()
				if sn=='5999406': import pdb;pdb.set_trace()
				
				for i in range(self.n_colorpars):
					#Color law is linear wrt to the color law parameters; therefore derivative of the color law
					# with respect to color law parameter i is the color law with all other values zeroed
					dcolorlaw_dcli = SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)
					#Multiply M0 and M1 components (already modulated with passband) by c* d colorlaw / d cl_i, with associated normalizations
					resultsdict['dmodelflux_dcl'][selectFilter,i] = np.sum((modulatedFlux)*-0.4*np.log(10)*c*dcolorlaw_dcli(self.wave[idx[flt]])[np.newaxis,:], axis=1)*dwave*self.fluxfactor[survey][flt]
				
				if self.computePCs:
					passbandColorExp=pbspl[flt]*colorexp[idx[flt]]
					intmult = dwave*self.fluxfactor[survey][flt]*_SCALE_FACTOR/(1+z)*x0
					for i in range(len(self.im0)):
						#Range of wavelength and phase values affected by changes in knot i
						waverange=self.waveknotloc[[i%(self.waveknotloc.size-bsorder-1),i%(self.waveknotloc.size-bsorder-1)+bsorder+1]]
						phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-bsorder-1),i//(self.waveknotloc.size-bsorder-1)+bsorder+1]]
						#Check if this filter is inside values affected by changes in knot i
						if waverange[0]*(1+z) > self.kcordict[survey][flt]['maxlam'] or waverange[1]*(1+z) < self.kcordict[survey][flt]['minlam']:
							pass
						#Check which phases are affected by knot i
						inPhase=(phase>phaserange[0]*(1+z) ) & (phase<phaserange[1]*(1+z) )
						if inPhase.any():
							#Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
							#Integrate only over wavelengths within the relevant range
							inbounds=(self.wave[idx[flt]]>waverange[0])	 & (self.wave[idx[flt]]<waverange[1])
							derivInterp = interp1d(obsphase/(1+z),self.spline_derivs[i][:,inbounds],axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")

							resultsdict['dmodelflux_dM0'][np.where(selectFilter)[0][inPhase],i] =  \
								np.sum( passbandColorExp[:,inbounds] * derivInterp(phase[inPhase]), axis=1)*\
								intmult

							#Dependence is the same for dM1, except with an extra factor of x1
							resultsdict['dmodelflux_dM1'][np.where(selectFilter)[0][inPhase],i] =  resultsdict['dmodelflux_dM0'][selectFilter,i][inPhase]*x1
					if ( (phase>obsphase.max())).any():
						resultsdict['dmodelflux_dM0'][np.where(selectFilter)[0][(phase>obsphase.max())],:] *= \
							10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))[(phase>obsphase.max())]
						resultsdict['dmodelflux_dM1'][np.where(selectFilter)[0][(phase>obsphase.max())],:] *= \
							10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))[(phase>obsphase.max())]
			
		return resultsdict


	def derivativesforPrior(self,x,components,colorLaw):
		M0derivinterp = self.M0derivinterp

		for i,h,j in zip(self.im0,self.im1,range(len(self.im0))):
			# apply tBmax prior
			#if self.waveknotsout[j] > self.kcordict['default']['minlam']-500 and \
			#	self.waveknotsout[j] < self.kcordict['default']['maxlam']+500:
			M0derivB = M0derivinterp[j]([0])
			self.priorderivdict['Bmax'][i] = (self.m0prior(self.modcomp[j])-self.m0prior(components))/1.0e-3
			self.priorderivdict['M0end'][i] = (self.m0endprior(self.modcomp[j])-self.m0endprior(components))/1.0e-7
			self.priorderivdict['M1end'][h] = (self.m1endprior(self.modcomp[j])-self.m1endprior(components))/1.0e-7

		colorpars = x[self.parlist == 'cl']			
		for i,j in zip(self.iCL,range(self.n_colorpars)):
			colorpars[j] += 1.0e-3
			modcolorLaw = SALT2ColorLaw(self.colorwaverange, colorpars)
			colorpars[j] -= 1.0e-3
			self.priorderivdict['colorlaw'][i] = (self.EBVprior(modcolorLaw)-self.EBVprior(colorLaw))/1.0e-3
			
		x1tmp = np.zeros(self.nsn)
		for sn,i in zip(self.datadict.keys(),range(len(self.datadict.keys()))):
			x1tmp[i] = x[self.parlist == 'x1_%s'%sn] - np.mean(x[self.ix1])
		x1 = copy.deepcopy(x1tmp)
			
		for sn,i in zip(self.datadict.keys(),range(len(self.datadict.keys()))):
			#self.priorderivdict['x1mean'][self.parlist == 'x1_%s'%sn] = -np.sum(x[self.ix1])/self.nsn**2./0.02**2.

			x1tmp[i] += 1.0e-4
			#x1 = x[self.parlist == 'x1_%s'%sn]
			self.priorderivdict['x1std'][self.parlist == 'x1_%s'%sn] = (self.m1priorstd(x1tmp)-self.m1priorstd(x1))/1.0e-4
			self.priorderivdict['x1mean'][self.parlist == 'x1_%s'%sn] = (self.m1priormean(x1tmp)-self.m1priormean(x1))/1.0e-4
			x1tmp[i] -= 1.0e-4



	def loglikeforSN(self,args,sn=None,x=None,components=None,salterr=None,
					 colorLaw=None,colorScat=None,
					 debug=False,timeit=False):
		"""
		Calculates the likelihood of given SALT model to photometric and spectroscopic observations of a single SN 

		Parameters
		----------
		args : tuple or None
			Placeholder that contains all the other variables for multiprocessing quirks

		sn : str
			Name of supernova to compare to model
			
		x : array
			SALT model parameters
			
		components: array_like, optional
			SALT model components, if not provided will be derived from SALT model parameters passed in \'x\'
		
		colorLaw: function, optional
			SALT color law which takes wavelength as an argument

		debug : boolean, optional
			Debug flag
		
		Returns
		-------
		chi2: float
			Model chi2 relative to training data	
		"""

		if timeit: tstart = time.time()

		if args: empty,sn,x,components,salterr,colorLaw,colorScat,debug = args[:]
		x = np.array(x)
		
		#Set up SALT model
		if components is None:
			components = self.SALTModel(x)
		if salterr is None:
			salterr = self.ErrModel(x)
		if self.n_components == 1: M0 = copy.deepcopy(components[0])
		elif self.n_components == 2: M0,M1 = copy.deepcopy(components)

		photResidsDict = self.photResidsForSN(x,sn,components,colorLaw,False,bsorder=3)
		specResidsDict = self.specResidsForSN(x,sn,components,colorLaw,False,bsorder=3)
		
		if self.lsqfit:
			loglike=-1*np.append(photResidsDict['photresid'],specResidsDict['specresid'])
			#loglike = np.append(loglike,loglikes)
		else:
			loglike=-1*((photResidsDict['photresid']**2.).sum() + (specResidsDict['specresid']**2.).sum())
		#import pdb; pdb.set_trace()
			
		return loglike
		
		#Declare variables
		if timeit: tstartinit = time.time()
		photdata = self.datadict[sn]['photdata']
		specdata = self.datadict[sn]['specdata']
		pbspl = self.datadict[sn]['pbspl']
		dwave = self.datadict[sn]['dwave']
		idx = self.datadict[sn]['idx']
		survey = self.datadict[sn]['survey']
		filtwave = self.kcordict[survey]['filtwave']
		z = self.datadict[sn]['zHelio']
		obswave = self.datadict[sn]['obswave'] #self.wave*(1+z)
		obsphase = self.datadict[sn]['obsphase'] #self.phase*(1+z)
		#for i in range(len(obsphase)):
		M0 *= self.datadict[sn]['mwextcurve']
		M1 *= self.datadict[sn]['mwextcurve']
		if timeit: self.tdelt0 += time.time() - tstartinit

		x0,x1,c,tpkoff = \
			x[self.parlist == 'x0_%s'%sn][0],x[self.parlist == 'x1_%s'%sn][0],\
			x[self.parlist == 'c_%s'%sn][0],x[self.parlist == 'tpkoff_%s'%sn][0]
		#x1 -= np.mean(x[self.ix1])
		if self.fix_t0: tpkoff = 0

		#Calculate spectral model
		if self.n_components == 1:
			saltflux = x0*M0
		elif self.n_components == 2:
			saltflux = x0*(M0 + x1*M1)
		if colorLaw:
			saltflux *= 10. ** (-0.4 * colorLaw(self.wave) * c)
		saltflux *= _SCALE_FACTOR/(1+z)


		modeldict=self.photmodelvalsforSN(x,sn,components,colorLaw,False,3)
		
		if self.lsqfit: loglike = np.array([])
		else: loglike = 0
		int1d = interp1d(obsphase,saltflux,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
		interr1d = interp1d(obsphase,salterr,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
		intspecerr1d=interr1d
		for k in specdata.keys():
			phase=specdata[k]['tobs']+tpkoff
			#saltfluxinterp = int1d(phase)
			if phase < obsphase.min():
				pass
			elif phase > obsphase.max():
				saltfluxinterp*=10**(-0.4* self.extrapolateDecline* (phase-obsphase.max()))
			#Interpolate SALT flux at observed wavelengths and multiply by recalibration factor
			#coeffs=x[self.parlist=='specrecal_{}_{}'.format(sn,k)]
			#coeffs/=factorial(np.arange(len(coeffs)))
			#saltfluxinterp = np.interp(specdata[k]['wavelength'],obswave,saltfluxinterp)*\
			#	np.exp(np.poly1d(coeffs)((specdata[k]['wavelength']-np.mean(specdata[k]['wavelength']))/self.specrange_wavescale_specrecal))

			#modelerr = np.interp( specdata[k]['wavelength'],obswave,intspecerr1d(phase))
			
			#specvar=(specdata[k]['fluxerr']**2. + (modelerr*saltfluxinterp)**2.)
			specvar=(specdata[k]['fluxerr']**2.)# + (1e-3*specdata[k]['modelflux'])**2.)

			if self.lsqfit:
				loglikes = -(specdata[k]['flux']-specdata[k]['modelflux'])/np.sqrt(specvar)*self.num_phot/self.num_spec
				loglike = np.append(loglike,loglikes)
			else:
				loglike += (np.sum(-(specdata[k]['flux']-specdata[k]['modelflux'])**2./specvar/2.+np.log(1/(np.sqrt(2*np.pi)*specvar)))) *self.num_phot/self.num_spec

				
		if timeit: self.tdelt1 += time.time() - tstart
		for flt in np.unique(photdata['filt']):
			# check if filter 
			if timeit: time2 = time.time()
			phase=photdata['tobs']+tpkoff
			#Select data from the appropriate filter filter
			selectFilter=(photdata['filt']==flt)
			
			filtPhot={key:photdata[key][selectFilter] for key in photdata}
			phase=phase[selectFilter]

			#Array output indices match time along 0th axis, wavelength along 1st axis
			#saltfluxinterp = int1d(phase)
			salterrinterp = interr1d(phase)
			# synthetic photometry from SALT model
			# Integrate along wavelength axis

			if timeit:
				time3 = time.time()
				self.tdelt2 += time3 - time2
			#modelsynflux = np.sum(pbspl[flt]*saltfluxinterp[:,idx[flt]], axis=1)*dwave
			#modelflux = modelsynflux*self.fluxfactor[survey][flt]

			#if ( (phase>obsphase.max())).any():
			#	modelflux[(phase>obsphase.max())]*= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))[(phase>obsphase.max())]

			modelerr = np.sum(pbspl[flt]*salterrinterp[:,idx[flt]], axis=1) * dwave*HC_ERG_AA

			if colorScat: colorerr = splev(self.kcordict[survey][flt]['lambdaeff'],
										   (self.splinecolorwave,x[self.parlist == 'clscat'],3))
			else: colorerr = 0.0
			if timeit:
				time4 = time.time()
				self.tdelt3 += time4 - time3

			# likelihood function
			fluxVar=filtPhot['fluxcalerr']**2. + modeldict['modelflux'][selectFilter]**2.*modelerr**2. + colorerr**2.

			if self.lsqfit:
				loglikes=(-(filtPhot['fluxcal']-modeldict['modelflux'][selectFilter])/np.sqrt(fluxVar))#**2./2./(fluxVar)+np.log(1/(np.sqrt(2*np.pi)*np.sqrt(fluxVar))))
				loglike = np.append(loglike,loglikes)
			else:
				loglikes=(-(filtPhot['fluxcal']-modeldict['modelflux'][selectFilter])**2./2./(fluxVar)+np.log(1/(np.sqrt(2*np.pi)*np.sqrt(fluxVar))))
				loglike += loglikes.sum()

			if timeit:
				time5 = time.time()
				self.tdelt4 += time5 - time4

			if self.debug:
				if sn == 5999389 and flt == 'g':
					print(sn)
					import pylab as plt
					plt.ion()
					plt.clf()
					plt.errorbar(filtPhot['tobs'],modelflux,fmt='o',color='C0',label='model')
					plt.errorbar(filtPhot['tobs'],filtPhot['fluxcal'],yerr=filtPhot['fluxcalerr'],fmt='o',color='C1',label='obs')

		return loglike

		
	def specchi2(self):

		return chi2
	
	def SALTModel(self,x,bsorder=3,evaluatePhase=None,evaluateWave=None):
		
		try: m0pars = x[self.m0min:self.m0max]
		except: import pdb; pdb.set_trace()
		try:
			m0 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
						 self.wave if evaluateWave is None else evaluateWave,
						 (self.phaseknotloc,self.waveknotloc,m0pars,bsorder,bsorder))
		except:
			import pdb; pdb.set_trace()
			
		if self.n_components == 2:
			m1pars = x[self.parlist == 'm1']
			m1 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
						 self.wave if evaluateWave is None else evaluateWave,
						 (self.phaseknotloc,self.waveknotloc,m1pars,bsorder,bsorder))
			components = (m0,m1)
		elif self.n_components == 1:
			components = (m0,)
		else:
			raise RuntimeError('A maximum of two principal components is allowed')
			
		return components

	def SALTModelDeriv(self,x,bsorder=3,evaluatePhase=None,evaluateWave=None):
		
		try: m0pars = x[self.m0min:self.m0max]
		except: import pdb; pdb.set_trace()
		try:
			m0 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
						 self.wave if evaluateWave is None else evaluateWave,
						 (self.phaseknotloc,self.waveknotloc,m0pars,bsorder,bsorder),
						 dx=1,dy=1)
		except:
			import pdb; pdb.set_trace()
			
		if self.n_components == 2:
			m1pars = x[self.parlist == 'm1']
			m1 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
						 self.wave if evaluateWave is None else evaluateWave,
						 (self.phaseknotloc,self.waveknotloc,m1pars,bsorder,bsorder),
						 dx=1,dy=1)
			components = (m0,m1)
		elif self.n_components == 1:
			components = (m0,)
		else:
			raise RuntimeError('A maximum of two principal components is allowed')
			
		return components

	
	def ErrModel(self,x,bsorder=3,evaluatePhase=None,evaluateWave=None):

		try: errpars = x[self.errmin:self.errmax]
		except: import pdb; pdb.set_trace()

		modelerr = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
						   self.wave if evaluateWave is None else evaluateWave,
						   (self.errphaseknotloc,self.errwaveknotloc,errpars,bsorder,bsorder))

		return modelerr

	def getParsGN(self,x,bsorder=3):

		m0pars = x[self.parlist == 'm0']
		m0err = np.zeros(len(x[self.parlist == 'm0']))
		m1pars = x[self.parlist == 'm1']
		m1err = np.zeros(len(x[self.parlist == 'm1']))
		
		# covmat (diagonals only?)
		m0_m1_cov = np.zeros(len(m0pars))

		modelerrpars = x[self.parlist == 'modelerr']
		modelerrerr = np.zeros(len(x[self.parlist == 'modelerr']))

		clpars = x[self.parlist == 'cl']
		clerr = np.zeros(len(x[self.parlist == 'cl']))
		
		clscatpars = x[self.parlist == 'clscat']
		clscaterr = np.zeros(x[self.parlist == 'clscat'])
		

		resultsdict = {}
		n_sn = len(self.datadict.keys())
		for k in self.datadict.keys():
			tpk_init = self.datadict[k]['photdata']['mjd'][0] - self.datadict[k]['photdata']['tobs'][0]
			resultsdict[k] = {'x0':x[self.parlist == 'x0_%s'%k],
							  'x1':x[self.parlist == 'x1_%s'%k],# - np.mean(x[self.ix1]),
							  'c':x[self.parlist == 'c_%s'%k],
							  'tpkoff':x[self.parlist == 'tpkoff_%s'%k],
							  'x0err':x[self.parlist == 'x0_%s'%k],
							  'x1err':x[self.parlist == 'x1_%s'%k],
							  'cerr':x[self.parlist == 'c_%s'%k],
							  'tpkofferr':x[self.parlist == 'tpkoff_%s'%k]}


		m0 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars,bsorder,bsorder))
		m0errp = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars+m0err,bsorder,bsorder))
		m0errm = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars-m0err,bsorder,bsorder))
		m0err = (m0errp-m0errm)/2.
		if len(m1pars):
			m1 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars,bsorder,bsorder))
			m1errp = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars+m1err,bsorder,bsorder))
			m1errm = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars-m1err,bsorder,bsorder))
			m1err = (m1errp-m1errm)/2.
		else:
			m1 = np.zeros(np.shape(m0))
			m1err = np.zeros(np.shape(m0))

		cov_m0_m1 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0_m1_cov,bsorder,bsorder))
		modelerr = bisplev(self.phase,self.wave,(self.errphaseknotloc,self.errwaveknotloc,modelerrpars,bsorder,bsorder))
		clscat = splev(self.wave,(self.errwaveknotloc,clscatpars,3))
		if not len(clpars): clpars = []

		return(x,self.phase,self.wave,m0,m0err,m1,m1err,cov_m0_m1,modelerr,
			   clpars,clerr,clscat,resultsdict)

	def getPars(self,loglikes,x,nburn=500,bsorder=3,mkplots=False):

		axcount = 0; parcount = 0
		from matplotlib.backends.backend_pdf import PdfPages
		pdf_pages = PdfPages('output/MCMC_hist.pdf')
		fig = plt.figure()
		
		m0pars = np.array([])
		m0err = np.array([])
		for i in self.im0:
			m0pars = np.append(m0pars,x[i,nburn:].mean())#/_SCALE_FACTOR)
			m0err = np.append(m0err,x[i,nburn:].std())#/_SCALE_FACTOR)
			if mkplots:
				if not parcount % 9:
					subnum = axcount%9+1
					ax = plt.subplot(3,3,subnum)
					axcount += 1
					md,std = np.mean(x[i,nburn:]),np.std(x[i,nburn:])
					histbins = np.linspace(md-3*std,md+3*std,50)
					ax.hist(x[i,nburn:],bins=histbins)
					ax.set_title('M0')
					if axcount % 9 == 8:
						pdf_pages.savefig(fig)
						fig = plt.figure()
				parcount += 1

		m1pars = np.array([])
		m1err = np.array([])
		parcount = 0
		for i in self.im1:
			m1pars = np.append(m1pars,x[i,nburn:].mean())
			m1err = np.append(m1err,x[i,nburn:].std())
			if mkplots:
				if not parcount % 9:
					subnum = axcount%9+1
					ax = plt.subplot(3,3,subnum)
					axcount += 1
					md,std = np.mean(x[i,nburn:]),np.std(x[i,nburn:])
					histbins = np.linspace(md-3*std,md+3*std,50)
					ax.hist(x[i,nburn:],bins=histbins)
					ax.set_title('M1')
					if axcount % 9 == 8:
						pdf_pages.savefig(fig)
						fig = plt.figure()
				parcount += 1

		# covmat (diagonals only?)
		m0_m1_cov = np.zeros(len(m0pars))
		chain_len = len(m0pars)
		m0mean = np.repeat(x[self.im0,nburn:].mean(axis=0),np.shape(x[self.im0,nburn:])[0]).reshape(np.shape(x[self.im0,nburn:]))
		m1mean = np.repeat(x[self.im1,nburn:].mean(axis=0),np.shape(x[self.im1,nburn:])[0]).reshape(np.shape(x[self.im1,nburn:]))
		m0var = x[self.im0,nburn:]-m0mean
		m1var = x[self.im1,nburn:]-m1mean
		for i in range(len(m0pars)):
			for j in range(len(m1pars)):
				if i == j: m0_m1_cov[i] = np.sum(m0var[j,:]*m1var[i,:])
		m0_m1_cov /= chain_len


		modelerrpars = np.array([])
		modelerrerr = np.array([])
		for i in np.where(self.parlist == 'modelerr')[0]:
			modelerrpars = np.append(modelerrpars,x[i,nburn:].mean())
			modelerrerr = np.append(modelerrerr,x[i,nburn:].std())

		clpars = np.array([])
		clerr = np.array([])
		for i in self.iCL:
			clpars = np.append(clpars,x[i,nburn:].mean())
			clerr = np.append(clpars,x[i,nburn:].std())

		clscatpars = np.array([])
		clscaterr = np.array([])
		for i in np.where(self.parlist == 'clscat')[0]:
			clscatpars = np.append(clpars,x[i,nburn:].mean())
			clscaterr = np.append(clpars,x[i,nburn:].std())



		result=np.mean(x[:,nburn:],axis=1)

		resultsdict = {}
		n_sn = len(self.datadict.keys())
		for k in self.datadict.keys():
			tpk_init = self.datadict[k]['photdata']['mjd'][0] - self.datadict[k]['photdata']['tobs'][0]
			resultsdict[k] = {'x0':x[self.parlist == 'x0_%s'%k,nburn:].mean(),
							  'x1':x[self.parlist == 'x1_%s'%k,nburn:].mean(),# - x[self.ix1,nburn:].mean(),
							  'c':x[self.parlist == 'c_%s'%k,nburn:].mean(),
							  'tpkoff':x[self.parlist == 'tpkoff_%s'%k,nburn:].mean(),
							  'x0err':x[self.parlist == 'x0_%s'%k,nburn:].std(),
							  'x1err':x[self.parlist == 'x1_%s'%k,nburn:].std(),
							  'cerr':x[self.parlist == 'c_%s'%k,nburn:].std(),
							  'tpkofferr':x[self.parlist == 'tpkoff_%s'%k,nburn:].std()}


		m0 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars,bsorder,bsorder))
		m0errp = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars+m0err,bsorder,bsorder))
		m0errm = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars-m0err,bsorder,bsorder))
		m0err = (m0errp-m0errm)/2.
		if len(m1pars):
			m1 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars,bsorder,bsorder))
			m1errp = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars+m1err,bsorder,bsorder))
			m1errm = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars-m1err,bsorder,bsorder))
			m1err = (m1errp-m1errm)/2.
		else:
			m1 = np.zeros(np.shape(m0))
			m1err = np.zeros(np.shape(m0))

		cov_m0_m1 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0_m1_cov,bsorder,bsorder))
		modelerr = bisplev(self.phase,self.wave,(self.errphaseknotloc,self.errwaveknotloc,modelerrpars,bsorder,bsorder))
		#phaseout,waveout,fluxout = np.array([]),np.array([]),np.array([])
		#for i in range(len(self.phase)):
		#	for j in range(len(self.wave)):
		#		phaseout = np.append(phaseout,self.phase[i])
		#		waveout = np.append(waveout,self.wave[j])
		#		fluxout = np.append(fluxout,modelerr[i,j])
		#bspl = bisplrep(phaseout,waveout,fluxout,kx=3,ky=3,tx=self.errphaseknotloc,ty=self.errwaveknotloc,task=-1)
		#modelerr2 = bisplev(self.phase,self.wave,bspl)
		#plt.plot(self.wave,modelerr[self.phase == 0,:].flatten())
		#plt.plot(self.wave,modelerr2[self.phase == 0,:].flatten())
		clscat = splev(self.wave,(self.errwaveknotloc,clscatpars,3))
		if not len(clpars): clpars = []

		for snpar in ['x0','x1','c','tpkoff']:
			subnum = axcount%9+1
			ax = plt.subplot(3,3,subnum)
			axcount += 1
			md = np.mean(x[self.parlist == '%s_%s'%(snpar,k),nburn:])
			std = np.std(x[self.parlist == '%s_%s'%(snpar,k),nburn:])
			histbins = np.linspace(md-3*std,md+3*std,50)
			ax.hist(x[self.parlist == '%s_%s'%(snpar,k),nburn:],bins=histbins)
			ax.set_title('%s_%s'%(snpar,k))
			if axcount % 9 == 8:
				pdf_pages.savefig(fig)
				fig = plt.figure()


		pdf_pages.savefig(fig)			
		pdf_pages.close()
		
		return(result,self.phase,self.wave,m0,m0err,m1,m1err,cov_m0_m1,modelerr,
			   clpars,clerr,clscat,resultsdict)

	
	def synphot(self,sourceflux,zpoff,survey=None,flt=None,wave=None,redshift=0):
		if wave is None:
			obswave = self.wave*(1+redshift)
		else:
			obswave = wave*(1+redshift)
			
		filtwave = self.kcordict[survey]['filtwave']
		filttrans = self.kcordict[survey][flt]['filttrans']

		g = (obswave >= filtwave[0]) & (obswave <= filtwave[-1])  # overlap range

		pbspl = np.interp(obswave[g],filtwave,filttrans)
		pbspl *= obswave[g]

		res = np.trapz(pbspl*sourceflux[g]/HC_ERG_AA,obswave[g])/np.trapz(pbspl,obswave[g])
		return(zpoff-2.5*np.log10(res))
		
	def updateEffectivePoints(self,x):
		"""
		Updates the "effective number of points" constraining a given bin in 
		phase/wavelength space. Should be called any time tpkoff values are recalculated
		
		Parameters
		----------
			
		x : array
			SALT model parameters
					
		"""
		#Clean out array
		self.neff=np.zeros((self.phasebins.size-1,self.wavebins.size-1))
		for sn in self.datadict.keys():
			tpkoff=x[self.parlist == 'tpkoff_%s'%sn]
			photdata = self.datadict[sn]['photdata']
			specdata = self.datadict[sn]['specdata']
			survey = self.datadict[sn]['survey']
			filtwave = self.kcordict[survey]['filtwave']
			z = self.datadict[sn]['zHelio']

			#For each spectrum, add one point to each bin for every spectral measurement in that bin
			for k in specdata.keys():
				restWave=specdata[k]['wavelength']/(1+z)
				restWave=restWave[(restWave>self.wavebins.min())&(restWave<self.wavebins.max())]
				phase=(specdata[k]['tobs']+tpkoff)/(1+z)
				phaseIndex=np.clip(np.searchsorted(self.phasebins,phase)-1,0,self.phasebins.size-2)[0]
				waveIndices=np.clip(np.searchsorted(self.wavebins,restWave)-1,0,self.wavebins.size-2)
				self.neff[phaseIndex][waveIndices]+=1			
			
			#For each photometric filter, weight the contribution by  
			for flt in np.unique(photdata['filt']):
				g = (self.wavebins[:-1]	 >= filtwave[0]/(1+z)) & (self.wavebins[1:] <= filtwave[-1]/(1+z))	# overlap range
				if flt+'-neffweighting' in self.datadict[sn]:
					pbspl=self.datadict[sn][flt+'-neffweighting']
				else:
					filttrans = self.kcordict[survey][flt]['filttrans']
					pbspl = np.zeros(g.sum())
					for i in range(g.sum()):
						j=np.where(g)[0][i]
						pbspl[i]=trapIntegrate(self.wavebins[j],self.wavebins[j+1],filtwave/(1+z),filttrans*filtwave/(1+z))
					#Normalize it so that total number of points added is 1
					pbspl /= np.sum(pbspl)
					self.datadict[sn][flt+'-neffweighting']=pbspl
				#Couple of things going on: -1 to ensure everything lines up, clip to put extrapolated points on last phasebin
				phaseindices=np.clip(np.searchsorted(self.phasebins,(photdata['tobs'][(photdata['filt']==flt)]+tpkoff)/(1+z))-1,0,self.phasebins.size-2)
				#Consider weighting neff by variance for each measurement?
				self.neff[phaseindices,:][:,g]+=pbspl[np.newaxis,:]

		#Smear it out a bit along phase axis
		self.neff=gaussian_filter1d(self.neff,1,0)

		self.neff=np.clip(self.neff,1e-2*self.neff.max(),None)
		self.plotEffectivePoints([-12.5,0,12.5,40],'neff.png')
		self.plotEffectivePoints(None,'neff-heatmap.png')

	def plotEffectivePoints(self,phases=None,output=None):

		import matplotlib.pyplot as plt
		if phases is None:
			plt.imshow(self.neff,cmap='Greys',aspect='auto')
			xticks=np.linspace(0,self.wavebins.size,8,False)
			plt.xticks(xticks,['{:.0f}'.format(self.wavebins[int(x)]) for x in xticks])
			plt.xlabel('$\lambda$ / Angstrom')
			yticks=np.linspace(0,self.phasebins.size,8,False)
			plt.yticks(yticks,['{:.0f}'.format(self.phasebins[int(x)]) for x in yticks])
			plt.ylabel('Phase / days')
		else:
			inds=np.searchsorted(self.phasebins,phases)
			for i in inds:
				plt.plot(self.wavebins[:-1],self.neff[i,:],label='{:.1f} days'.format(self.phasebins[i]))
			plt.ylabel('$N_eff$')
			plt.xlabel('$\lambda (\AA)$')
			plt.xlim(self.wavebins.min(),self.wavebins.max())
			plt.legend()
		
		if output is None:
			plt.show()
		else:
			plt.savefig(output,dpi=288)
		plt.clf()


	def regularizationChi2(self, x,gradientPhase,gradientWave,dyad):
		
		fluxes=self.SALTModel(x,evaluatePhase=self.phasebins[:-1],evaluateWave=self.wavebins[:-1])

		chi2wavegrad=0
		chi2phasegrad=0
		chi2dyad=0
		for i in range(self.n_components):
			exponent=1
			fluxvals=fluxes[i]/np.mean(fluxes[i])
			if gradientWave !=0:
				chi2wavegrad+= self.waveres**exponent/((self.wavebins.size-1)**2 *(self.phasebins.size-1)) * (( (fluxvals[:,:,np.newaxis]-fluxvals[:,np.newaxis,:])**2	 /	 (np.sqrt(self.neff[:,:,np.newaxis]*self.neff[:,np.newaxis,:])* np.abs(self.wavebins[np.newaxis,np.newaxis,:-1]-self.wavebins[np.newaxis,:-1,np.newaxis])**exponent))[:,~np.diag(np.ones(self.wavebins.size-1,dtype=bool))]).sum()
			if gradientPhase != 0:
				chi2phasegrad+= self.phaseres**exponent/((self.phasebins.size-1)**2 *(self.wavebins.size-1) ) * ((	(fluxvals[np.newaxis,:,:]-fluxvals[:,np.newaxis,:])**2	 /	 (np.sqrt(self.neff[np.newaxis,:,:]*self.neff[:,np.newaxis,:])* np.abs(self.phasebins[:-1,np.newaxis,np.newaxis]-self.phasebins[np.newaxis,:-1,np.newaxis])**exponent))[~np.diag(np.ones(self.phasebins.size-1,dtype=bool)),:]).sum()
			if dyad!= 0:
				chi2dyadvals=(	 (fluxvals[:,np.newaxis,:,np.newaxis] * fluxvals[np.newaxis,:,np.newaxis,:] - fluxvals[np.newaxis,:,:,np.newaxis] * fluxvals[:,np.newaxis,np.newaxis,:])**2)   /   (np.sqrt(np.sqrt(self.neff[np.newaxis,:,np.newaxis,:]*self.neff[np.newaxis,:,:,np.newaxis]*self.neff[:,np.newaxis,:,np.newaxis]*self.neff[:,np.newaxis,np.newaxis,:]))*np.abs(self.wavebins[np.newaxis,np.newaxis,:-1,np.newaxis]-self.wavebins[np.newaxis,np.newaxis,np.newaxis,:-1])*np.abs(self.phasebins[:-1,np.newaxis,np.newaxis,np.newaxis]-self.phasebins[np.newaxis,:-1,np.newaxis,np.newaxis]))
				chi2dyad+=self.phaseres*self.waveres/( (self.wavebins.size-1) *(self.phasebins.size-1))**2	* chi2dyadvals[~np.isnan(chi2dyadvals)].sum()
		
		return gradientWave*chi2wavegrad+dyad*chi2dyad+gradientPhase*chi2phasegrad


class mcmc(loglike):
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

class GaussNewton(loglike):
	def __init__(self,guess,datadict,parlist,chain=[],loglikes=[],**kwargs):
		self.loglikes=loglikes
		self.chain=chain
		self.warnings = []
		
		super().__init__(guess,datadict,parlist,**kwargs)
		self.lsqfit = False
		
		self._robustify = False
		self._writetmp = False
		self.chi2_diff_cutoff = 1

	def addwarning(self,warning):
		print(warning)
		self.warnings.append(warning)
		
	def convergence_loop(self,guess,loop_niter=10):
		lastResid = 1e20

		print('Initializing')

		residuals = self.lsqwrap(guess,False,doPriors=True)
		chi2_init = (residuals**2.).sum()
		X = copy.deepcopy(guess[:])

		print('starting loop')
		for superloop in range(loop_niter):

			X,chi2 = self.robust_process_fit(X,chi2_init)
			
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

	def lsqwrap(self,guess,computeDerivatives,doPriors=True):
		tstart = time.time()

		if self.n_colorscatpars:
			colorScat = True
		else: colorScat = None

		if self.n_colorpars:
			colorLaw = SALT2ColorLaw(self.colorwaverange, guess[self.parlist == 'cl'])
		else: colorLaw = None
		
		components = self.SALTModel(guess)
		# set model vals
		numResids=self.num_phot+self.num_spec + (0 if doPriors else 0)
		residuals = np.zeros( numResids)
		jacobian =	np.zeros((numResids,self.npar)) # Jacobian matrix from r


		idx = 0
		for sn in self.datadict.keys():
			photresidsdict=self.photResidsForSN(guess,sn,components,colorLaw,computeDerivatives)
			idxp = photresidsdict['photresid'].size

			residuals[idx:idx+idxp] = photresidsdict['photresid']
			if computeDerivatives:
				jacobian[idx:idx+idxp,self.parlist=='cl'] = photresidsdict['dphotresid_dcl']
				jacobian[idx:idx+idxp,self.parlist=='m0'] = photresidsdict['dphotresid_dM0']
				jacobian[idx:idx+idxp,self.parlist=='m1'] = photresidsdict['dphotresid_dM1']

				for snparam in ('x0','x1','c'): #tpkoff should go here
					jacobian[idx:idx+idxp,self.parlist == '{}_{}'.format(snparam,sn)] = photresidsdict['dphotresid_d{}'.format(snparam)]
			idx += idxp

			specresidsdict=self.specResidsForSN(guess,sn,components,colorLaw,computeDerivatives)
			idxp = specresidsdict['specresid'].size

			residuals[idx:idx+idxp] = specresidsdict['specresid']
			if computeDerivatives:
				jacobian[idx:idx+idxp,self.parlist=='cl'] = specresidsdict['dspecresid_dcl']
				jacobian[idx:idx+idxp,self.parlist=='m0'] = specresidsdict['dspecresid_dM0']
				jacobian[idx:idx+idxp,self.parlist=='m1'] = specresidsdict['dspecresid_dM1']

				for snparam in ('x0','x1','c'): #tpkoff should go here
					jacobian[idx:idx+idxp,self.parlist == '{}_{}'.format(snparam,sn)] = specresidsdict['dspecresid_d{}'.format(snparam)]
			idx += idxp

			
#			for k in specdata.keys():
#				idxp = len(specdata[k]['flux'])
#				if fit == 'all' or fit == 'colorlaw':
#					dmodelflux_dcl[idx:idx+idxp,:] = specdata[k]['dmodelflux_dcl']
#				if fit == 'all' or fit == 'component0' or fit == 'components' or fit == 'component0spec':
#					dmodelflux_dM0[idx:idx+idxp,:] = specdata[k]['dmodelflux_dM0']
#				if fit == 'all' or fit == 'component1' or fit == 'components' or fit == 'component1spec':
#					dmodelflux_dM1[idx:idx+idxp,:] = specdata[k]['dmodelflux_dM1']
#				
#				residuals[idx:idx+idxp] = specdata[k]['residuals']
# 
#				if fit == 'all' or fit == 'sn' or fit == 'x0' or fit == 'x0spec' or fit == 'snspec':
#					for j in range(len(specdata[k]['dmodelflux_dx0'])):
#						jacobian[idx+j,self.parlist == 'x0_%s'%sn] = specdata[k]['dmodelflux_dx0'][j]
#						if not fit.startswith('x0'): jacobian[idx+j,self.parlist == 'x1_%s'%sn] = specdata[k]['dmodelflux_dx1'][j]
#				if fit == 'all' or fit == 'color':
#					for j in range(len(specdata[k]['dmodelflux_dx0'])):
#						jacobian[idx+j,self.parlist == 'c_%s'%sn] = specdata[k]['dmodelflux_dc'][j]
#				idx += idxp

		# priors
		if False:#doPriors:
			jacobian[idx,:] = self.priorderivdict['Bmax'][self.parlist == 'm0']
			residuals[idx] = -1*self.m0prior(components)[0]
			idx += 1
			# THESE PRIORS DON'T WORK
			#if fit == 'all' or fit == 'priors' or fit == 'x1meanprior':
			#	for sn in self.datadict.keys():
			#		jacobian[idx,self.parlist == 'x1_%s'%sn] = self.priorderivdict['x1mean'][self.parlist == 'x1_%s'%sn]
			#residuals[idx] = 1*self.m1priormean(X[self.ix1])
			#import pdb; pdb.set_trace()
			#idx += 1
			#if fit == 'all' or fit == 'priors' or fit == 'x1stdprior':
			#	for sn in self.datadict.keys():
			#		jacobian[idx,self.parlist == 'x1_%s'%sn] = self.priorderivdict['x1std'][self.parlist == 'x1_%s'%sn]
			#residuals[idx] = 1*self.m1priorstd(X[self.ix1])
			#idx += 1
			#if fit == 'all' or fit == 'colorlaw':
			#	for cl,i in zip(np.where(self.parlist == 'cl')[0],range(len(np.where(self.parlist == 'cl')[0]))):
			#		jacobian[idx,cl] = self.priorderivdict['colorlaw'][i]
			#residuals[idx] = -1*self.EBVprior(colorLaw)
			if fit == 'all' or fit == 'components':
				dmodelflux_dM0[idx:idx+1,:] = self.priorderivdict['M0end'][self.parlist == 'm0']
			residuals[idx] = -1*self.m0endprior(components)
			idx += 1
			if fit == 'all' or fit == 'components':
				dmodelflux_dM1[idx:idx+1,:] = self.priorderivdict['M1end'][self.parlist == 'm1']
			residuals[idx] = -1*self.m1endprior(components)
		
		if computeDerivatives:
			print('loop took %i seconds'%(time.time()-tstart))
			return residuals,jacobian
		else:
			return residuals
	
	def robust_process_fit(self,X,chi2_init):

		print('initial priors: M0 B abs mag, mean x1, x1 std, M0 start, M1 start')
		components = self.SALTModel(X)
		print(self.m0prior(components,m0=True)[0],
			  np.mean(X[self.ix1]),np.std(X[self.ix1]),
			  np.sum(components[0][0,:]),np.sum(components[1][0,:]))

		print('hack!')
		Xtmp = copy.deepcopy(X)
		if not 'hi': Xtmp,chi2_all = self.process_fit(Xtmp,fit='all')
		
		if not 'hi': #if chi2_init - chi2_all > 1:
			return Xtmp,chi2_all
		else:
			# "basic flipflop"??!?!
			print("fitting SNe")
			X,chi2 = self.process_fit(X,fit='sn')
			print("fitting principal components")
			X,chi2 = self.process_fit(X,fit='components')
			print("fitting color")
			X,chi2 = self.process_fit(X,fit='color')
			print("fitting color law")
			X,chi2 = self.process_fit(X,fit='colorlaw')

		return X,chi2 #_init
	
	def process_fit(self,X,fit='all',doPriors=False):

		if fit == 'all' or fit == 'components': self.computePCs = True
		else: self.computePCs = False

#			if fit == 'all' or fit == 'colorlaw':
#			if fit == 'all' or fit == 'component0' or fit == 'components' or fit == 'components0phot':
#			if fit == 'all' or fit == 'component1' or fit == 'components':
#			if fit == 'all' or fit == 'component1' or fit == 'components' or fit == 'component1spec':
#			if fit == 'all' or fit == 'component0' or fit == 'components' or fit == 'component0spec':
#			if fit == 'all' or fit == 'sn' or fit == 'x0' or fit == 'x0spec' or fit == 'snspec':
#			if fit == 'all' or fit == 'sn' or fit == 'x0' or fit == 'x0phot' or fit == 'snphot':
#			if fit == 'all' or fit == 'color':
#			if not fit.startswith('x0'):
#				if fit == 'all' or fit == 'color':

		residuals,jacobian=self.lsqwrap(X,True,doPriors)
		if fit == 'all':
			includePars=np.ones(self.npar,dtype=bool)
		else:
			includePars=np.zeros(self.npar,dtype=bool)
			if fit=='components':
				includePars[self.im0]=True
				includePars[self.im1]=True
			elif fit=='sn':
				#print('hack: x0 only')
				includePars[self.ix0]=True
				includePars[self.ix1]=True
			elif fit=='color':
				includePars[self.ic]=True
			elif fit=='colorlaw':
				includePars[self.iCL]=True
			else:
				raise NotImplementedError("""This option for a Gaussian Process fit with a 
restricted parameter set has not been implemented: {}""".format(fit))
		print('Number of parameters fit this round: {}'.format(includePars.sum()))
		jacobian=jacobian[:,includePars]
		stepsize = np.dot(np.dot(pinv(np.dot(jacobian.T,jacobian)),jacobian.T),
						  residuals.reshape(residuals.size,1)).reshape(includePars.sum())


		X[includePars] -= stepsize
		
		print('priors: M0 B abs mag, mean x1, x1 std, M0 start, M1 start')
		components = self.SALTModel(X)
		# check new spectra
		#plt.plot(specdata[0]['wavelength'],specdata[0]['flux'])
		#plt.plot(specdata[0]['wavelength'],specdata[0]['modelflux'])
		# off by like 1e-5

		print(self.m0prior(components,m0=True)[0],
			  np.mean(X[self.ix1]),np.std(X[self.ix1]),
			  np.sum(components[0][0,:]),np.sum(components[1][0,:]))


		# quick eval

		chi2 = np.sum(self.lsqwrap(X,False,doPriors=doPriors)**2.)
		print("chi2: old, new, diff")
		print((residuals**2).sum(),chi2,(residuals**2).sum()-chi2)

		return X,chi2
	
def trapIntegrate(a,b,xs,ys):
	if (a<xs.min()) or (b>xs.max()):
		raise ValueError('Bounds of integration outside of provided values')
	aInd,bInd=np.searchsorted(xs,[a,b])
	if aInd==bInd:
		return ((ys[aInd]-ys[aInd-1])/(xs[aInd]-xs[aInd-1])*((a+b)/2-xs[aInd-1])+ys[aInd-1])*(b-a)
	elif aInd+1==bInd:
		return ((ys[aInd]-ys[aInd-1])/(xs[aInd]-xs[aInd-1])*((a+xs[aInd])/2-xs[aInd-1])+ys[aInd-1])*(xs[aInd]-a) + ((ys[bInd]-ys[bInd-1])/(xs[bInd]-xs[bInd-1])*((xs[bInd-1]+b)/2-xs[bInd-1])+ys[bInd-1])*(b-xs[bInd-1])
	else:
		return np.trapz(ys[(xs>=a)&(xs<b)],xs[(xs>=a)&(xs<b)])+((ys[aInd]-ys[aInd-1])/(xs[aInd]-xs[aInd-1])*((a+xs[aInd])/2-xs[aInd-1])+ys[aInd-1])*(xs[aInd]-a) + ((ys[bInd]-ys[bInd-1])/(xs[bInd]-xs[bInd-1])*((xs[bInd-1]+b)/2-xs[bInd-1])+ys[bInd-1])*(b-xs[bInd-1])
