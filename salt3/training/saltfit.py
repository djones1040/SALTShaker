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

_SCALE_FACTOR = 1e-12
_B_LAMBDA_EFF = np.array([4353.])
_V_LAMBDA_EFF = np.array([5477.])

class fitting:
	def __init__(self,n_components,n_colorpars,
				 n_phaseknots,n_waveknots,datadict):

		self.n_phaseknots = n_phaseknots
		self.n_waveknots = n_waveknots
		self.n_components = n_components
		self.n_colorpars = n_colorpars
		self.datadict = datadict
		
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

		self.nstep = 0
		self.parlist = parlist
		self.datadict = datadict
		self.guess = guess
		self.lsqfit = False
		
		for key, value in kwargs.items(): 
			self.__dict__[key] = value

		# pre-set some indices
		self.m0min = np.min(np.where(self.parlist == 'm0')[0])
		self.m0max = np.max(np.where(self.parlist == 'm0')[0])
		self.errmin = np.min(np.where(self.parlist == 'modelerr')[0])
		self.errmax = np.max(np.where(self.parlist == 'modelerr')[0])
		self.ix1 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x1')])
		
		# set some phase/wavelength arrays
		self.splinecolorwave = np.linspace(self.colorwaverange[0],self.colorwaverange[1],self.n_colorpars)
		self.phasebins = np.linspace(self.phaserange[0],self.phaserange[1],
							 1+ (self.phaserange[1]-self.phaserange[0])/self.phaseres)
		self.wavebins = np.linspace(self.waverange[0],self.waverange[1],
							 1+(self.waverange[1]-self.waverange[0])/self.waveres)

		self.phase = np.linspace(self.phaserange[0],self.phaserange[1],
								 (self.phaserange[1]-self.phaserange[0])/self.phaseoutres,False)
		self.wave = np.linspace(self.waverange[0],self.waverange[1],(self.waverange[1]-self.waverange[0])/self.waveoutres,False)

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
		
		self.m0guess = -19.49
		self.m1guess = 1
		self.extrapolateDecline=0.015
		# set up the filters
		self.stdmag = {}
		self.fluxfactor = {}
		for survey in self.kcordict.keys():
			if survey == 'default': 
				self.stdmag[survey] = {}
				self.stdmag[survey]['B']=synphot(
					self.kcordict[survey]['primarywave'],self.kcordict[survey]['AB'],
					filtwave=self.kcordict['default']['Bwave'],filttp=self.kcordict[survey]['Btp'],
					zpoff=0)
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
				

		#Count number of photometric and spectroscopic points
		self.num_spec=0
		self.num_phot=0
		for sn in self.datadict.keys():
			photdata = self.datadict[sn]['photdata']
			specdata = self.datadict[sn]['specdata']
			survey = self.datadict[sn]['survey']
			filtwave = self.kcordict[survey]['filtwave']
			z = self.datadict[sn]['zHelio']
			
			self.num_spec+=len(specdata.keys())


			self.datadict[sn]['mwextcurve'] = np.zeros([len(self.phase),len(self.wave)])
			for i in range(len(self.phase)):
				self.datadict[sn]['mwextcurve'][i,:] = 10**(-0.4*extinction.fitzpatrick99(self.wave*(1+z),self.datadict[sn]['MWEBV']*3.1))
			
			for flt in np.unique(photdata['filt']):
				# synthetic photometry
				filtwave = self.kcordict[survey]['filtwave']
				filttrans = self.kcordict[survey][flt]['filttrans']
			
				#Check how much mass of the filter is inside the wavelength range
				filtRange=(filtwave/(1+z)>self.wavebins.min()) &(filtwave/(1+z) <self.wavebins.max())
				num = np.trapz((filttrans*filtwave/(1+z))[filtRange],filtwave[filtRange]/(1+z))
				denom = np.trapz(filttrans*filtwave/(1+z),filtwave/(1+z))
				if not num/denom < 1-self.filter_mass_tolerance:
					self.num_phot+=1

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

				g = (self.datadict[sn]['obswave'] >= filtwave[0]) & (self.datadict[sn]['obswave'] <= filtwave[-1])  # overlap range
				self.datadict[sn]['idx'][flt] = g
			
				pbspl = np.interp(self.datadict[sn]['obswave'][g],filtwave,filttrans)
				pbspl *= self.datadict[sn]['obswave'][g]
				denom = np.trapz(pbspl,self.datadict[sn]['obswave'][g])
				pbspl /= denom*HC_ERG_AA
				
				self.datadict[sn]['pbspl'][flt] = pbspl[np.newaxis,:]
				self.datadict[sn]['denom'][flt] = denom

				
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
			self.tdelt1,self.tdelt2,self.tdelt3,self.tdelt4,self.tdelt5 = 0,0,0,0,0
		
		chi2 = 0
		#Construct arguments for maxlikeforSN method
		#If worker pool available, use it to calculate chi2 for each SN; otherwise, do it in this process
		args=[(None,sn,x,components,salterr,colorLaw,colorScat,debug,timeit) for sn in self.datadict.keys()]
		if pool:
			loglike=sum(pool.map(self.loglikeforSN,args))
		else:
			loglike=sum(starmap(self.loglikeforSN,args))

		loglike -= self.regularizationChi2(x,self.regulargradientphase,self.regulargradientwave,self.regulardyad)
		logp = loglike + self.m0prior(components) + self.m1prior(x[self.ix1]) + self.endprior(components)+self.peakprior(components)
		if colorLaw:
			logp += self.EBVprior(colorLaw)

		self.nstep += 1
		print(logp*-2)
		if timeit:
			print('%.3f %.3f %.3f %.3f'%(self.tdelt1,self.tdelt2,self.tdelt3,self.tdelt4))
		return logp

	def m0prior(self,components):

		int1d = interp1d(self.phase,components[0],axis=0)
		m0B = synphot(self.wave,int1d(0),filtwave=self.kcordict['default']['Bwave'],
					  filttp=self.kcordict['default']['Btp'])-self.stdmag['default']['B']
		logprior = norm.logpdf(m0B,self.m0guess,0.02)

		return logprior

	def m1prior(self,x1pars):

		logprior = norm.logpdf(np.std(x1pars),self.m1guess,0.02)
		logprior += norm.logpdf(np.mean(x1pars),0,0.02)
		
		return logprior
		
	def peakprior(self,components):
		lightcurve=np.trapz(self.wave[np.newaxis,:]*components[0],self.wave,axis=1)
		logprior = norm.logpdf(self.phase[np.argmax(lightcurve)],0,1) #+ norm.logpdf(np.sum(components[0][-1,:]),0,0.1)
		
		return logprior

	def endprior(self,components):
		
		logprior = norm.logpdf(np.sum(components[0][0,:]),0,0.1) #+ norm.logpdf(np.sum(components[0][-1,:]),0,0.1)
		logprior += norm.logpdf(np.sum(components[1][0,:]),0,0.1) #+ norm.logpdf(np.sum(components[1][-1,:]),0,0.1)
		
		return logprior

	
	def EBVprior(self,colorLaw):
		# 0.4*np.log(10) = 0.921
		logpriorB = norm.logpdf(colorLaw(_B_LAMBDA_EFF)[0], 0.0, 0.02)
		logpriorV = norm.logpdf(colorLaw(_V_LAMBDA_EFF)[0], 0.921, 0.02)
		return logpriorB + logpriorV

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
		#if sn==5999425: import pdb; pdb.set_trace()
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

		
		#Declare variables
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
		
		x0,x1,c,tpkoff = \
			x[self.parlist == 'x0_%s'%sn][0],x[self.parlist == 'x1_%s'%sn][0],\
			x[self.parlist == 'c_%s'%sn][0],x[self.parlist == 'tpkoff_%s'%sn][0]
		if self.fix_t0: tpkoff = 0

		#Calculate spectral model
		if self.n_components == 1:
			saltflux = x0*M0
		elif self.n_components == 2:
			saltflux = x0*(M0 + x1*M1)
		if colorLaw:
			saltflux *= 10. ** (-0.4 * colorLaw(self.wave) * c)
		saltflux *= _SCALE_FACTOR/(1+z)

		
		loglike = 0
		int1d = interp1d(obsphase,saltflux,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
		interr1d = interp1d(obsphase,salterr,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
		for k in specdata.keys():
			phase=specdata[k]['tobs']+tpkoff
			saltfluxinterp = int1d(phase)
			if phase < obsphase.min():
				pass
			elif phase > obsphase.max():
				saltfluxinterp*=10**(-0.4* self.extrapolateDecline* (phase-obsphase.max()))
			#Interpolate SALT flux at observed wavelengths and multiply by recalibration factor
			coeffs=x[self.parlist=='specrecal_{}_{}'.format(sn,k)]
			coeffs/=factorial(np.arange(len(coeffs)))
			saltfluxinterp2 = np.interp(specdata[k]['wavelength'],obswave,saltfluxinterp)*\
				np.exp(np.poly1d(coeffs)((specdata[k]['wavelength']-np.mean(specdata[k]['wavelength']))/self.specrange_wavescale_specrecal))
			#print(np.mean(saltfluxinterp2),np.mean(specdata[k]['flux']))
			loglike -= np.sum((saltfluxinterp2-specdata[k]['flux'])**2./(specdata[k]['fluxerr']**2. + 0.01**2.))*self.num_phot/self.num_spec/2.
			#if 'g' in photdata['filt']: import pdb; pdb.set_trace()
			#if sn == 5999433 and k == 0 and not 'hi':
			#	import pylab as plt
			#	plt.clf()
			#	plt.ion()
			#	plt.plot(specdata[k]['wavelength'],specdata[k]['flux']/(1+z))
			#	plt.plot(specdata[k]['wavelength'],saltfluxinterp2)
			#	plt.show()
			#	import pdb; pdb.set_trace()

		if timeit: self.tdelt1 += time.time() - tstart
		if self.lsqfit: loglike = np.array([])
		for flt in np.unique(photdata['filt']):
			# check if filter 
			if timeit: time2 = time.time()			
			phase=photdata['tobs']+tpkoff
			#Select data from the appropriate filter filter
			selectFilter=(photdata['filt']==flt)
			
			filtPhot={key:photdata[key][selectFilter] for key in photdata}
			phase=phase[selectFilter]

			#Array output indices match time along 0th axis, wavelength along 1st axis
			saltfluxinterp = int1d(phase)
			salterrinterp = interr1d(phase)
			# synthetic photometry from SALT model
			# Integrate along wavelength axis

			if timeit:
				time3 = time.time()
				self.tdelt2 += time3 - time2
			modelsynflux = np.sum(pbspl[flt]*saltfluxinterp[:,idx[flt]], axis=1)*dwave
			#import pdb; pdb.set_trace()
			#modelsynflux=np.trapz(pbspl[flt]*saltfluxinterp[:,idx[flt]],obswave[idx[flt]],axis=1)
			modelflux = modelsynflux*self.fluxfactor[survey][flt]
			if ( (phase>obsphase.max())).any():
				modelsynflux[(phase>obsphase.max())]*= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))[(phase>obsphase.max())]

			#modelerr=np.trapz(pbspl[flt]*salterrinterp[:,idx[flt]],obswave[idx[flt]],axis=1)
			modelerr = np.sum(pbspl[flt]*salterrinterp[:,idx[flt]], axis=1) * dwave

			
			if colorScat: colorerr = splev(self.kcordict[survey][flt]['lambdaeff'],
										   (self.splinecolorwave,x[self.parlist == 'clscat'],3))
			else: colorerr = 0.0
			if timeit:
				time4 = time.time()
				self.tdelt3 += time4 - time3
			#import pdb; pdb.set_trace()
			# likelihood function
			if self.lsqfit:
				loglike = np.append(loglike,(-(filtPhot['fluxcal']-modelflux)**2./2./(filtPhot['fluxcalerr']**2. + modelflux**2.*modelerr**2. + colorerr**2.)+\
											 np.log(1/(np.sqrt(2*np.pi)*np.sqrt(filtPhot['fluxcalerr']**2. + modelflux**2.*modelerr**2. + colorerr**2.)))))
			else:
				loglike += (-(filtPhot['fluxcal']-modelflux)**2./2./(filtPhot['fluxcalerr']**2. + modelflux**2.*modelerr**2. + colorerr**2.)+\
							np.log(1/(np.sqrt(2*np.pi)*np.sqrt(filtPhot['fluxcalerr']**2. + modelflux**2.*modelerr**2. + colorerr**2.)))).sum()

				
			if timeit:
				time5 = time.time()
				self.tdelt4 += time5 - time4
			if self.debug and self.nstep > 7000:
				if self.nstep > 1500 and flt == 'd' and sn == 5999398:
					print(sn)
					import pylab as plt
					plt.ion()
					plt.clf()
					plt.errorbar(filtPhot['tobs'],modelflux,fmt='o',color='C0',label='model')
					plt.errorbar(filtPhot['tobs'],filtPhot['fluxcal'],yerr=filtPhot['fluxcalerr'],fmt='o',color='C1',label='obs')
					import pdb; pdb.set_trace()
		
		return loglike

		
	def specchi2(self):

		return chi2
	
	def SALTModel(self,x,bsorder=3,evaluatePhase=None,evaluateWave=None):
		
		try: m0pars = x[self.m0min:self.m0max]
		except: import pdb; pdb.set_trace()
		m0 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
					 self.wave if evaluateWave is None else evaluateWave,
					 (self.phaseknotloc,self.waveknotloc,m0pars,bsorder,bsorder))

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

	def ErrModel(self,x,bsorder=3,evaluatePhase=None,evaluateWave=None):

		try: errpars = x[self.errmin:self.errmax]
		except: import pdb; pdb.set_trace()

		modelerr = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
						   self.wave if evaluateWave is None else evaluateWave,
						   (self.errphaseknotloc,self.errwaveknotloc,errpars,bsorder,bsorder))
		
		return modelerr

	def getPars(self,loglikes,x,nburn=500,bsorder=3,mkplots=False):

		axcount = 0; parcount = 0
		from matplotlib.backends.backend_pdf import PdfPages
		pdf_pages = PdfPages('output/MCMC_hist.pdf')
		fig = plt.figure()
		
		m0pars = np.array([])
		m0err = np.array([])
		for i in np.where(self.parlist == 'm0')[0]:
			m0pars = np.append(m0pars,x[i][nburn:].mean())#/_SCALE_FACTOR)
			m0err = np.append(m0err,x[i][nburn:].std())#/_SCALE_FACTOR)
			if mkplots:
				if not parcount % 9:
					subnum = axcount%9+1
					ax = plt.subplot(3,3,subnum)
					axcount += 1
					md,std = np.mean(x[i][nburn:]),np.std(x[i][nburn:])
					histbins = np.linspace(md-3*std,md+3*std,50)
					ax.hist(x[i][nburn:],bins=histbins)
					ax.set_title('M0')
					if axcount % 9 == 8:
						pdf_pages.savefig(fig)
						fig = plt.figure()
				parcount += 1

		m1pars = np.array([])
		m1err = np.array([])
		parcount = 0
		for i in np.where(self.parlist == 'm1')[0]:
			m1pars = np.append(m1pars,x[i][nburn:].mean())
			m1err = np.append(m1err,x[i][nburn:].std())
			if mkplots:
				if not parcount % 9:
					subnum = axcount%9+1
					ax = plt.subplot(3,3,subnum)
					axcount += 1
					md,std = np.mean(x[i][nburn:]),np.std(x[i][nburn:])
					histbins = np.linspace(md-3*std,md+3*std,50)
					ax.hist(x[i][nburn:],bins=histbins)
					ax.set_title('M1')
					if axcount % 9 == 8:
						pdf_pages.savefig(fig)
						fig = plt.figure()
				parcount += 1

		# covmat (diagonals only?)
		m0_m1_cov = np.zeros(len(m0pars))
		chain_len = len(m0pars)
		iM0 = self.parlist == 'm0'
		iM1 = self.parlist == 'm1'
		m0mean = np.repeat(x[iM0][:,nburn:].mean(axis=1),np.shape(x[iM0][:,nburn:])[1]).reshape(np.shape(x[iM0][:,nburn:]))
		m1mean = np.repeat(x[iM1][:,nburn:].mean(axis=1),np.shape(x[iM1][:,nburn:])[1]).reshape(np.shape(x[iM1][:,nburn:]))
		m0var = x[iM0][:,nburn:]-m0mean
		m1var = x[iM1][:,nburn:]-m1mean

		for i in range(len(m0pars)):
			for j in range(len(m1pars)):
				if i == j: m0_m1_cov[i] = np.sum(m0var[j]*m1var[i])
		m0_m1_cov /= chain_len


		modelerrpars = np.array([])
		modelerrerr = np.array([])
		for i in np.where(self.parlist == 'modelerr')[0]:
			modelerrpars = np.append(modelerrpars,x[i][nburn:].mean())
			modelerrerr = np.append(modelerrerr,x[i][nburn:].std())

		clpars = np.array([])
		clerr = np.array([])
		for i in np.where(self.parlist == 'cl')[0]:
			clpars = np.append(clpars,x[i][nburn:].mean())
			clerr = np.append(clpars,x[i][nburn:].std())

		clscatpars = np.array([])
		clscaterr = np.array([])
		for i in np.where(self.parlist == 'clscat')[0]:
			clscatpars = np.append(clpars,x[i][nburn:].mean())
			clscaterr = np.append(clpars,x[i][nburn:].std())


		result=np.mean(x[:,nburn:],axis=1)
		resultsdict = {}
		n_sn = len(self.datadict.keys())
		for k in self.datadict.keys():
			tpk_init = self.datadict[k]['photdata']['mjd'][0] - self.datadict[k]['photdata']['tobs'][0]
			resultsdict[k] = {'x0':x[self.parlist == 'x0_%s'%k][0][nburn:].mean(),
							  'x1':x[self.parlist == 'x1_%s'%k][0][nburn:].mean(),
							  'c':x[self.parlist == 'c_%s'%k][0][nburn:].mean(),
							  'tpkoff':x[self.parlist == 'tpkoff_%s'%k][0][nburn:].mean(),
							  'x0err':x[self.parlist == 'x0_%s'%k][0][nburn:].std(),
							  'x1err':x[self.parlist == 'x1_%s'%k][0][nburn:].std(),
							  'cerr':x[self.parlist == 'c_%s'%k][0][nburn:].std(),
							  'tpkofferr':x[self.parlist == 'tpkoff_%s'%k][0][nburn:].std()}


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
		#import pdb; pdb.set_trace()
		clscat = splev(self.wave,(self.errwaveknotloc,clscatpars,3))
		if not len(clpars): clpars = []

		for snpar in ['x0','x1','c','tpkoff']:
			subnum = axcount%9+1
			ax = plt.subplot(3,3,subnum)
			axcount += 1
			md = np.mean(x[self.parlist == '%s_%s'%(snpar,k)][0][nburn:])
			std = np.std(x[self.parlist == '%s_%s'%(snpar,k)][0][nburn:])
			histbins = np.linspace(md-3*std,md+3*std,50)
			ax.hist(x[self.parlist == '%s_%s'%(snpar,k)][0][nburn:],bins=histbins)
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
				phaseIndex=np.searchsorted(self.phasebins,phase,'left')[0]
				waveIndex=np.searchsorted(self.wavebins,restWave,'left')[0]
				self.neff[phaseIndex][waveIndex]+=1
			#For each photometric filter, weight the contribution by  
			for flt in np.unique(photdata['filt']):
				g = (self.wavebins[:-1]  >= filtwave[0]/(1+z)) & (self.wavebins[1:] <= filtwave[-1]/(1+z))  # overlap range
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
		#print(self.neff)
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
				chi2wavegrad+= self.waveres**exponent/((self.wavebins.size-1)**2 *(self.phasebins.size-1)) * (( (fluxvals[:,:,np.newaxis]-fluxvals[:,np.newaxis,:])**2   /   (np.sqrt(self.neff[:,:,np.newaxis]*self.neff[:,np.newaxis,:])* np.abs(self.wavebins[np.newaxis,np.newaxis,:-1]-self.wavebins[np.newaxis,:-1,np.newaxis])**exponent))[:,~np.diag(np.ones(self.wavebins.size-1,dtype=bool))]).sum()
			if gradientPhase != 0:
				chi2phasegrad+= self.phaseres**exponent/((self.phasebins.size-1)**2 *(self.wavebins.size-1) ) * ((  (fluxvals[np.newaxis,:,:]-fluxvals[:,np.newaxis,:])**2   /   (np.sqrt(self.neff[np.newaxis,:,:]*self.neff[:,np.newaxis,:])* np.abs(self.phasebins[:-1,np.newaxis,np.newaxis]-self.phasebins[np.newaxis,:-1,np.newaxis])**exponent))[~np.diag(np.ones(self.phasebins.size-1,dtype=bool)),:]).sum()
			if dyad!= 0:
				chi2dyadvals=(   (fluxvals[:,np.newaxis,:,np.newaxis] * fluxvals[np.newaxis,:,np.newaxis,:] - fluxvals[np.newaxis,:,:,np.newaxis] * fluxvals[:,np.newaxis,np.newaxis,:])**2)   /   (np.sqrt(np.sqrt(self.neff[np.newaxis,:,np.newaxis,:]*self.neff[np.newaxis,:,:,np.newaxis]*self.neff[:,np.newaxis,:,np.newaxis]*self.neff[:,np.newaxis,np.newaxis,:]))*np.abs(self.wavebins[np.newaxis,np.newaxis,:-1,np.newaxis]-self.wavebins[np.newaxis,np.newaxis,np.newaxis,:-1])*np.abs(self.phasebins[:-1,np.newaxis,np.newaxis,np.newaxis]-self.phasebins[np.newaxis,:-1,np.newaxis,np.newaxis]))
				chi2dyad+=self.phaseres*self.waveres/( (self.wavebins.size-1) *(self.phasebins.size-1))**2  * chi2dyadvals[~np.isnan(chi2dyadvals)].sum()
        
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
			return C_0, False
		else:
			# We can always divide M2 by n-1 since n > init_period
			return (s_opt/(self.nsteps_adaptive_memory - 1))*M2, True
	
	def generate_AM_candidate(self, current, M2, n):
		prop_cov,adjust_flag = self.get_proposal_cov(M2, n)

		candidate = np.zeros(self.npar)
		for i,par in zip(range(self.npar),self.parlist):
			if self.adjust_snpars and (par == 'm0' or par == 'm1' or par == 'modelerr'):
				candidate[i] = current[i]
			elif self.adjust_modelpars and par != 'm0' and par != 'm1' and par != 'modelerr':
				candidate[i] = current[i]
			else:
				if par.startswith('x0') or par == 'm0' or par == 'modelerr' or par == 'clscat':
					candidate[i] = current[i]*10**(0.4*np.random.normal(scale=np.sqrt(prop_cov[i,i])))
				else:
					candidate[i] = np.random.normal(loc=current[i],scale=np.sqrt(prop_cov[i,i]))

		return candidate
		
	def lsqguess(self, current, snpars=False, M0=False, M1=False):

		candidate = copy.deepcopy(current)
		
		salterr = self.ErrModel(candidate)
		if self.n_colorpars:
			colorLaw = SALT2ColorLaw(self.colorwaverange, candidate[self.parlist == 'cl'])
		else: colorLaw = None
		if self.n_colorscatpars:
			colorScat = True
		else: colorScat = None

		if snpars:
			print('using scipy minimizer to find SN params...')		

			components = self.SALTModel(candidate)
			for sn in self.datadict.keys():
			
				def lsqwrap(guess):

					candidate[self.parlist == 'x0_%s'%sn] = guess[0]
					candidate[self.parlist == 'x1_%s'%sn] = guess[1]
					candidate[self.parlist == 'c_%s'%sn] = guess[2]
					candidate[self.parlist == 'tpkoff_%s'%sn] = guess[3]
				
					args = (None,sn,candidate,components,salterr,colorLaw,colorScat,False)
					return -self.loglikeforSN(args)


				guess = np.array([candidate[self.parlist == 'x0_%s'%sn][0],candidate[self.parlist == 'x1_%s'%sn][0],
								  candidate[self.parlist == 'c_%s'%sn][0],candidate[self.parlist == 'tpkoff_%s'%sn][0]])
			
				result = minimize(lsqwrap,guess)
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


			
					
			if not (nstep) % self.nsteps_between_lsqfit:
				X = self.lsqguess(current=self.chain[-1],snpars=True)
			else:
				X = self.generate_AM_candidate(current=self.chain[-1], M2=M2_recent, n=nstep)
			#elif not (nstep-2) % self.nsteps_between_lsqfit:
			#	X = self.lsqguess(current=Xlast,M0=True)
			#elif not (nstep-3) % self.nsteps_between_lsqfit:
			#	X = self.lsqguess(current=Xlast,M1=True)
				
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
		
