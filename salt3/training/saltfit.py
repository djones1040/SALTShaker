#!/usr/bin/env python

import numpy as np
from scipy.interpolate import splprep,splev,BSpline,griddata,bisplev
from salt3.util.synphot import synphot
from scipy.interpolate import interp1d
from sncosmo.salt2utils import SALT2ColorLaw
import time
from itertools import starmap
from salt3.training import init_hsiao
from sncosmo.models import StretchSource
from scipy.optimize import minimize
from scipy.stats import norm
#import pysynphot as S

_SCALE_FACTOR = 1e-12

lambdaeff = {'g':4900.1409,'r':6241.2736,'i':7563.7672,'z':8690.0840}

class chi2:
	def __init__(self,guess,datadict,parlist,phaseknotloc,waveknotloc,
				 phaserange,waverange,phaseres,waveres,phaseoutres,waveoutres,
				 colorwaverange,kcordict,initmodelfile,initBfilt,n_components=1,
				 n_colorpars=0,days_interp=5,onlySNpars=False,mcmc=False,
				 fitstrategy='leastsquares'):
		self.datadict = datadict
		self.parlist = parlist
		self.phaserange = phaserange
		self.waverange = waverange
		self.phaseres = phaseres
		self.phaseoutres = phaseoutres
		self.waveoutres = waveoutres
		self.kcordict = kcordict
		self.n_components = n_components
		self.n_colorpars = n_colorpars
		self.colorwaverange = colorwaverange
		self.onlySNpars = onlySNpars
		self.mcmc = mcmc
		self.fitstrategy = fitstrategy
		self.guess = guess
		
		assert type(parlist) == np.ndarray
		self.splinephase = phaseknotloc #np.linspace(phaserange[0],phaserange[1],(phaserange[1]-phaserange[0])/phaseres)
		self.splinewave = waveknotloc #np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/waveres)
		self.phase = np.linspace(phaserange[0]-days_interp,phaserange[1]+days_interp,
								 (phaserange[1]-phaserange[0]+2*days_interp)/phaseoutres)
		self.wave = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/waveoutres)

		self.hsiaoflux = init_hsiao.get_hsiao(hsiaofile=initmodelfile,Bfilt=initBfilt,
											  phaserange=phaserange,waverange=waverange,
											  phaseinterpres=phaseoutres,waveinterpres=waveoutres,
											  phasesplineres=phaseres,wavesplineres=waveres,
											  days_interp=days_interp)
		
		self.extrapsource = StretchSource(self.phase,self.wave,self.hsiaoflux)
		self.extrapsource.set(amplitude=1,s=1)
		# sample every 50 wavelengths to make fitting faster
		self.extrapidx = np.arange(0,len(self.wave),waveoutres*25,dtype='int')
		self.iExtrapFittingPhaseMin = self.phase < 0
		self.iExtrapFittingPhaseMax = self.phase > 40

		self.iExtrapFittingPhaseMin2D = np.zeros([len(self.phase),len(self.wave)],dtype='bool')
		self.iExtrapFittingPhaseMax2D = np.zeros([len(self.phase),len(self.wave)],dtype='bool')
		for p in range(len(self.phase)):
			for w in range(len(self.wave)):
				if self.phase[p] < 0 and w in self.extrapidx:
					self.iExtrapFittingPhaseMin2D[p,w] = True
				if self.phase[p] > 40 and w in self.extrapidx:
					self.iExtrapFittingPhaseMax2D[p,w] = True
		
		self.iExtrapPhaseMin = self.phase < phaserange[0]
		self.iExtrapPhaseMax = self.phase > phaserange[1]
		
		self.components = self.SALTModel(guess)
		
		self.stdmag = {}
		for survey in self.kcordict.keys():
			self.stdmag[survey] = {}
			primarywave = kcordict[survey]['primarywave']
			for flt in self.kcordict[survey].keys():
				if flt == 'filtwave' or flt == 'primarywave' or flt == 'snflux' or flt == 'AB' or flt == 'BD17': continue
				if kcordict[survey][flt]['magsys'] == 'AB': primarykey = 'AB'
				elif kcordict[survey][flt]['magsys'] == 'Vega': primarykey = 'Vega'
				elif kcordict[survey][flt]['magsys'] == 'BD17': primarykey = 'BD17'
				self.stdmag[survey][flt] = synphot(primarywave,kcordict[survey][primarykey],filtwave=self.kcordict[survey]['filtwave'],
												   filttp=kcordict[survey][flt]['filttrans'],
												   zpoff=0) - kcordict[survey][flt]['primarymag'] #kcordict[survey][flt]['zpoff'])
				
	def extrapolate(self,saltflux,x0):

		def errfunc_min(params):
			self.extrapsource.set(amplitude=params[0],s=params[1])
			if params[1] <= 0: return np.inf
			return np.sum(abs(saltflux[self.iExtrapFittingPhaseMin2D] - self.extrapsource.flux(self.phase[self.iExtrapFittingPhaseMin], self.wave[self.extrapidx]).flatten()))
		def errfunc_max(params):
			self.extrapsource.set(amplitude=params[0],s=params[1])
			if params[1] <= 0: return np.inf
			return np.sum(abs(saltflux[self.iExtrapFittingPhaseMax2D] - self.extrapsource.flux(self.phase[self.iExtrapFittingPhaseMax], self.wave[self.extrapidx]).flatten()))

		guess = (x0,1)
		MinResult = minimize(errfunc_min,guess,method='Nelder-Mead')
		MaxResult = minimize(errfunc_max,guess,method='Nelder-Mead')

		saltfluxbkp = saltflux[:]
		
		self.extrapsource.set(amplitude=MinResult.x[0],s=MinResult.x[1])
		saltflux[self.iExtrapPhaseMin,:] = self.extrapsource.flux(self.phase[self.iExtrapPhaseMin],self.wave)
		self.extrapsource.set(amplitude=MaxResult.x[0],s=MaxResult.x[1])
		saltflux[self.iExtrapPhaseMax,:] = self.extrapsource.flux(self.phase[self.iExtrapPhaseMax],self.wave)

		return saltflux
		
	def chi2fit(self,x,ndim=None,nparams=None,pool=None,SNpars=(),SNparlist=(),debug=False,debug2=False):
		"""
		Calculates the goodness of fit of given SALT model to photometric and spectroscopic data given during initialization
		
		Parameters
		----------
		x : array
			SALT model parameters
			
		onlySNpars : boolean, optional
			Only fit the individual SN parameters, while retaining fixed model parameters
			
		pool : 	multiprocessing.pool.Pool, optional
			Optional worker pool to be used for calculating chi2 values for each SN. If not provided, all work is done in root process
		
		debug : boolean, optional
		debug2 : boolean, optional
			Debug flags
		
		Returns
		-------
		
		chi2: float
			Goodness of fit of model to training data	
		"""
		# TODO: fit to t0

		x = np.array(x)
		
		#Set up SALT model
		if self.onlySNpars:
			components = self.components
		else:
			components = self.SALTModel(x)

		if self.n_components == 1: M0 = components[0]
		elif self.n_components == 2: M0,M1 = components
		if self.n_colorpars:
			colorLaw = SALT2ColorLaw(self.colorwaverange, x[self.parlist == 'cl'])
		else: colorLaw = None

		chi2 = 0
		#Construct arguments for chi2forSN method
		args=[(sn,x,components,SNpars,SNparlist,colorLaw,self.onlySNpars,debug,debug2) for sn in self.datadict.keys()]
		
		#If worker pool available, use it to calculate chi2 for each SN; otherwise, do it in this process
		if pool:
			chi2=sum(pool.starmap(self.chi2forSN,args))
		else:
			chi2=sum(starmap(self.chi2forSN,args))
		
		#lnp = self.prior(M0[abs(self.phase) == np.min(abs(self.phase)),:][0])
		#chi2 += lnp
		
		#Debug statements
		#if debug2: import pdb; pdb.set_trace()
		#if self.onlySNpars: print(chi2,x)
		#else: print(chi2,x[0],x[self.parlist == 'x0_ASASSN-16bc'],x[self.parlist == 'cl'])
		print(chi2.sum())
		
		if self.mcmc:
			return -chi2
		else:
			return chi2
		
	def chi2forSN(self,sn,x,components=None,SNpars=(),SNparlist=(),
				  colorLaw=None,onlySNpars=False,
				  debug=False,debug2=False):
		"""
		Calculates the goodness of fit of given SALT model to photometric and spectroscopic observations of a single SN 
		
		Parameters
		----------
		sn : str
			Name of supernova to compare to model
			
		x : array
			SALT model parameters
			
		components: array_like, optional
			SALT model components, if not provided will be derived from SALT model parameters passed in \'x\'
		
		colorLaw: function, optional
			SALT color law which takes wavelength as an argument

		onlySNpars : boolean, optional
			Only fit the individual SN parameters, while retaining fixed model parameters
					
		debug : boolean, optional
		debug2 : boolean, optional
			Debug flags
		
		Returns
		-------
		chi2: float
			Model chi2 relative to training data	
		"""
		x = np.array(x)
		
		#Set up SALT model
		if components is None:
			if onlySNpars:
				components = self.components
			else:
				components = self.SALTModel(x)
		if self.n_components == 1: M0 = components[0]
		elif self.n_components == 2: M0,M1 = components

		#Declare variables
		photdata = self.datadict[sn]['photdata']
		specdata = self.datadict[sn]['specdata']
		survey = self.datadict[sn]['survey']
		filtwave = self.kcordict[survey]['filtwave']
		z = self.datadict[sn]['zHelio']
		obswave = self.wave*(1+z)
	
		if not len(SNpars):
			x0,x1,c,tpkoff = \
				x[self.parlist == 'x0_%s'%sn][0],x[self.parlist == 'x1_%s'%sn][0],\
				x[self.parlist == 'c_%s'%sn][0],x[self.parlist == 'tpkoff_%s'%sn][0]
		else:
			x0,x1,c,tpkoff = \
				SNpars[SNparlist == 'x0_%s'%sn][0][0],SNpars[SNparlist == 'x1_%s'%sn][0][0],\
				SNpars[SNparlist == 'c_%s'%sn][0][0],SNpars[SNparlist == 'tpkoff_%s'%sn][0][0]

		#Calculate spectral model
		if self.n_components == 1:
			saltflux = x0*M0/_SCALE_FACTOR
		elif self.n_components == 2:
			saltflux = x0*(M0*_SCALE_FACTOR + x1*M1*_SCALE_FACTOR)
		if colorLaw:
			saltflux *= 10. ** (-0.4 * colorLaw(self.wave) * c)
			if debug2: import pdb; pdb.set_trace()
		saltflux = self.extrapolate(saltflux,x0)

		if self.fitstrategy == 'leastsquares': chi2 = np.array([])
		else: chi2 = 0
		int1d = interp1d(self.phase,saltflux,axis=0)
		for k in specdata.keys():
			if specdata[k]['tobs'] < self.phaserange[0] or specdata[k]['tobs'] > self.phaserange[1]: continue
			saltfluxinterp = int1d(specdata[k]['tobs']+tpkoff)
			saltfluxinterp2 = np.interp(specdata[k]['wavelength'],obswave,saltfluxinterp)
			chi2 += np.sum((saltfluxinterp2-specdata[k]['flux'])**2./specdata[k]['fluxerr']**2.)
			
		for flt in np.unique(photdata['filt']):
			# synthetic photometry
			filtwave = self.kcordict[survey]['filtwave']
			filttrans = self.kcordict[survey][flt]['filttrans']

			g = (obswave >= filtwave[0]) & (obswave <= filtwave[-1])  # overlap range

			pbspl = np.interp(obswave[g],filtwave,filttrans)
			pbspl *= obswave[g]

			denom = np.trapz(pbspl,obswave[g])
			
			#Select data from the appropriate time range and filter
			selectFilter=(photdata['filt']==flt)&(photdata['tobs']>self.phaserange[0]) & (photdata['tobs']<self.phaserange[1])
			filtPhot={key:photdata[key][selectFilter] for key in photdata}
			try:
				#Array output indices match time along 0th axis, wavelength along 1st axis
				saltfluxinterp = int1d(filtPhot['tobs']+tpkoff)
			except:
				import pdb; pdb.set_trace()
			# synthetic photometry from SALT model
			# Integrate along wavelength axis
			modelsynflux=np.trapz(pbspl[np.newaxis,:]*saltfluxinterp[:,g],obswave[g],axis=1)/denom
			modelflux = modelsynflux*10**(-0.4*self.kcordict[survey][flt]['zpoff'])*10**(0.4*self.stdmag[survey][flt])*10**(0.4*27.5)

			# chi2 function
			# TODO - model error/dispersion parameters
			#chi2 += ((filtPhot['fluxcal']-modelflux)**2./filtPhot['fluxcalerr']**2.).sum()
			if self.fitstrategy == 'leastsquares':
				chi2 = np.append(chi2,(filtPhot['fluxcal']-modelflux)**2./(filtPhot['fluxcal']*0.05)**2.)
			else:
				chi2 += ((filtPhot['fluxcal']-modelflux)**2./(filtPhot['fluxcal']*0.05)**2.).sum()
			if debug:
				if chi2 > 1357: continue
				import pylab as plt
				plt.ion()
				plt.clf()
				plt.errorbar(filtPhot['tobs'],modelflux,fmt='o',color='C0',label='model')
				plt.errorbar(filtPhot['tobs'],filtPhot['fluxcal'],yerr=filtPhot['fluxcalerr'],fmt='o',color='C1',label='obs')

				hint1d = interp1d(self.phase,self.hsiaoflux,axis=0)
				hsiaofluxinterp = hint1d(filtPhot['tobs']+tpkoff)
				hsiaomodelsynflux=np.trapz(pbspl[np.newaxis,:]*hsiaofluxinterp[:,g],obswave[g],axis=1)/denom
				hsiaomodelflux = hsiaomodelsynflux*10**(-0.4*self.kcordict[survey][flt]['zpoff'])*10**(0.4*self.stdmag[survey][flt])*10**(0.4*27.5)
				#plt.errorbar(filtPhot['tobs'],hsiaomodelflux,fmt='o',color='C2',label='hsiao model')
				#import pdb; pdb.set_trace()
				
				if chi2 < 1357: import pdb; pdb.set_trace()

		return chi2
		
		
	def specchi2(self):

		return chi2
	
	def SALTModel(self,x,bsorder=3):
		# parlist,phaserange,waverange,phaseres,waveres,phaseoutres,waveoutres
		x = np.array(x)
		#self.parlist = np.array(self.parlist)
		
		m0pars = x[self.parlist == 'm0']
		m0 = bisplev(self.phase,self.wave,(self.splinephase,self.splinewave,m0pars,bsorder,bsorder))

		#import pdb; pdb.set_trace()
		# extrapolate
		#import pdb; pdb.set_trace()
		#iPhaseLow = (self.phase - self.phaserange[0])**2. == np.min((self.phase - self.phaserange[0])**2.)
		#LowScale = np.mean(m0[iPhaseLow]/self.hsiaoflux[iPhaseLow])
		#iPhaseHigh = (self.phase - self.phaserange[1])**2. == np.min((self.phase - self.phaserange[1])**2.)
		#HighScale = np.mean(m0[iPhaseHigh]/self.hsiaoflux[iPhaseHigh])
		
		#m0[self.phase < self.phaserange[0],:] = self.hsiaoflux[self.phase < self.phaserange[0],:]*LowScale
		#m0[self.phase > self.phaserange[1],:] = self.hsiaoflux[self.phase > self.phaserange[1],:]*HighScale
		#self.hsiaofluxlow
		#self.hsiaofluxhigh


		if self.n_components == 2:
			m1pars = x[self.parlist == 'm1']
			m1 = bisplev(self.phase,self.wave,(self.splinephase,self.splinewave,m1pars,bsorder,bsorder))
			components = (m0,m1)
		elif self.n_components == 1:
			components = (m0,)
		else:
			raise RuntimeError('A maximum of two principal components is allowed')
			
		return components
		
	def getPars(self,x,bsorder=3):

		m0pars = x[self.parlist == 'm0']
		m1pars = x[self.parlist == 'm1']
		clpars = x[self.parlist == 'cl']
	
		m0 = bisplev(self.phase,self.wave,(self.splinephase,self.splinewave,m0pars,bsorder,bsorder))
		if len(m1pars):
			m1 = bisplev(self.phase,self.wave,(self.splinephase,self.splinewave,m1pars,bsorder,bsorder))
		else: m1 = np.zeros(np.shape(m0))
		if not len(clpars): clpars = []
	
		resultsdict = {}
		n_sn = len(self.datadict.keys())
		for k in self.datadict.keys():
			tpk_init = self.datadict[k]['photdata']['mjd'][0] - self.datadict[k]['photdata']['tobs'][0]
			resultsdict[k] = {'x0':x[self.parlist == 'x0_%s'%k],
							  'x1':x[self.parlist == 'x1_%s'%k],
							  'c':x[self.parlist == 'x1_%s'%k],
							  't0':x[self.parlist == 'tpkoff_%s'%k]+tpk_init}
			
		return self.phase,self.wave,m0,m1,clpars,resultsdict

	def synphot(self,sourceflux,zpoff,survey=None,flt=None,redshift=0):
		obswave = self.wave*(1+redshift)

		filtwave = self.kcordict[survey]['filtwave']
		filttrans = self.kcordict[survey][flt]['filttrans']

		g = (obswave >= filtwave[0]) & (obswave <= filtwave[-1])  # overlap range

		pbspl = np.interp(obswave[g],filtwave,filttrans)
		pbspl *= obswave[g]

		res = np.trapz(pbspl*sourceflux[g],obswave[g])/np.trapz(pbspl,obswave[g])
		return(zpoff-2.5*np.log10(res))

	def prior(self,saltflux,prior_mean=-19.36,prior_std=0.1):
		
		Bmag = self.synphot(saltflux/_SCALE_FACTOR,0,survey='SSS',flt='B')-self.stdmag['SSS']['B']
		p_theta = 1 + norm.logpdf(Bmag,prior_mean,prior_std)
		print(Bmag)
		return -p_theta
