#!/usr/bin/env python

import numpy as np
from scipy.interpolate import splprep,splev,BSpline,griddata,bisplev
from salt3.util.synphot import synphot
from scipy.interpolate import interp1d
from sncosmo.salt2utils import SALT2ColorLaw
import time
from itertools import starmap
#import pysynphot as S

lambdaeff = {'g':4900.1409,'r':6241.2736,'i':7563.7672,'z':8690.0840}

class chi2:
	def __init__(self,guess,datadict,parlist,phaserange,waverange,phaseres,waveres,phaseoutres,waveoutres,
				 colorwaverange,kcordict,n_components=1,n_colorpars=0):
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
		
		assert type(parlist) == np.ndarray
		self.splinephase = np.linspace(phaserange[0],phaserange[1],(phaserange[1]-phaserange[0])/phaseres)
		self.splinewave = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/waveres)
		self.phase = np.linspace(phaserange[0]-5,phaserange[1]+5,(phaserange[1]-phaserange[0])/phaseoutres)
		self.wave = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/waveoutres)

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
												   zpoff=0)#kcordict[survey][flt]['zpoff'])

	def chi2fit(self,x,onlySNpars=False,pool=None,debug=False,debug2=False):
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
		
		if debug:
			import pylab as plt
			plt.ion()
			plt.clf()
		
		#Set up SALT model
		if onlySNpars:
			components = self.components
		else:
			components = self.SALTModel(x)
		if self.n_components == 1: M0 = components[0]
		elif self.n_components == 2: M0,M1 = components
		if self.n_colorpars:
			colorLaw = SALT2ColorLaw(self.colorwaverange, x[self.parlist == 'cl'])
		
		chi2 = 0
		#Construct arguments for chi2forSN method
		args=[(sn,x,components,colorLaw,onlySNpars,False,False) for sn in self.datadict.keys()]
		
		#If worker pool available, use it to calculate chi2 for each SN; otherwise, do it in this process
		if pool:
			chi2=sum(pool.starmap(self.chi2forSN,args))
		else:
			chi2=sum(starmap(self.chi2forSN,args))
			
		#Debug statements
		if debug:
			import pdb; pdb.set_trace()
			plt.close()
		if debug2: import pdb; pdb.set_trace()
		if onlySNpars: print(chi2,x)
		else: print(chi2,x[0],x[self.parlist == 'x0_ASASSN-16bc'],x[self.parlist == 'cl'])
		
		return chi2
		
	def chi2forSN(self,sn,x,components=None,colorLaw=None,onlySNpars=False,debug=False,debug2=False):
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
	
		x0,x1,c,tpkoff = \
			x[self.parlist == 'x0_%s'%sn][0],x[self.parlist == 'x1_%s'%sn][0],\
			x[self.parlist == 'c_%s'%sn][0],x[self.parlist == 'tpkoff_%s'%sn][0]
			
		#Calculate spectral model
		if self.n_components == 1:
			saltflux = x0*M0
		elif self.n_components == 2:
			saltflux = x0*(M0 + x1*M1)
		if colorLaw:
			saltflux *= 10. ** (-0.4 * colorLaw(self.wave) * c)
			if debug2: import pdb; pdb.set_trace()

		chi2=0
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
			chi2 += ((filtPhot['fluxcal']-modelflux)**2./filtPhot['fluxcalerr']**2.).sum()
			if debug:
					plt.errorbar(filtPhot['tobs'],modelflux,fmt='o',color='C0',label='model')
					plt.errorbar(filtPhot['tobs'],filtPhot['fluxcal'],yerr=filtPhot['fluxcalerr'],fmt='o',color='C1',label='obs')
		return chi2
		
		
	def specchi2(self):

		return chi2
	
	def SALTModel(self,x,bsorder=3):
		# parlist,phaserange,waverange,phaseres,waveres,phaseoutres,waveoutres
		
		m0pars = x[self.parlist == 'm0']
		m0 = bisplev(self.phase,self.wave,(self.splinephase,self.splinewave,m0pars,bsorder,bsorder))
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

	def synflux(self,sourceflux,zpoff,survey=None,flt=None,redshift=0):
		obswave = self.wave*(1+redshift)

		filtwave = self.kcordict[survey]['filtwave']
		filttrans = self.kcordict[survey][flt]['filttrans']

		g = (obswave >= filtwave[0]) & (obswave <= filtwave[-1])  # overlap range

		pbspl = np.interp(obswave[g],filtwave,filttrans)
		pbspl *= obswave[g]

		res = np.trapz(pbspl*sourceflux[g],obswave[g])/np.trapz(pbspl,obswave[g])

		return(zpoff-2.5*np.log10(res))
