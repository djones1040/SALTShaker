#!/usr/bin/env python

import numpy as np
from scipy.interpolate import splprep,splev,BSpline,griddata,bisplev
from salt3.util.synphot import synphot
from scipy.interpolate import interp1d
#import pysynphot as S

lambdaeff = {'g':4900.1409,'r':6241.2736,'i':7563.7672,'z':8690.0840}

class chi2:
	def __init__(self,guess,datadict,parlist,phaserange,waverange,phaseres,waveres,phaseoutres,waveoutres,
				 kcordict,n_components=1,n_colorpars=0):
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

		assert type(parlist) == np.ndarray
		
		self.splinephase = np.linspace(phaserange[0],phaserange[1],(phaserange[1]-phaserange[0])/phaseres)
		self.splinewave = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/waveres)
		self.phase = np.linspace(phaserange[0],phaserange[1],(phaserange[1]-phaserange[0])/phaseoutres)
		self.wave = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/waveoutres)

		self.components = self.SALTModel(guess)
		
		self.stdmag = {}
		for survey in self.kcordict.keys():
			self.stdmag[survey] = {}
			primarywave = kcordict[survey]['primarywave']
			for flt in self.kcordict[survey].keys():
				if flt == 'filtwave' or flt == 'primarywave' or flt == 'snflux' or flt == 'AB': continue
				if kcordict[survey][flt]['magsys'] == 'AB': primarykey = 'AB'
				elif kcordict[survey][flt]['magsys'] == 'Vega': primarykey = 'Vega'
				self.stdmag[survey][flt] = synphot(primarywave,kcordict[survey][primarykey],filtwave=self.kcordict[survey]['filtwave'],
												   filttp=kcordict[survey][flt]['filttrans'],
												   zpoff=0)#kcordict[survey][flt]['zpoff'])

	def chi2fit(self,x,onlySNpars=False,debug=False):

		if debug:
			import pylab as plt
			plt.ion()
			plt.clf()

		if onlySNpars:
			components = self.components
		else:
			components = self.SALTModel(x)
		if self.n_components == 1: M0 = components[0]
		elif self.n_components == 2: M0,M1 = components

		
		chi2 = 0
		tused = 0
		for sn in self.datadict.keys():
			photdata = self.datadict[sn]['photdata']
			survey = self.datadict[sn]['survey']
			filtwave = self.kcordict[survey]['filtwave']
			z = self.datadict[sn]['zHelio']
			
			x0,x1,c = x[self.parlist == 'x0_%s'%sn],x[self.parlist == 'x1_%s'%sn],x[self.parlist == 'c_%s'%sn]
			if self.n_components == 1:
				saltflux = x0*M0
			elif self.n_components == 2:
				saltflux = x0*(M0 + x1*M1)
			if self.n_colorpars:
				saltflux *= np.exp(c*colorlaw(saltwave,x[self.parlist == 'cl']))
			if len(saltflux[saltflux < 0]):
				return 1e50
			
			int1d = interp1d(self.phase,saltflux,axis=0)

			
			for t,f,fe,flt in zip(photdata['tobs'],photdata['fluxcal'],photdata['fluxcalerr'],photdata['filt']):
				# HACK - will need to change this when we fit for t0
				if t < self.phaserange[0] or t > self.phaserange[1]: continue

				saltfluxinterp = int1d(t)
				
				# synthetic photometry from SALT model
				modelphot = self.synflux(saltfluxinterp,self.kcordict[survey][flt]['zpoff'],
										 survey=survey,flt=flt,redshift=z)
				modelflux = 10**(-0.4*(modelphot-self.stdmag[survey][flt]-27.5))
				# chi2 function
				# TODO - model error/dispersion parameters
				chi2 += (f-modelflux)**2./fe**2.
				if debug:
					if t == photdata['tobs'][0]:
						plt.errorbar(t,modelflux,fmt='o',color='C0',label='model')
						plt.errorbar(t,f,yerr=fe,fmt='o',color='C1',label='obs')
					else:
						plt.errorbar(t,modelflux,fmt='o',color='C0')
						plt.errorbar(t,f,yerr=fe,fmt='o',color='C1')
				
		if debug:
			import pdb; pdb.set_trace()
			plt.close()
		#if debug:
		#	import pylab as plt
		#	plt.ion()
		#	plt.plot(saltwave,M0[14,:])
		#	import pdb; pdb.set_trace()
		print(chi2,x[0],x[self.parlist == 'x0_ASASSN-16bc'])
		if chi2 != chi2:
			import pdb; pdb.set_trace()
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
	
		return components

	def colorlaw(self,wavelength,colorpars):
		colormod = 0
		for i,n in zip(range(len(colorpars)),colorpars):
			colormod += n*wavelength**i
		#TODO: linear extrapolation
		return colormod
		
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
			resultsdict[k] = {'x0':x[self.parlist == 'x0_%s'%k],'x1':x[self.parlist == 'x0_%s'%k]}

		return self.phase,self.wave,m0,m1,clpars,resultsdict

	def synflux(self,spc,zpoff,survey=None,flt=None,redshift=0):
		x = self.wave*(1+redshift)
		pbphot = 1

		pbx = self.kcordict[survey]['filtwave']
		pby = self.kcordict[survey][flt]['filttrans']

		g = (x >= pbx[0]) & (x <= pbx[-1])  # overlap range

		pbspl = np.interp(x[g],pbx,pby)
		pbspl *= x[g]

		res = np.trapz(pbspl*spc[g],x[g])/np.trapz(pbspl,x[g])

		return(zpoff-2.5*np.log10(res))
