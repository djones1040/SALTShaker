#!/usr/bin/env python

import numpy as np
from scipy.interpolate import splprep,splev,BSpline,griddata,bisplev
from salt3.util.synphot import synphot
from scipy.interpolate import interp1d
from sncosmo.salt2utils import SALT2ColorLaw
import time
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
				if flt == 'filtwave' or flt == 'primarywave' or flt == 'snflux' or flt == 'AB': continue
				if kcordict[survey][flt]['magsys'] == 'AB': primarykey = 'AB'
				elif kcordict[survey][flt]['magsys'] == 'Vega': primarykey = 'Vega'
				self.stdmag[survey][flt] = synphot(primarywave,kcordict[survey][primarykey],filtwave=self.kcordict[survey]['filtwave'],
												   filttp=kcordict[survey][flt]['filttrans'],
												   zpoff=0)#kcordict[survey][flt]['zpoff'])

	def chi2fit(self,x,onlySNpars=False,debug=False,debug2=False):

		# TODO: fit to t0
		
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
			specdata = self.datadict[sn]['specdata']
			survey = self.datadict[sn]['survey']
			filtwave = self.kcordict[survey]['filtwave']
			z = self.datadict[sn]['zHelio']
			obswave = self.wave*(1+z)
			
			x0,x1,c,tpkoff = \
				x[self.parlist == 'x0_%s'%sn][0],x[self.parlist == 'x1_%s'%sn][0],\
				x[self.parlist == 'c_%s'%sn][0],x[self.parlist == 'tpkoff_%s'%sn][0]
			if self.n_components == 1:
				saltflux = x0*M0
			elif self.n_components == 2:
				saltflux = x0*(M0 + x1*M1)
			if self.n_colorpars:
				self._colorlaw = SALT2ColorLaw(self.colorwaverange, x[self.parlist == 'cl'])
				saltflux *= 10. ** (-0.4 * self._colorlaw(self.wave) * c)
				if debug2: import pdb; pdb.set_trace()

				
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

				for t,f,fe,flt in zip(photdata['tobs'][photdata['filt'] == flt],photdata['fluxcal'][photdata['filt'] == flt],
									  photdata['fluxcalerr'][photdata['filt'] == flt],photdata['filt'][photdata['filt'] == flt]):
					# HACK - will need to change this when we fit for t0
					if t < self.phaserange[0] or t > self.phaserange[1]: continue

					try:
						saltfluxinterp = int1d(t+tpkoff)
					except:
						import pdb; pdb.set_trace()
						
					# synthetic photometry from SALT model
					modelsynflux = np.trapz(pbspl*saltfluxinterp[g],obswave[g])/denom
					modelflux = modelsynflux*10**(-0.4*self.kcordict[survey][flt]['zpoff'])*10**(0.4*self.stdmag[survey][flt])*10**(0.4*27.5)

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

		if debug2: import pdb; pdb.set_trace()
		if onlySNpars: print(chi2,x,tpkoff)
		else: print(chi2,x[0],x[self.parlist == 'x0_ASASSN-16bc'],x[self.parlist == 'cl'])
		if chi2 != chi2:
			import pdb; pdb.set_trace()
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
