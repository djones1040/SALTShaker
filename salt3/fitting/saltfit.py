#!/usr/bin/env python

import numpy as np
from scipy.interpolate import splprep,splev,BSpline,griddata,bisplev
from salt3.util.synphot import synphot

def chi2fit(x,datadict,parlist,phaserange,waverange,phaseres,waveres,kcordict):

	saltphase,saltwave,M0,M1 = SALTModel(x,parlist,phaserange,waverange,phaseres,waveres)
	
	chi2 = 0
	for sn in datadict.keys():
		photdata = datadict[sn]['photdata']
		survey = datadict[sn]['survey']
		filtwave = kcordict[survey]['filtwave']
		primarywave = kcordict[survey]['primarywave']

		for t,f,fe,flt in zip(photdata['tobs'],photdata['fluxcal'],photdata['fluxcalerr'],photdata['filt']):
			# interpolate model to correct phase

			# HACK - will need to change this when we fit for t0
			if t < phaserange[0] or t > phaserange[1]: continue
			
			x0,x1,c = x[parlist == 'x0_%s'%sn],x[parlist == 'x1_%s'%sn],x[parlist == 'c_%s'%sn]
			saltflux = x0*(M0 + x1*M1)
			saltfluxinterp = np.array([np.interp(t,saltphase,saltflux[:,i]) for i in range(len(saltwave))])
			
			# synthetic photometry from SALT model
			if kcordict[survey][flt]['magsys'] == 'AB': primarykey = 'AB'
			elif kcordict[survey][flt]['magsys'] == 'Vega': primarykey = 'Vega'
			modelphot = synphot(saltwave,saltfluxinterp,primarywave=primarywave,
								primaryflux=kcordict[survey][primarykey],filtwave=filtwave,filttp=kcordict[survey][flt]['filttrans'],
								zpoff=kcordict[survey][flt]['zpoff'])
			modelflux = 10**(-0.4*(modelphot-27.5))
			print(modelphot)
			# chi2 function
			# TODO - model error/dispersion parameters
			chi2 += (f-modelflux)**2./fe**2.	
	import pdb; pdb.set_trace()
	print(chi2)
	return chi2
	
def SALTModel(x,parlist,phaserange,waverange,phasesplineres,wavesplineres,
			  phaseinterpres=1,waveinterpres=2,bsorder=3):

	m0pars = x[parlist == 'm0']
	m1pars = x[parlist == 'm1']
	
	splinephase = np.linspace(phaserange[0],phaserange[1],(phaserange[1]-phaserange[0])/phasesplineres)
	splinewave = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/wavesplineres)
	phase = np.linspace(phaserange[0],phaserange[1],(phaserange[1]-phaserange[0])/phaseinterpres)
	wave = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/waveinterpres)

	m0 = bisplev(phase,wave,(splinephase,splinewave,m0pars,bsorder,bsorder))
	m1 = bisplev(phase,wave,(splinephase,splinewave,m1pars,bsorder,bsorder))
	
	return phase,wave,m0,m1

def getPars(x,parlist,datadict,phaserange,waverange,phasesplineres,phasewaveres,
			phaseinterpres=1,waveinterpres=2,bsorder=3):

	m0pars = x[parlist == 'm0']
	m1pars = x[parlist == 'm1']
	
	splinephase = np.linspace(phaserange[0],phaserange[1],(phaserange[1]-phaserange[0])/phasesplineres)
	splinewave = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/wavesplineres)
	phase = np.linspace(phaserange[0],phaserange[1],(phaserange[1]-phaserange[0])/phaseinterpres)
	wave = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/waveinterpres)

	m0 = bisplev(phase,wave,(splinephase,splinewave,m0pars,bsorder,bsorder))
	m1 = bisplev(phase,wave,(splinephase,splinewave,m1pars,bsorder,bsorder))
	resultsdict = {}
	n_sn = len(datadict.keys())
	for k in datadict.keys():
		resultsdict[k] = {'x0':x[parlist == 'x0_%s'%k],'x1':x[parlist == 'x0_%s'%k]}

	return phase,wave,m0,m1,resultsdict
