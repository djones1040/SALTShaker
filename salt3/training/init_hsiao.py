#!/usr/bin/env python
import numpy as np
from scipy.interpolate import bisplrep,bisplev
from scipy.interpolate import interp1d

def init_hsiao(hsiaofile='initfiles/Hsiao07.dat',
			   phaserange=[-20,50],waverange=[2000,9200],phaseinterpres=1.0,
			   waveinterpres=2.0,phasesplineres=3.2,wavesplineres=72):

	phase,wave,flux = np.loadtxt(hsiaofile,unpack=True)
	m1phase = phase*1.1
	splinephase = np.linspace(phaserange[0],phaserange[1],
							  (phaserange[1]-phaserange[0])/phasesplineres)
	splinewave = np.linspace(waverange[0],waverange[1],
							 (waverange[1]-waverange[0])/wavesplineres)
	bspl = bisplrep(phase,wave,flux,kx=3,ky=3,
					tx=splinephase,ty=splinewave,task=-1)

	intphase = np.linspace(phaserange[0],phaserange[1],
						   (phaserange[1]-phaserange[0])/phaseinterpres)
	intwave = np.linspace(waverange[0],waverange[1],
						  (waverange[1]-waverange[0])/waveinterpres)


	m0 = bisplev(intphase,intwave,bspl)
	m0sub = bisplev(np.unique(phase),np.unique(wave),bspl)
	
	bsplm1 = bisplrep(m1phase,wave,
					  flux-m0sub.reshape(len(flux)),kx=3,ky=3,
					  tx=splinephase,ty=splinewave,task=-1)
	m1 = bisplev(intphase,intwave,bsplm1)
	
	return intphase,intwave,m0,m1,bspl[2],bsplm1[2]

def get_hsiao(hsiaofile='initfiles/Hsiao07.dat',
			  phaserange=[-20,50],waverange=[2000,9200],phaseinterpres=1.0,
			  waveinterpres=2.0,phasesplineres=3.2,wavesplineres=72,
			  days_interp=5.0):

	hphase,hwave,hflux = np.loadtxt(hsiaofile,unpack=True)
	hflux = hflux.reshape([len(np.unique(hphase)),len(np.unique(hwave))])
	
	phaseinterp = np.linspace(phaserange[0]-5,phaserange[1]+5,
							  (phaserange[1]-phaserange[0]+2*days_interp)/phaseinterpres)
	waveinterp = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/waveinterpres)

	int1dphase = interp1d(np.unique(hphase),hflux,axis=0)
	hflux_phaseinterp = int1dphase(phaseinterp)
	int1dwave = interp1d(np.unique(hwave),hflux_phaseinterp,axis=1)
	hflux_phasewaveinterp = int1dwave(waveinterp)

	return hflux_phasewaveinterp
