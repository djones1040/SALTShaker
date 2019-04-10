#!/usr/bin/env python
import numpy as np
from scipy.interpolate import bisplrep,bisplev
from scipy.interpolate import interp1d

_SCALE_FACTOR = 1e-12

def init_hsiao(hsiaofile='initfiles/Hsiao07.dat',
			   salt2file='initfiles/salt2_template_0.dat.gz',
			   Bfilt='initfiles/Bessell90_B.dat',
			   flatnu='initfiles/flatnu.dat',
			   phaserange=[-20,50],waverange=[2000,9200],phaseinterpres=1.0,
			   waveinterpres=2.0,phasesplineres=3.2,wavesplineres=72,
			   days_interp=5,debug=False):

	phase,wave,flux = np.loadtxt(hsiaofile,unpack=True)
	refWave,refFlux=np.loadtxt(flatnu,unpack=True)
	#sphase,swave,sflux = np.loadtxt(hsiaofile,unpack=True)
	#HBmag = synphotB(wave[phase == 0],flux[phase == 0],0,Bfilt=Bfilt)
	#SBmag = synphotB(swave[phase == 0],sflux[sphase == 0],0,Bfilt=Bfilt)
	#import pdb; pdb.set_trace()
	
	#flux /= 4*np.pi*(10*3.086e18)**2.
	#flux *= 10**(0.4*19.36)*_SCALE_FACTOR
	flux *= 10**(-0.4*(10.635+(synphotB(refWave,refFlux,0,0,Bfilt)-synphotB(wave[phase==0],flux[phase==0],0,0,Bfilt))))*_SCALE_FACTOR
	
	m1phase = phase*1.1
	splinephase = np.linspace(phaserange[0],phaserange[1],
							  (phaserange[1]-phaserange[0])/phasesplineres,False)
	splinewave = np.linspace(waverange[0],waverange[1],
							 (waverange[1]-waverange[0])/wavesplineres,False)
	bspl = bisplrep(phase,wave,flux,kx=3,ky=3,
					tx=splinephase,ty=splinewave,task=-1)

	intphase = np.linspace(phaserange[0],phaserange[1],
						   (phaserange[1]-phaserange[0])/phaseinterpres,False)
	intwave = np.linspace(waverange[0],waverange[1],
						  (waverange[1]-waverange[0])/waveinterpres,False)


	m0 = bisplev(intphase,intwave,bspl)
	m0sub = bisplev(np.unique(phase),np.unique(wave),bspl)
	
	bsplm1 = bisplrep(m1phase,wave,
					  flux-m0sub.reshape(len(flux)),kx=3,ky=3,
					  tx=splinephase,ty=splinewave,task=-1)
	m1 = bisplev(intphase,intwave,bsplm1)
	if debug:
		import pylab as plt
		plt.ion()
		plt.plot(wave[phase == 0],flux[phase == 0],label='hsiao, phase = 0 days')
		#plt.plot(wave[phase == 1],flux[phase == 1],label='hsiao, phase = +1 day')
		#plt.plot(intwave,m0[7,:],label='interp')
		m0test = bisplev(np.array([0]),intwave,(bspl[0],bspl[1],bspl[2],3,3))
		plt.xlim([2000,9200])
		plt.legend()
		plt.plot(intwave,m0test,label='interp')

		#bspl[2][bspl[2] < 0] = 0
		bspltmp = bspl[2].reshape([len(splinephase)-4,len(splinewave)-4])
		#bspltmp[bspltmp
		#m0test = bisplev(np.array([0]),intwave,(bspl[0],bspl[1],bspltmp.flatten(),3,3))
		#plt.plot(intwave,m0test,label='interp')
		import pdb; pdb.set_trace()
	
	return intphase,intwave,m0,m1,bspl[0],bspl[1],bspl[2],bsplm1[2]

def get_hsiao(hsiaofile='initfiles/Hsiao07.dat',
			  Bfilt='initfiles/Bessell90_B.dat',
			  phaserange=[-20,50],waverange=[2000,9200],phaseinterpres=1.0,
			  waveinterpres=2.0,phasesplineres=3.2,wavesplineres=72,
			  days_interp=5.0):

	hphase,hwave,hflux = np.loadtxt(hsiaofile,unpack=True)
	#hflux /= 4*np.pi*(10*3.086e18)**2.
	hflux = hflux.reshape([len(np.unique(hphase)),len(np.unique(hwave))])
	
	phaseinterp = np.linspace(phaserange[0]-days_interp,phaserange[1]+days_interp,
							  (phaserange[1]-phaserange[0]+2*days_interp)/phaseinterpres,False)
	waveinterp = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/waveinterpres,False)

	int1dphase = interp1d(np.unique(hphase),hflux,axis=0)
	hflux_phaseinterp = int1dphase(phaseinterp)
	int1dwave = interp1d(np.unique(hwave),hflux_phaseinterp,axis=1)
	hflux_phasewaveinterp = int1dwave(waveinterp)
	
	return hflux_phasewaveinterp

def synphotB(sourcewave,sourceflux,zpoff,redshift=0,
			 Bfilt='initfiles/Bessell90_B.dat'):
	obswave = sourcewave*(1+redshift)

	filtwave,filttrans = np.genfromtxt(Bfilt,unpack=True)

	g = (obswave >= filtwave[0]) & (obswave <= filtwave[-1])  # overlap range

	pbspl = np.interp(obswave[g],filtwave,filttrans)
	pbspl *= obswave[g]

	res = np.trapz(pbspl*sourceflux[g],obswave[g])/np.trapz(pbspl,obswave[g])
	return(zpoff-2.5*np.log10(res))
