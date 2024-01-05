#!/usr/bin/env python
import numpy as np
from scipy.interpolate import bisplrep,bisplev,RegularGridInterpolator
from scipy.interpolate import interp1d
from sncosmo.constants import HC_ERG_AA
from saltshaker.initfiles import init_rootdir
from scipy.optimize import least_squares
from scipy.special import factorial

import logging
log=logging.getLogger(__name__)

_SCALE_FACTOR = 1e-12

def init_salt2(m0file=None,m1file=None,M0triplet=None,M1triplet=None,
			   Bfilt='initfiles/Bessell90_B.dat',
			   flatnu='initfiles/flatnu.dat',
			   hsiaofile=f'{init_rootdir}/hsiao07.dat',
			   phaserange=[-20,50],waverange=[2000,9200],phaseinterpres=1.0,
			   waveinterpres=2.0,phasesplineres=3.2,wavesplineres=72,
			   days_interp=5,debug=False,normalize=True,order=3,use_snpca_knots=True):

	if m0file:
		phase,wave,m0flux = np.loadtxt(m0file,unpack=True)
	else: phase,wave,m0flux = M0triplet
	iGood = np.where((phase >= phaserange[0]-phasesplineres*0) & (phase <= phaserange[1]+phasesplineres*0) &
					 (wave >= waverange[0]-wavesplineres*0) & (wave <= waverange[1]+wavesplineres*0))[0]
	phase,wave,m0flux = phase[iGood],wave[iGood],m0flux[iGood]	
	
	splinephase = np.linspace(phaserange[0],phaserange[1],int((phaserange[1]-phaserange[0])/phasesplineres)+1,True)
	splinewave	= np.linspace(waverange[0],waverange[1],int((waverange[1]-waverange[0])/wavesplineres)+1,True)

	if use_snpca_knots:
		splinephase = np.array([-20.0,-20.0,-20.0,-20.0,-16.72,-12.90,-8.20,-3.85,-0.08,3.65,7.91,12.54,16.89,20.87,25.22,30.94,38.59,50.0,50.0,50.0,50.0])
		splinewave = np.array([2000., 2000., 2000., 2000., 2067., 2133., 2198., 2262., 2325., 2387., 2449., 2510.,
							   2570., 2630., 2689., 2748., 2806., 2863., 2920., 2977., 3033.,
							   3089., 3145., 3200., 3255., 3310., 3364., 3418., 3472., 3526.,
							   3580., 3633., 3687., 3740., 3793., 3846., 3899., 3952., 4005.,
							   4058., 4111., 4164., 4217., 4270., 4323., 4377., 4430., 4484.,
							   4538., 4592., 4646., 4700., 4755., 4810., 4865., 4921., 4977.,
							   5033., 5090., 5147., 5205., 5263., 5322., 5381., 5441., 5501.,
							   5563., 5624., 5687., 5750., 5814., 5879., 5946.,
							   6012., 6081., 6150., 6220., 6292., 6365., 6439., 6515.,
							   6593., 6672., 6754., 6837., 6922., 7010., 7101., 7194., 7290.,
							   7389., 7492., 7598., 7709., 7825., 7946., 8072., 8205., 8345.,
							   8494., 8652., 8821., 9003.,9200.,9200.,9200.,9200.])
	if m1file is not None:
			m1phase,m1wave,m1flux = np.loadtxt(m1file,unpack=True) #[2][iGood]
			iGood = np.where((m1phase >= phaserange[0]-phasesplineres*0) & (m1phase <= phaserange[1]+phasesplineres*0) &
							 (m1wave >= waverange[0]-wavesplineres*0) & (m1wave <= waverange[1]+wavesplineres*0))[0]
			m1phase,m1wave,m1flux = m1phase[iGood],m1wave[iGood],m1flux[iGood]
	else: m1flux = M1triplet[2]

	# extend the wavelength range using the hsiao model
	# and a bunch of zeros for M1
	if waverange[1] > wave.max(): #9200:
		if len(m1phase) != len(phase): raise RuntimeError('M0 and M1 phases must match')
		if len(m1wave) != len(wave): raise RuntimeError('M0 and M1 wavelengths must match')
		delwave = wave[1]-wave[0]
		delphase = np.unique(phase)[1]-np.unique(phase)[0]
		new_wave_grid = np.arange(waverange[0],waverange[1],delwave)
		new_phase_grid = np.arange(phaserange[0],phaserange[1],delphase)
		
		hphase,hwave,hflux = np.loadtxt(hsiaofile,unpack=True)

		# scale hsiao to match end of SALT spectrum
		m0_out,m1_out = np.zeros([len(new_phase_grid),len(new_wave_grid)]),\
						np.zeros([len(new_phase_grid),len(new_wave_grid)])
		phase_new, wave_new = np.array([]),np.array([])
		for i,p in enumerate(new_phase_grid):
			#hscale = np.median(m0flux[(phase == p) & (wave > 9100)])/\
			#		 np.median(hflux[(hphase == p) & (hwave > 9100) & (hwave < 9200)])
			#m0_p = np.zeros(len(new_wave_grid))
			#m1_p = np.zeros(len(new_wave_grid))
			#idx = np.where(new_wave_grid==9200)[0][0]
			hscale = np.median(m0flux[(phase == p) & (wave > wave.max()-100)])/\
					 np.median(hflux[(hphase == p) & (hwave > wave.max()-100) & (hwave < wave.max())])
			m0_p = np.zeros(len(new_wave_grid))
			m1_p = np.zeros(len(new_wave_grid))
			idx = np.where(new_wave_grid==wave.max())[0][0]
			
			m0_p[:idx+1] = m0flux[phase == p]; m1_p[:idx+1] = m1flux[phase == p]
			m0_p[idx:] = np.interp(new_wave_grid,hwave[hphase == p],hflux[hphase == p])[idx:]*hscale; m1_p[idx:] = 0.0
			m0_out[i,:] = m0_p
			m1_out[i,:] = m1_p
			phase_new = np.append(phase_new,[p]*len(new_wave_grid))
			wave_new = np.append(wave_new,new_wave_grid)

		bspl = bisplrep(phase_new,wave_new,m0_out.flatten(),kx=order,ky=order, tx=splinephase,ty=splinewave,task=-1)
		bsplm1 = bisplrep(phase_new,wave_new,
						  m1_out.flatten(),kx=order,ky=order,
						  tx=splinephase,ty=splinewave,task=-1)
	
	else:
		bspl = bisplrep(phase,wave,m0flux,kx=order,ky=order, tx=splinephase,ty=splinewave,task=-1,nxest=len(splinephase),nyest=len(splinewave))
		bsplm1 = bisplrep(m1phase,m1wave,
						  m1flux,kx=order,ky=order,
						  tx=splinephase,ty=splinewave,task=-1,nxest=len(splinephase),nyest=len(splinewave))

	intphase = np.linspace(phaserange[0],phaserange[1]+phaseinterpres,
						   int((phaserange[1]-phaserange[0])/phaseinterpres)+1,False)
	intwave = np.linspace(waverange[0],waverange[1]+waveinterpres,
						  int((waverange[1]-waverange[0])/waveinterpres)+1,False)

	m0 = bisplev(intphase,intwave,bspl)
	m1 = bisplev(intphase,intwave,bsplm1)
	if debug:
		import pylab as plt
		plt.ion()
		plt.plot(wave[phase == 0],flux[phase == 0],label='hsiao, phase = 0 days')
		m0test = bisplev(np.array([0]),intwave,(bspl[0],bspl[1],bspl[2],3,3))
		plt.xlim([2000,9200])
		plt.legend()
		plt.plot(intwave,m0test,label='interp')
		bspltmp = bspl[2].reshape([len(splinephase)-4,len(splinewave)-4])

	return intphase,intwave,m0,m1,bspl[0],bspl[1],bspl[2],bsplm1[2]


def init_hsiao(hsiaofile='initfiles/hsiao07.dat',
			   Bfilt='initfiles/Bessell90_B.dat',
			   flatnu='initfiles/flatnu.dat',
			   phaserange=[-20,50],waverange=[2000,9200],phaseinterpres=1.0,
			   waveinterpres=2.0,phasesplineres=3.2,wavesplineres=72,
			   days_interp=5,debug=False,normalize=True,order=3,use_snpca_knots=False):

	phase,wave,flux = np.loadtxt(hsiaofile,unpack=True)
	
	refWave,refFlux=np.loadtxt(flatnu,unpack=True)
	# was *6
	iGood = np.where((phase >= phaserange[0]-phasesplineres*0) & (phase <= phaserange[1]+phasesplineres*0) &
					 (wave >= waverange[0]-wavesplineres*0) & (wave <= waverange[1]+wavesplineres*0))[0]
	phase,wave,flux = phase[iGood],wave[iGood],flux[iGood]
	
	if normalize:
		m0flux = flux*10**(-0.4*(-19.49+(synphotB(refWave,refFlux,0,0,Bfilt)-synphotB(wave[phase==0],flux[phase==0],0,0,Bfilt))))#*_SCALE_FACTOR
	else:
		m0flux = flux[:]

	#m1phase = phase*1.1
	# was *3 (phase), *5 (wave)
	splinephase = np.linspace(phaserange[0],phaserange[1],int((phaserange[1]-phaserange[0])/phasesplineres)+1,True)
	splinewave	= np.linspace(waverange[0],waverange[1],int((waverange[1]-waverange[0])/wavesplineres)+1,True)
	#splinephase = [-8.2, -3.85, -0.05, 3.65, 7.9, 12.55, 16.9, 20.9, 25.25, 30.95, 38.55, 50, 50.05, 50.05]
	#splinewave = [2198,2262.08,2325.44,2387.36,2449.28,2510.48,2570.24,2630,2689.04,2748.08,2805.68,2863.28,2920.88,2977.76,3033.92,3089.36,3145.52,3200.24,3255.68,3310.4,3364.4,3418.4,3472.4,3526.4,3580.4,3633.68,3686.96,3740.24,3793.52,3846.08,3899.36,3952.64,4005.2,4058.48,4111.04,4164.32,4217.6,4270.16,4323.44,4376.72,4430.72,4484,4538,4592,4646,4700.72,4755.44,4810.16,4865.6,4921.76,4977.2,5034.08,5090.24,5147.84,5205.44,5263.76,5322.08,5381.84,5441.6,5502.08,5562.56,5624.48,5687.12,5750.48,5814.56,5880.08,5945.6,6012.56,6080.96,6150.08,6220.64,6291.92,6365.36,6439.52,6515.84,6593.6,6672.8,6754.16,6836.96,6922.64,7010.48,7101.2,7194.08,7289.84,7389.2,7492.16,7598.72,7709.6,7825.52,7945.76,8072.48,8205.68,8345.36,8494.4,8652.08,8821.28,9003.44,9200,9200.72,9200.72]
	if use_snpca_knots:
		splinephase = np.array([-20.0,-20.0,-20.0,-20.0,-16.72,-12.90,-8.20,-3.85,-0.08,3.65,7.91,12.54,16.89,20.87,25.22,30.94,38.59,50.0,50.0,50.0,50.0])
		splinewave = np.array([2000., 2000., 2000., 2000., 2067., 2133., 2198., 2262., 2325., 2387., 2449., 2510.,
							   2570., 2630., 2689., 2748., 2806., 2863., 2920., 2977., 3033.,
							   3089., 3145., 3200., 3255., 3310., 3364., 3418., 3472., 3526.,
							   3580., 3633., 3687., 3740., 3793., 3846., 3899., 3952., 4005.,
							   4058., 4111., 4164., 4217., 4270., 4323., 4377., 4430., 4484.,
							   4538., 4592., 4646., 4700., 4755., 4810., 4865., 4921., 4977.,
							   5033., 5090., 5147., 5205., 5263., 5322., 5381., 5441., 5501.,
							   5563., 5624., 5687., 5750., 5814., 5879., 5946.,
							   6012., 6081., 6150., 6220., 6292., 6365., 6439., 6515.,
							   6593., 6672., 6754., 6837., 6922., 7010., 7101., 7194., 7290.,
							   7389., 7492., 7598., 7709., 7825., 7946., 8072., 8205., 8345.,
							   8494., 8652., 8821., 9003.,9200.,9200.,9200.,9200.])

	bspl = bisplrep(phase,wave,m0flux,kx=order,ky=order, tx=splinephase,ty=splinewave,task=-1,nxest=len(splinephase),nyest=len(splinewave))
	
	intphase = np.linspace(phaserange[0],phaserange[1]+phaseinterpres,
						   int((phaserange[1]-phaserange[0])/phaseinterpres)+1,False)
	intwave = np.linspace(waverange[0],waverange[1]+waveinterpres,
						  int((waverange[1]-waverange[0])/waveinterpres)+1,False)


	m0 = bisplev(intphase,intwave,bspl)
	#Define m1 guess by a simple stretch of the Hsiao template (chose these numbers off of the effect of x1 on B-band stretch)
	stretch=(1.07+0.069-0.015+0.00067)/1.07
	stretchedPhase=np.clip(intphase*stretch,phaserange[0],phaserange[1])
	m1fluxguess = (m0-bisplev(stretchedPhase,intwave,bspl)).flatten()
	intwavetmp,intphasetmp = np.meshgrid(intwave,intphase)
	bsplm1 = bisplrep(intphasetmp.flatten(),intwavetmp.flatten(),m1fluxguess,kx=order,ky=order,tx=splinephase,ty=splinewave,task=-1,nxest=len(splinephase),nyest=len(splinewave))
	m1 = bisplev(intphase,intwave,bsplm1)
	if debug:
		import pylab as plt
		plt.ion()
		plt.plot(wave[phase == 0],flux[phase == 0],label='hsiao, phase = 0 days')
		m0test = bisplev(np.array([0]),intwave,(bspl[0],bspl[1],bspl[2],3,3))
		plt.xlim([2000,9200])
		plt.legend()
		plt.plot(intwave,m0test,label='interp')
		bspltmp = bspl[2].reshape([len(splinephase)-4,len(splinewave)-4])

	return intphase,intwave,m0,m1,bspl[0],bspl[1],bspl[2],bsplm1[2]

def init_kaepora(x10file='initfiles/Kaepora_dm15_1.1.txt',
				 x11file='initfiles/Kaepora_dm15_0.94.txt',
				 Bfilt='initfiles/Bessell90_B.dat',
				 flatnu='initfiles/flatnu.dat',
				 phaserange=[-20,50],waverange=[2000,9200],phaseinterpres=1.0,
				 waveinterpres=2.0,phasesplineres=3.2,wavesplineres=72,
				 days_interp=5,debug=False,normalize=True):

	phase,wave,flux = np.loadtxt(x10file,unpack=True)
	x11phase,x11wave,x11flux = np.loadtxt(x11file,unpack=True)
	refWave,refFlux=np.loadtxt(flatnu,unpack=True)

	if normalize:
		m0flux = flux*10**(-0.4*(-19.49+(synphotB(refWave,refFlux,0,0,Bfilt)-synphotB(wave[phase==0],flux[phase==0],0,0,Bfilt))))#*_SCALE_FACTOR
	else:
		m0flux = flux[:]
		
	#m1phase = phase*1.1
	splinephase = np.linspace(phaserange[0],phaserange[1],
							  (phaserange[1]-phaserange[0])/phasesplineres,False)
	splinewave = np.linspace(waverange[0],waverange[1],
							 (waverange[1]-waverange[0])/wavesplineres,False)
	bspl = bisplrep(phase,wave,m0flux,kx=3,ky=3,
					tx=splinephase,ty=splinewave,task=-1)

	intphase = np.linspace(phaserange[0],phaserange[1],
						   (phaserange[1]-phaserange[0])/phaseinterpres,False)
	intwave = np.linspace(waverange[0],waverange[1],
						  (waverange[1]-waverange[0])/waveinterpres,False)


	m0 = bisplev(intphase,intwave,bspl)
	m1fluxguess = (x11flux-flux)*10**(-0.4*(-19.49+(synphotB(refWave,refFlux,0,0,Bfilt)-synphotB(wave[phase==0],flux[phase==0],0,0,Bfilt)))) #10**(-0.4*(-8.93+(synphotB(refWave,refFlux,0,0,Bfilt)-synphotB(wave[phase==0],x11flux[phase==0] - flux[phase==0],0,0,Bfilt))))
	bsplm1 = bisplrep(phase,wave,
					  m1fluxguess,kx=3,ky=3,
					  tx=splinephase,ty=splinewave,task=-1)
	m1 = bisplev(intphase,intwave,bsplm1)
	if debug:
		import pylab as plt
		plt.ion()
		plt.plot(wave[phase == 0],flux[phase == 0],label='hsiao, phase = 0 days')
		m0test = bisplev(np.array([0]),intwave,(bspl[0],bspl[1],bspl[2],3,3))
		plt.xlim([2000,9200])
		plt.legend()
		plt.plot(intwave,m0test,label='interp')
		bspltmp = bspl[2].reshape([len(splinephase)-4,len(splinewave)-4])

	return intphase,intwave,m0,m1,bspl[0],bspl[1],bspl[2],bsplm1[2]

def init_errs(m0varfile=None,m0m1file=None,m1varfile=None,scalefile=None,clscatfile=None,
			  Bfilt='initfiles/Bessell90_B.dat',
			  phaserange=[-20,50],waverange=[2000,9200],phaseinterpres=1.0,
			  waveinterpres=10.0,phasesplineres=6,wavesplineres=1200,n_colorscatpars=4,
			  order=3,normalize=True):
	splinephase = np.linspace(phaserange[0],phaserange[1],int((phaserange[1]-phaserange[0])/phasesplineres)+1,True)
	splinewave	= np.linspace(waverange[0],waverange[1],int((waverange[1]-waverange[0])/wavesplineres)+1,True)

	def loadfilewithdefault(filename,fillval=0):
		if filename is None:
			phase,wave=np.meshgrid(np.linspace(phaserange[0],phaserange[1],int((phaserange[1]-phaserange[0])/phaseinterpres)+1,True), 
								   np.linspace(waverange[0],waverange[1],int((waverange[1]-waverange[0])/waveinterpres)+1,True))
			return phase.flatten(),wave.flatten(),fillval*np.ones(phase.size)
		else:
			return np.loadtxt(filename,unpack=True)
	def initbsplwithzeroth(phase,wave,flux,kx=order,ky=order, tx=splinephase,ty=splinewave):
		if order==0:
			binphasecenter=((splinephase)[1:]+(splinephase)[:-1])/2
			binwavecenter =((splinewave)[1:]+(splinewave)[:-1])/2
			fluxmeans= np.empty((binphasecenter.size,binwavecenter.size))
			for i,phaseup,phaselow in	zip(range(splinephase.size-1),(splinephase)[1:],(splinephase)[:-1]):
				for j,waveup,wavelow in zip(range(splinewave.size-1),(splinewave)[1:], (splinewave)[:-1]):
					phasebin=(phase<phaseup)&(phase>=phaselow)
					wavebin= (wave <waveup)&(wave>=wavelow)
					fluxmeans[i][j]=np.mean(flux[wavebin&phasebin])
			return splinephase,splinewave,fluxmeans.flatten(),0,0
		else:
			return bisplrep(phase,wave,m0var,kx=order,ky=order, tx=splinephase,ty=splinewave,task=-1)
	scalephase,scalewave,scale=loadfilewithdefault(scalefile,1)
	
	#Subtract out statistical error from SALT2
	if m0varfile is not None: scale=np.sqrt(scale**2-1)
	scalephase,scalewave=np.unique(scalephase),np.unique(scalewave)
	scaleinterp=RegularGridInterpolator((scalephase,scalewave),scale.reshape(scalephase.size,scalewave.size),'nearest')
	clipinterp=lambda x,y: scaleinterp((np.clip(x,scalephase.min(),scalephase.max()),np.clip(y,scalewave.min(),scalewave.max())))
	phase,wave,m0var = loadfilewithdefault(m0varfile,fillval=1e-6)
	iGood = np.where((phase >= phaserange[0]-phasesplineres*0) & (phase <= phaserange[1]+phasesplineres*0) &
					 (wave >= waverange[0]-wavesplineres*0) & (wave <= waverange[1]+wavesplineres*0))[0]
	phase,wave,m0var = phase[iGood],wave[iGood],np.sqrt(m0var[iGood])
	m0var*=clipinterp(phase,wave)
	m0varbspl = initbsplwithzeroth(phase,wave,m0var,kx=order,ky=order, tx=splinephase,ty=splinewave)

	phase,wave,m1var = loadfilewithdefault(m1varfile,fillval=1e-6)
	iGood = np.where((phase >= phaserange[0]-phasesplineres*0) & (phase <= phaserange[1]+phasesplineres*0) &
					 (wave >= waverange[0]-wavesplineres*0) & (wave <= waverange[1]+wavesplineres*0))[0]
	phase,wave,m1var = phase[iGood],wave[iGood],np.sqrt(m1var[iGood])
	m1var*=clipinterp(phase,wave)
	m1varbspl = initbsplwithzeroth(phase,wave,m1var,kx=order,ky=order, tx=splinephase,ty=splinewave)
	
	phase,wave,m0m1covar = loadfilewithdefault(m0m1file)
	iGood = np.where((phase >= phaserange[0]-phasesplineres*0) & (phase <= phaserange[1]+phasesplineres*0) &
					 (wave >= waverange[0]-wavesplineres*0) & (wave <= waverange[1]+wavesplineres*0))[0]
	phase,wave,m0m1covar = phase[iGood],wave[iGood],m0m1covar[iGood]
	corr=m0m1covar/(m0var*m1var)*clipinterp(phase,wave)**2
	corr[np.isnan(corr)]=0
	m0m1corrbspl = initbsplwithzeroth(phase,wave,corr,kx=order,ky=order, tx=splinephase,ty=splinewave)
	if n_colorscatpars>0:
		if clscatfile is None:
			clscatpars=np.zeros(n_colorscatpars)
			clscatpars[-1]=-np.inf
		else:
			wave,clscat=np.loadtxt(clscatfile,unpack=True)
			wave,clscat=wave[wave<9200],clscat[wave<9200]
			pow=n_colorscatpars-1-np.arange(n_colorscatpars)
			clscatpars=np.polyfit((wave-5500)/1000,np.log(clscat),n_colorscatpars-1)*factorial(pow)#guess[resids.iclscat]

	return m0varbspl[0],m0varbspl[1],m0varbspl[2],m1varbspl[2],m0m1corrbspl[2],clscatpars
	
def init_errs_percent(
		phase,wave,phaseknotloc,waveknotloc,m0knots,m1knots,mhostknots=None,
		m0file=None,m0m1file=None,m1file=None,scalefile=None,clscatfile=None,
		Bfilt='initfiles/Bessell90_B.dat',
		phaserange=[-20,50],waverange=[2000,9200],phaseinterpres=1.0,
		waveinterpres=10.0,phasesplineres=6,wavesplineres=1200,n_colorscatpars=4,
		order=3,normalize=True):
	splinephase = np.linspace(phaserange[0],phaserange[1],int((phaserange[1]-phaserange[0])/phasesplineres)+1,True)
	splinewave	= np.linspace(waverange[0],waverange[1],int((waverange[1]-waverange[0])/wavesplineres)+1,True)

	def loadfromspline():
		m0 = bisplev(phase,
					 wave,
					 (phaseknotloc,waveknotloc,m0knots,3,3))
		m1 = bisplev(phase,
					 wave,
					 (phaseknotloc,waveknotloc,m1knots,3,3))
		return (m0*0.0005)**2.,(m1*0.0005)**2.
		
	def loadfilewithdefault(filename,fillval=0):
		if filename is None:
			phase,wave=np.meshgrid(np.linspace(phaserange[0],phaserange[1],int((phaserange[1]-phaserange[0])/phaseinterpres)+1,True), 
								   np.linspace(waverange[0],waverange[1],int((waverange[1]-waverange[0])/waveinterpres)+1,True))
			return phase.flatten(),wave.flatten(),fillval*np.ones(phase.size)
		else:
			return np.loadtxt(filename,unpack=True)
	
	def initbsplwithzeroth(phase,wave,flux,kx=order,ky=order, tx=splinephase,ty=splinewave):
		if order==0:
			binphasecenter=((splinephase)[1:]+(splinephase)[:-1])/2
			binwavecenter =((splinewave)[1:]+(splinewave)[:-1])/2
			fluxmeans= np.empty((binphasecenter.size,binwavecenter.size))
			for i,phaseup,phaselow in	zip(range(splinephase.size-1),(splinephase)[1:],(splinephase)[:-1]):
				for j,waveup,wavelow in zip(range(splinewave.size-1),(splinewave)[1:], (splinewave)[:-1]):
					phasebin=(phase<phaseup)&(phase>=phaselow)
					wavebin= (wave <waveup)&(wave>=wavelow)
					fluxmeans[i][j]=np.mean(flux[wavebin&phasebin])
			return splinephase,splinewave,fluxmeans.flatten(),0,0
		else:
			return bisplrep(phase,wave,m0var,kx=order,ky=order, tx=splinephase,ty=splinewave,task=-1)
	scalephase,scalewave,scale=loadfilewithdefault(scalefile,1)
	
	#Subtract out statistical error from SALT2
	if m0file is not None: scale=np.sqrt(scale**2-1)
	scalephase,scalewave=np.unique(scalephase),np.unique(scalewave)
	scaleinterp=RegularGridInterpolator((scalephase,scalewave),scale.reshape(scalephase.size,scalewave.size),'nearest')
	clipinterp=lambda x,y: scaleinterp((np.clip(x,scalephase.min(),scalephase.max()),np.clip(y,scalewave.min(),scalewave.max())))
	m0var,m1var = loadfromspline()
	phase,wave = np.meshgrid(phase,wave)
	phase,wave,m0var,m1var = phase.flatten(),wave.flatten(),m0var.flatten(),m1var.flatten()
	iGood = np.where((phase >= phaserange[0]-phasesplineres*0) & (phase <= phaserange[1]+phasesplineres*0) &
					 (wave >= waverange[0]-wavesplineres*0) & (wave <= waverange[1]+wavesplineres*0))[0]
	newphase,newwave,m0var = phase[iGood],wave[iGood],np.sqrt(m0var[iGood])
	m0var*=clipinterp(newphase,newwave)
	m0varbspl = initbsplwithzeroth(newphase,newwave,m0var,kx=order,ky=order, tx=splinephase,ty=splinewave)

	iGood = np.where((phase >= phaserange[0]-phasesplineres*0) & (phase <= phaserange[1]+phasesplineres*0) &
					 (wave >= waverange[0]-wavesplineres*0) & (wave <= waverange[1]+wavesplineres*0))[0]
	newphase,newwave,m1var = phase[iGood],wave[iGood],np.sqrt(m1var[iGood])
	m1var*=clipinterp(newphase,newwave)
	m1varbspl = initbsplwithzeroth(newphase,newwave,m1var,kx=order,ky=order, tx=splinephase,ty=splinewave)
	
	phase,wave,m0m1covar = loadfilewithdefault(m0m1file)
	iGood = np.where((phase >= phaserange[0]-phasesplineres*0) & (phase <= phaserange[1]+phasesplineres*0) &
					 (wave >= waverange[0]-wavesplineres*0) & (wave <= waverange[1]+wavesplineres*0))[0]
	phase,wave,m0m1covar = phase[iGood],wave[iGood],m0m1covar[iGood]
	corr=m0m1covar/(m0var*m1var)*clipinterp(phase,wave)**2
	corr[np.isnan(corr)]=0
	m0m1corrbspl = initbsplwithzeroth(phase,wave,corr,kx=order,ky=order, tx=splinephase,ty=splinewave)
	if n_colorscatpars>0:
		if clscatfile is None:
			clscatpars=np.zeros(n_colorscatpars)
			clscatpars[-1]=-np.inf
		else:
			wave,clscat=np.loadtxt(clscatfile,unpack=True)
			wave,clscat=wave[wave<9200],clscat[wave<9200]
			pow=n_colorscatpars-1-np.arange(n_colorscatpars)
			clscatpars=np.polyfit((wave-5500)/1000,np.log(clscat),n_colorscatpars-1)*factorial(pow)#guess[resids.iclscat]

	if mhostknots is None:
		return m0varbspl[0],m0varbspl[1],m0varbspl[2],m1varbspl[2],m0m1corrbspl[2],clscatpars
	else:
		return m0varbspl[0],m0varbspl[1],m0varbspl[2],m1varbspl[2],np.zeros(len(m0varbspl[2])),m0m1corrbspl[2],clscatpars

def init_custom(M0,M1,
				Bfilt='initfiles/Bessell90_B.dat',
				phaserange=[-20,50],waverange=[2000,9200],phaseinterpres=1.0,
				waveinterpres=10.0,phasesplineres=6,wavesplineres=1200,n_colorscatpars=4,
				order=3,normalize=True):
	splinephase = np.linspace(phaserange[0],phaserange[1],int((phaserange[1]-phaserange[0])/phasesplineres)+1,True)
	splinewave	= np.linspace(waverange[0],waverange[1],int((waverange[1]-waverange[0])/wavesplineres)+1,True)

	def load(M):
		phase,wave=np.meshgrid(np.linspace(phaserange[0],phaserange[1],int((phaserange[1]-phaserange[0])/phaseinterpres)+1,True), 
							   np.linspace(waverange[0],waverange[1],int((waverange[1]-waverange[0])/waveinterpres)+1,True))
		return phase.flatten(),wave.flatten(),M.flatten()
		
	def initbsplwithzeroth(phase,wave,flux,kx=order,ky=order, tx=splinephase,ty=splinewave):
		if order==0:
			binphasecenter=((splinephase)[1:]+(splinephase)[:-1])/2
			binwavecenter =((splinewave)[1:]+(splinewave)[:-1])/2
			fluxmeans= np.empty((binphasecenter.size,binwavecenter.size))
			for i,phaseup,phaselow in zip(range(splinephase.size-1),(splinephase)[1:],(splinephase)[:-1]):
				for j,waveup,wavelow in zip(range(splinewave.size-1),(splinewave)[1:], (splinewave)[:-1]):
					phasebin=(phase<phaseup)&(phase>=phaselow)
					wavebin= (wave <waveup)&(wave>=wavelow)
					fluxmeans[i][j]=np.mean(flux[wavebin&phasebin])
			return splinephase,splinewave,fluxmeans.flatten(),0,0
		else:
			return bisplrep(phase,wave,flux,kx=order,ky=order, tx=splinephase,ty=splinewave,task=-1)
	
	phase,wave,m0 = load(M0)
	iGood = np.where((phase >= phaserange[0]-phasesplineres*0) & (phase <= phaserange[1]+phasesplineres*0) &
					 (wave >= waverange[0]-wavesplineres*0) & (wave <= waverange[1]+wavesplineres*0))[0]
	phase,wave,m0 = phase[iGood],wave[iGood],m0[iGood]
	m0bspl = initbsplwithzeroth(phase,wave,m0,kx=order,ky=order, tx=splinephase,ty=splinewave)

	phase,wave,m1 = load(M1)
	iGood = np.where((phase >= phaserange[0]-phasesplineres*0) & (phase <= phaserange[1]+phasesplineres*0) &
					 (wave >= waverange[0]-wavesplineres*0) & (wave <= waverange[1]+wavesplineres*0))[0]
	phase,wave,m1 = phase[iGood],wave[iGood],m1[iGood]
	m1bspl = initbsplwithzeroth(phase,wave,m1,kx=order,ky=order, tx=splinephase,ty=splinewave)
	
	return m0bspl[0],m0bspl[1],m0bspl[2],m1bspl[2]


def init_salt2_cdisp(cdfile=None,order=4):

	def loadfilewithdefault(filename,fillval=0):
		if filename is None:
			wave=np.linspace(waverange[0],waverange[1],int((waverange[1]-waverange[0])/waveinterpres)+1,True)
			return fillval*np.ones(wave.size)
		else:
			return np.loadtxt(filename,unpack=True)
	
	wave,scale=loadfilewithdefault(cdfile,1)
	return np.polyfit(wave/1000,np.log(scale),order-1)

def get_hsiao(hsiaofile='initfiles/hsiao07.dat',
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

	int1dphase = interp1d(np.unique(hphase),hflux,axis=0,
						  fill_value='extrapolate')
	hflux_phaseinterp = int1dphase(phaseinterp)
	int1dwave = interp1d(np.unique(hwave),hflux_phaseinterp,axis=1,
						 fill_value='extrapolate')
	hflux_phasewaveinterp = int1dwave(waveinterp)
	
	return hflux_phasewaveinterp

def synphotB(sourcewave,sourceflux,zpoff,redshift=0,
			 Bfilt='initfiles/Bessell90_B.dat'):
	obswave = sourcewave*(1+redshift)

	filtwave,filttrans = np.genfromtxt(Bfilt,unpack=True)

	g = (obswave >= filtwave[0]) & (obswave <= filtwave[-1])  # overlap range

	pbspl = np.interp(obswave[g],filtwave,filttrans)
	pbspl *= obswave[g]

	res = np.trapz(pbspl*sourceflux[g]/HC_ERG_AA,obswave[g])/np.trapz(pbspl,obswave[g])
	return(zpoff-2.5*np.log10(res))

def synphotBflux(sourcewave,sourceflux,zpoff,redshift=0,
			 Bfilt='initfiles/Bessell90_B.dat'):
	obswave = sourcewave*(1+redshift)

	filtwave,filttrans = np.genfromtxt(Bfilt,unpack=True)

	g = (obswave >= filtwave[0]) & (obswave <= filtwave[-1])  # overlap range

	pbspl = np.interp(obswave[g],filtwave,filttrans)
	pbspl *= obswave[g]

	res = np.trapz(pbspl*sourceflux[g]/HC_ERG_AA,obswave[g])/np.trapz(pbspl,obswave[g])
	return res
