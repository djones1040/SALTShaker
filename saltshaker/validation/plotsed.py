#!/usr/bin/env python
import numpy as np
import pylab as plt
import sncosmo
from saltshaker.util import snana
from astropy.table import Table

import logging
log=logging.getLogger(__name__)

def plot_io_sed(inputfile='../initfiles/hsiao07.dat',
				outputfile='../../examples/output/salt3_template_0.dat',phase=None):

	iphase,iwave,iflux = np.loadtxt(inputfile,unpack=True)
	ophase,owave,oflux = np.loadtxt(outputfile,unpack=True)

	
	#Normalize fluxes at peak s.t. total emitted light is equal
	owave=np.unique(owave)
	iwave=np.unique(iwave)
	normalization=np.trapz(get_interpolated_flux(0,iflux,iphase), iwave )/np.trapz(get_interpolated_flux(0,oflux,ophase), owave )
	oflux*=normalization
	
		
	for p in np.unique([-8,0,8,20] if phase is None else phase):
		if p < -14: continue
		plt.close()

		plt.plot(iwave,get_interpolated_flux(p,iflux,iphase),label='Hsiao model')
		
		plt.plot(owave,get_interpolated_flux(p,oflux,ophase),label='output model')
		plt.xlabel('Wavelength ($\AA$)')
		plt.ylabel('Flux')
		plt.title('phase = %.1f days'%p)
		plt.xlim([2000,9200])
		plt.savefig('io_sed_{}.png'.format(p),dpi=288)
		#plt.show()
		#import pdb; pdb.set_trace()
		
	return

def get_interpolated_flux(phase,fluxes,phases):
	phaseVals=np.sort(np.unique(phases))
	leftPhase=np.searchsorted(phaseVals,phase,'left')
	rightPhase=phaseVals[leftPhase+1]
	leftPhase=phaseVals[leftPhase]
	return (fluxes[phases==rightPhase] - fluxes[phases==leftPhase])/(rightPhase-leftPhase) *(phase-leftPhase)+fluxes[phases==leftPhase]
	
def complc(lcfile='../../examples/exampledata/photdata/foundASASSN-16av.txt',zpsys='AB',
		   fitparams = ['t0', 'x0', 'x1', 'c'], hsiaofitparams = ['t0','amplitude']):

	plt.clf()
	ax1 = plt.subplot(131)
	ax2 = plt.subplot(132)
	ax3 = plt.subplot(133)
	
	salt3 = sncosmo.SALT2Source(modeldir='../../examples/output',
								m0file='salt3_template_0.dat',
								m1file='salt3_template_1.dat')
	salt3model = sncosmo.Model(salt3)
	salt2model = sncosmo.Model(source='salt2')
	hsiaomodel = sncosmo.Model(source='hsiao')
	
	sn = snana.SuperNova(lcfile)
	sn.FLT = sn.FLT.astype('U20')
	for i in range(len(sn.FLT)):
		sn.FLT[i] = 'sdss%s'%sn.FLT[i]

	data = Table([sn.MJD,sn.FLT,sn.FLUXCAL,sn.FLUXCALERR,
				  np.array([27.5]*len(sn.MJD)),np.array([zpsys]*len(sn.MJD))],
						   names=['mjd','band','flux','fluxerr','zp','zpsys'],
						   meta={'t0':sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]})
	flux = sn.FLUXCAL
	result_salt3, fitted_salt3_model = sncosmo.fit_lc(
        data, salt3model, fitparams,
	    bounds={'t0':(sn.MJD[flux == np.max(flux)]-10, sn.MJD[flux == np.max(flux)]+10),
                'z':(sn.z-0.01,sn.z+0.01),'x1':(-3,3),'c':(-0.3,0.3)})
	result_salt2, fitted_salt2_model = sncosmo.fit_lc(
        data, salt2model, fitparams,
	    bounds={'t0':(sn.MJD[flux == np.max(flux)]-10, sn.MJD[flux == np.max(flux)]+10),
                'z':(sn.z-0.01,sn.z+0.01),'x1':(-3,3),'c':(-0.3,0.3)})
	result_hsiao, fitted_hsiao_model = sncosmo.fit_lc(
        data, hsiaomodel, hsiaofitparams,
	    bounds={'t0':(sn.MJD[flux == np.max(flux)]-10, sn.MJD[flux == np.max(flux)]+10),
                'z':(sn.z-0.01,sn.z+0.01)})
	
	plotmjd = np.linspace(sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]-20,
						  sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]+50,70)
	for flt,i,ax in zip(['sdssg','sdssr','sdssi','sdssz'],range(4),[ax1,ax2,ax3]):
		if flt == 'sdssz': continue
		salt3flux = fitted_salt3_model.bandflux(flt, plotmjd,
												zp=27.5,zpsys='AB')
		hsiaoflux = fitted_hsiao_model.bandflux(flt, plotmjd,
												zp=27.5,zpsys='AB')
		salt2flux = fitted_salt2_model.bandflux(flt, plotmjd,
												zp=27.5,zpsys='AB')

		ax.plot(plotmjd,hsiaoflux,color='C0',
				label='Hsiao Model, $\chi^2_{red} = %.1f$'%(result_hsiao['chisq']/result_hsiao['ndof']))
		ax.plot(plotmjd,salt2flux,color='k',
				label='SALT2 Model, $\chi^2_{red} = %.1f$'%(result_salt2['chisq']/result_salt2['ndof']))
		ax.plot(plotmjd,salt3flux,color='C1',
				label='SALT3 Model, $\chi^2_{red} = %.1f$'%(result_salt3['chisq']/result_salt3['ndof']))
		ax.errorbar(sn.MJD[sn.FLT == flt],sn.FLUXCAL[sn.FLT == flt],
					yerr=1.086*sn.FLUXCALERR[sn.FLT == flt]/sn.FLUXCAL[sn.FLT == flt],
					fmt='o',label=sn.SNID,color='C2')
		ax.set_title(flt.replace('sdss',''))


		ax.set_ylabel('Mag')
		ax.set_xlabel('MJD')
		mag = -2.5*np.log10(sn.FLUXCAL)+27.5
		ax.set_ylim([np.min(sn.FLUXCAL),
					 np.max(sn.FLUXCAL*10**(0.4*0.25))])
		ax.set_xlim([sn.MJD[flux == np.max(flux)]-20, sn.MJD[flux == np.max(flux)]+50])

	ax1.legend()
	plt.savefig('comp.png')
	
	return

if __name__ == "__main__":
	complc()

