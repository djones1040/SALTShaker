#!/usr/bin/env python
"""initial recalibration of spectra to the photometric
data itself"""

from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid as trapz
import numpy as np
from saltshaker.util.synphot import synphot
from sncosmo.constants import HC_ERG_AA
from scipy.special import factorial
import pylab as plt

def SpecRecal(photdata,specdata,kcordict,survey,specrange_wavescale_specrecal,nrecalpars=0,doplot=False,sn=None):

	# from photdata, find all obs w/i two days of phase
	# of those, choose the closest in each filter
	# if nothing exists, relax to four days
	dwave = np.median(specdata['wavelength'][1:]-specdata['wavelength'][:-1])
	primarywave = kcordict[survey]['primarywave']
	maxwave = np.max(specdata['wavelength'])
	minwave = np.min(specdata['wavelength'])

	iGoodSNR = np.where(specdata['flux']/specdata['fluxerr'] > 3)[0]

	specflux,photflux,photwave,specfluxerr,photfluxerr,photcorr = [],[],[],[],[],[]
	for flt in np.unique(photdata['filt']):
		filtwave = kcordict[survey][flt]['filtwave']
		# preliminaries
		if kcordict[survey][flt]['magsys'] == 'AB': primarykey = 'AB'
		elif kcordict[survey][flt]['magsys'].upper() == 'VEGA': primarykey = 'Vega'
		elif kcordict[survey][flt]['magsys'] == 'BD17': primarykey = 'BD17'
		stdmag = synphot(
			primarywave,kcordict[survey][primarykey],
			filtwave=filtwave,
			filttp=kcordict[survey][flt]['filttrans'],
			zpoff=0) - kcordict[survey][flt]['primarymag']
		fluxfactor = 10**(0.4*(stdmag+27.5))
		wht = np.sum(kcordict[survey][flt]['filttrans'][
			(filtwave > maxwave) |
			(filtwave < minwave)])/np.sum(kcordict[survey][flt]['filttrans'])
		if wht > 0.02: continue
		if len(photdata['filt'][photdata['filt'] == flt]) == 1: continue

		filttrans = kcordict[survey][flt]['filttrans']
		pbspl = np.interp(specdata['wavelength'][iGoodSNR],filtwave,filttrans)

		pbspl *= specdata['wavelength'][iGoodSNR]
		denom = trapz(pbspl,specdata['wavelength'][iGoodSNR])
		pbspl /= denom*HC_ERG_AA
				
		phot1d = interp1d(photdata['mjd'][photdata['filt'] == flt],
						  photdata['fluxcal'][photdata['filt'] == flt],
						  axis=0,kind='linear',#bounds_error=True,
						  assume_sorted=True,fill_value="extrapolate")
		photerr1d = interp1d(photdata['mjd'][photdata['filt'] == flt],
							 photdata['fluxcalerr'][photdata['filt'] == flt],
							 axis=0,kind='linear',#bounds_error=True,
							 assume_sorted=True,fill_value="extrapolate")
		#try:
		flux = phot1d(specdata['mjd'])
		fluxerr = photerr1d(specdata['mjd'])
		#except: continue
		if np.min(np.abs(photdata['mjd'][photdata['filt'] == flt] - specdata['mjd'])) > 4:
			continue
		
		synph = np.sum(specdata['flux'][iGoodSNR]*pbspl)*dwave*fluxfactor
		synpherr = np.sqrt(np.sum((specdata['fluxerr'][iGoodSNR]*pbspl)**2.))*dwave*fluxfactor

		specflux += [synph]
		photflux += [flux]
		specfluxerr += [synpherr]
		photfluxerr += [fluxerr]
		# really need to multiply throughput by spectrum, but this is probably ok for now
		photwave += [kcordict[survey][flt]['lambdaeff']]
		photcorr += [synph/np.interp(kcordict[survey][flt]['lambdaeff'],specdata['wavelength'],specdata['flux'])]

	specflux,photflux,photwave,specfluxerr,photfluxerr,photcorr = \
		np.array(specflux),np.array(photflux),np.array(photwave),\
		np.array(specfluxerr),np.array(photfluxerr),np.array(photcorr)
		

	if doplot:
		plt.clf()
		plt.plot(specdata['wavelength'],specdata['flux'],label='original spec')#/recalexp)
		plt.errorbar(photwave,specflux/photcorr,yerr=specfluxerr/photcorr,fmt='o',label='synth. phot')
		plt.xlabel('Wavelength ($\mathrm{\AA}$)',fontsize=15)
		plt.ylabel('Flux',fontsize=15)

	md = minimize(chifunc,np.array([0.]*(nrecalpars+1)),args=(
		photflux,specflux,photfluxerr,specfluxerr,photwave,specrange_wavescale_specrecal))
	if doplot: # or sn == '06D1ab': #5999409':
		plt.plot(specdata['wavelength'],specdata['flux']/recalfunc(md.x,specdata['wavelength'],specrange_wavescale_specrecal),
				 label='recalibrated spec.')
		plt.errorbar(photwave,photflux/photcorr/md.x[0],yerr=photfluxerr/photcorr/md.x[0],fmt='D',label='original phot. (scaled)')
		plt.errorbar(photwave,specflux/photcorr/recalfunc(
			md.x,photwave,specrange_wavescale_specrecal),
					 yerr=specfluxerr/photcorr/recalfunc(
						 md.x,photwave,specrange_wavescale_specrecal),
					 fmt='o',label='warped phot.')
		plt.legend()
		#import pdb; pdb.set_trace()
	if md.success: return md.x
	else: return [0.]*(nrecalpars+1)
	
	# if nothing exists after that, we're boned
	# could consider cutting the whole spectrum,
	# but for now I'm keeping it and trusting the
	# Gauss-Newton process
	#return [0.]*nrecalpars

def chifunc(coeffs,photflux,specflux,photfluxerr,specfluxerr,photwave,specrange_wavescale_specrecal):

	recalexp = recalfunc(coeffs[1:],photwave,specrange_wavescale_specrecal)
	return np.sum((photflux*recalexp-coeffs[0]*specflux)**2./(2*((photfluxerr*recalexp)**2.+specfluxerr**2.)))

def recalfunc(coeffs,photwave,specrange_wavescale_specrecal):

	pow=coeffs.size-np.arange(coeffs.size)
	recalCoord=(photwave-np.mean(photwave))/specrange_wavescale_specrecal
	drecaltermdrecal=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
	recalexp=np.exp((drecaltermdrecal*coeffs[np.newaxis,:]).sum(axis=1))
	
	return recalexp
