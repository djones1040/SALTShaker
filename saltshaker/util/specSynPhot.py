#!/usr/bin/env python
# D. Jones - 11/1/2019
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid as trapz
import numpy as np
from saltshaker.util.synphot import synphot
from sncosmo.constants import HC_ERG_AA

def getScaleForSN(spectrum,photdata,kcordict,survey,colorcut=0.1):

	dwave = np.median(spectrum['wavelength'][1:]-spectrum['wavelength'][:-1])
	primarywave = kcordict[survey]['primarywave']
	maxwave = np.max(spectrum['wavelength'])
	minwave = np.min(spectrum['wavelength'])
	
	scale_guess = []
	scale_guess_err = []
	specmag,photmag = [],[]
	flt_out = []
	for flt in np.unique(photdata['filt']):
		# preliminaries
		if kcordict[survey][flt]['magsys'] == 'AB': primarykey = 'AB'
		elif kcordict[survey][flt]['magsys'].upper() == 'VEGA': primarykey = 'Vega'
		elif kcordict[survey][flt]['magsys'] == 'BD17': primarykey = 'BD17'
		stdmag = synphot(
			primarywave,kcordict[survey][primarykey],
			filtwave=kcordict[survey][flt]['filtwave'],
			filttp=kcordict[survey][flt]['filttrans'],
			zpoff=0) - kcordict[survey][flt]['primarymag']
		fluxfactor = 10**(0.4*(stdmag+27.5))
		wht = np.sum(kcordict[survey][flt]['filttrans'][
			(kcordict[survey][flt]['filtwave'] > maxwave) |
			(kcordict[survey][flt]['filtwave'] < minwave)])/np.sum(kcordict[survey][flt]['filttrans'])
		if wht > 0.02: continue
		if len(photdata['filt'][photdata['filt'] == flt]) == 1: continue
		flt_out += [flt]

		filttrans = kcordict[survey][flt]['filttrans']
		pbspl = np.interp(spectrum['wavelength'],filtwave,filttrans)
		denom = np.trapz(pbspl,spectrum['wavelength'])

		pbspl = np.interp(spectrum['wavelength'],filtwave,filttrans)
		pbspl *= spectrum['wavelength']
		denom = trapz(pbspl,spectrum['wavelength'])
		pbspl /= denom*HC_ERG_AA

		
		phot1d = interp1d(photdata['mjd'][photdata['filt'] == flt],
						  photdata['fluxcal'][photdata['filt'] == flt],
						  axis=0,kind='linear',bounds_error=True,
						  assume_sorted=True)
		photerr1d = interp1d(photdata['mjd'][photdata['filt'] == flt],
							 photdata['fluxcalerr'][photdata['filt'] == flt],
							 axis=0,kind='linear',bounds_error=True,
							 assume_sorted=True)

		try:
			flux = phot1d(spectrum['mjd'])
			fluxerr = photerr1d(spectrum['mjd'])
		except: continue
		
		synph = np.sum(spectrum['flux']*pbspl)*dwave*fluxfactor
		synpherr = np.sqrt(np.sum((spectrum['fluxerr']*pbspl)**2.))*dwave*fluxfactor
		scale_guess += [flux/synph]
		scale_guess_err += [scale_guess[-1]*np.sqrt((synpherr/synph)**2. + (fluxerr/flux)**2.)]

		specmag += [-2.5*np.log10(synph) + 27.5]
		photmag += [-2.5*np.log10(flux) + 27.5]

		
	if not len(scale_guess):
		return 1,None
	elif len(scale_guess) == 1:
		return 1,None

	colordiffs = np.array([])
	for photmag1,photmag2,specmag1,specmag2 in zip(
			photmag[:-1],photmag[1:],specmag[:-1],specmag[1:]):
		photcolor = photmag1-photmag2
		speccolor = specmag1-specmag2
		colordiffs = np.append(colordiffs,photcolor-speccolor)

	scale_guess,scale_guess_err = np.array(scale_guess),np.array(scale_guess_err)
	scale_out = np.average(scale_guess,weights=1/scale_guess_err**2.)
	#print(scale_out)
	return np.log(1/scale_out),colordiffs

def getColorsForSN(spectrum,photdata,kcordict,survey,colorcut=0.1):

	dwave = np.median(spectrum['wavelength'][1:]-spectrum['wavelength'][:-1])
	primarywave = kcordict[survey]['primarywave']
	maxwave = np.max(spectrum['wavelength'])
	minwave = np.min(spectrum['wavelength'])
	
	scale_guess = []
	scale_guess_err = []
	specmag,photmag = [],[]
	flt_out = []
	for flt in np.unique(photdata['filt']):
		# preliminaries
		if kcordict[survey][flt]['magsys'] == 'AB': primarykey = 'AB'
		elif kcordict[survey][flt]['magsys'].upper() == 'VEGA': primarykey = 'Vega'
		elif kcordict[survey][flt]['magsys'] == 'BD17': primarykey = 'BD17'
		stdmag = synphot(
			primarywave,kcordict[survey][primarykey],
			filtwave=kcordict[survey][flt]['filtwave'],
			filttp=kcordict[survey][flt]['filttrans'],
			zpoff=0) - kcordict[survey][flt]['primarymag']
		fluxfactor = 10**(0.4*(stdmag+27.5))
		wht = np.sum(kcordict[survey][flt]['filttrans'][
			(kcordict[survey][flt]['filtwave'] > maxwave) |
			(kcordict[survey][flt]['filtwave'] < minwave)])/np.sum(kcordict[survey][flt]['filttrans'])
		if wht > 0.02: continue
		if len(photdata['filt'][photdata['filt'] == flt]) == 1: continue
		flt_out += [flt]
		filtwave=kcordict[survey][flt]['filtwave']
		filttrans = kcordict[survey][flt]['filttrans']
		pbspl = np.interp(spectrum['wavelength'],filtwave,filttrans)
		denom = np.trapz(pbspl,spectrum['wavelength'])

		pbspl = np.interp(spectrum['wavelength'],filtwave,filttrans)
		pbspl *= spectrum['wavelength']
		denom = trapz(pbspl,spectrum['wavelength'])
		pbspl /= denom*HC_ERG_AA

		try:
			phot1d = interp1d(photdata['mjd'][photdata['filt'] == flt],
							  photdata['fluxcal'][photdata['filt'] == flt],
							  axis=0,kind='linear',#bounds_error=True,
							  assume_sorted=True,fill_value="extrapolate")
		except:
			import pdb; pdb.set_trace()
		photerr1d = interp1d(photdata['mjd'][photdata['filt'] == flt],
							 photdata['fluxcalerr'][photdata['filt'] == flt],
							 axis=0,kind='linear',#bounds_error=True,
							 assume_sorted=True,fill_value="extrapolate")
		#try:
		flux = phot1d(spectrum['mjd'])
		fluxerr = photerr1d(spectrum['mjd'])
		#except: continue
		if np.min(np.abs(photdata['mjd'][photdata['filt'] == flt] - spectrum['mjd'])) > 3:
			continue
		
		synph = np.sum(spectrum['flux']*pbspl)*dwave*fluxfactor
		synpherr = np.sqrt(np.sum((spectrum['fluxerr']*pbspl)**2.))*dwave*fluxfactor
		scale_guess += [flux/synph]
		scale_guess_err += [scale_guess[-1]*np.sqrt((synpherr/synph)**2. + (fluxerr/flux)**2.)]

		specmag += [-2.5*np.log10(synph) + 27.5]
		if flux > 0: photmag += [-2.5*np.log10(flux) + 27.5]
		else: photmag += [np.nan]

	if not len(scale_guess):
		return None
	elif len(scale_guess) == 1:
		return None

	colordiffs = np.array([])
	for photmag1,photmag2,specmag1,specmag2 in zip(
			photmag[:-1],photmag[1:],specmag[:-1],specmag[1:]):
		photcolor = photmag1-photmag2
		speccolor = specmag1-specmag2
		colordiffs = np.append(colordiffs,photcolor-speccolor)
		
	scale_guess,scale_guess_err = np.array(scale_guess),np.array(scale_guess_err)
	scale_out = np.average(scale_guess,weights=1/scale_guess_err**2.)
	#import pdb; pdb.set_trace()
	return colordiffs
