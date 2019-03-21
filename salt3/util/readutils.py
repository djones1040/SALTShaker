import os
import numpy as np
from salt3.util import snana
from salt3.util.estimate_tpk_bazin import estimate_tpk_bazin
from astropy.io import fits
from salt3.initfiles import init_rootdir

def rdkcor(kcorpath,addwarning):

	kcordict = {}
	for k in kcorpath:
		survey,kcorfile = k.split(',')
		kcorfile = os.path.expandvars(kcorfile)
		if not os.path.exists(kcorfile):
			raise RuntimeError('kcor file %s does not exist'%kcorfile)
		kcordict[survey] = {}

		try:
			hdu = fits.open(kcorfile)
			zpoff = hdu[1].data
			snsed = hdu[2].data
			filtertrans = hdu[5].data
			primarysed = hdu[6].data
			hdu.close()
		except:
			raise RuntimeError('kcor file format is non-standard')

		kcordict[survey]['filtwave'] = filtertrans['wavelength (A)']
		kcordict[survey]['primarywave'] = primarysed['wavelength (A)']
		kcordict[survey]['snflux'] = snsed['SN Flux (erg/s/cm^2/A)']
		if 'AB' in primarysed.names:
			kcordict[survey]['AB'] = primarysed['AB']
		if 'Vega' in primarysed.names:
			kcordict[survey]['Vega'] = primarysed['Vega']
		if 'BD17' in primarysed.names:
			kcordict[survey]['BD17'] = primarysed['BD17']
		for filt in zpoff['Filter Name']:
			kcordict[survey][filt.split('-')[-1].split('/')[-1]] = {}
			kcordict[survey][filt.split('-')[-1].split('/')[-1]]['filttrans'] = filtertrans[filt]
			kcordict[survey][filt.split('-')[-1].split('/')[-1]]['zpoff'] = \
				zpoff['ZPOff(Primary)'][zpoff['Filter Name'] == filt][0]
			kcordict[survey][filt.split('-')[-1].split('/')[-1]]['magsys'] = \
				zpoff['Primary Name'][zpoff['Filter Name'] == filt][0]
			kcordict[survey][filt.split('-')[-1].split('/')[-1]]['primarymag'] = \
				zpoff['Primary Mag'][zpoff['Filter Name'] == filt][0]

	initBfilt = '%s/Bessell90_B.dat'%init_rootdir
	filtwave,filttp = np.genfromtxt(initBfilt,unpack=True)
		
	kcordict['default'] = {}
	kcordict['default']['Bwave'] = filtwave
	kcordict['default']['Btp'] = filttp
	return kcordict
			
def rdSpecData(datadict,speclist):
	if not os.path.exists(speclist):
		raise RuntimeError('speclist %s does not exist')
	
	try:
		snid,mjd,specfiles = np.genfromtxt(speclist,unpack=True,dtype='str')
		snid,mjd,specfiles = np.atleast_1d(snid),np.atleast_1d(mjd),np.atleast_1d(specfiles)
		snanaSpec=False
	except:
		specfiles=np.genfromtxt(speclist,dtype='str')
		specfiles=np.atleast_1d(specfiles)
		snanaSpec=True
		
	if snanaSpec:
		for sf in specfiles:
		
			if '/' not in sf:
				sf = '%s/%s'%(os.path.dirname(speclist),sf)
			if not os.path.exists(sf):
				raise RuntimeError('specfile %s does not exist'%sf)
			sn=snana.SuperNova(sf)
			s=sn.name
			if s in datadict.keys():
				tpk=datadict[s]['tpk']
				if 'specdata' not in datadict[s].keys():
					datadict[s]['specdata'] = {}
					speccount = 0
				else:
					speccount = len(datadict[s]['specdata'].keys())
					
				if len(sn.SPECTRA)==0:
					raise ValueError('File {} contains no supernova spectra'.format(sf))
				for k in sn.SPECTRA:
					spec=sn.SPECTRA[k]
					m=spec['SPECTRUM_MJD']
					datadict[s]['specdata'][speccount] = {}
					datadict[s]['specdata'][speccount]['fluxerr'] = spec['FLAMERR']
					if 'LAMAVG' in spec.keys():
						datadict[s]['specdata'][speccount]['wavelength'] = spec['LAMAVG']
					elif 'LAMMIN' in sn.SPECTRA[k].keys() and 'LAMMAX' in spec.keys():
						datadict[s]['specdata'][speccount]['wavelength'] = (spec['LAMMIN']+spec['LAMMAX'])/2
					else:
						raise RuntimeError('couldn\t find wavelength data in photometry file')

					datadict[s]['specdata'][speccount]['flux'] = spec['FLAM']
					datadict[s]['specdata'][speccount]['tobs'] = m - tpk
					datadict[s]['specdata'][speccount]['mjd'] = m
					speccount+=1
			else:
				print('SNID %s has no photometry so I\'m ignoring it')

	else:
		for s,m,sf in zip(snid,mjd,specfiles):
			try: m = float(m)
			except: m = snana.date_to_mjd(m)

			if '/' not in sf:
				sf = '%s/%s'%(os.path.dirname(speclist),sf)
				
			if not os.path.exists(sf):
				raise RuntimeError('specfile %s does not exist'%sf)
		
			if s in datadict.keys():
				tpk=datadict[s]['tpk']
				if 'specdata' not in datadict[s].keys():
					datadict[s]['specdata'] = {}
					speccount = 0
				else:
					speccount = len(datadict[s]['specdata'].keys())
		
				try:
					wave,flux,fluxerr = np.genfromtxt(sf,unpack=True,usecols=[0,1,2])
		
				except:
					wave,flux = np.genfromtxt(sf,unpack=True,usecols=[0,1])
					fluxerr=np.tile(np.nan,flux.size)

				datadict[s]['specdata'][speccount] = {}
				datadict[s]['specdata'][speccount]['fluxerr'] = fluxerr
				datadict[s]['specdata'][speccount]['wavelength'] = wave
				datadict[s]['specdata'][speccount]['flux'] = flux
				datadict[s]['specdata'][speccount]['tobs'] = m - tpk
				datadict[s]['specdata'][speccount]['mjd'] = m
			else:
				print('SNID %s has no photometry so I\'m ignoring it'%s)

	return datadict

def rdAllData(snlist,estimate_tpk,addwarning,speclist=None):
	datadict = {}

	if not os.path.exists(snlist):
		raise RuntimeError('SN list %s doesn\'t exist'%snlist)
	snfiles = np.genfromtxt(snlist,dtype='str')
	snfiles = np.atleast_1d(snfiles)

	for f in snfiles:
		if f.lower().endswith('.fits'):
			raise RuntimeError('FITS extensions are not supported yet')

		if '/' not in f:
			f = '%s/%s'%(os.path.dirname(snlist),f)
		sn = snana.SuperNova(f)

		if sn.SNID in datadict.keys():
			addwarning('SNID %s is a duplicate!  Skipping'%sn.SNID)
			continue

		if not 'SURVEY' in sn.__dict__.keys():
			raise RuntimeError('File %s has no SURVEY key, which is needed to find the filter transmission curves'%PhotSNID[0])
		if not 'REDSHIFT_HELIO' in sn.__dict__.keys():
			raise RuntimeError('File %s has no heliocentric redshift information in the header'%PhotSNID[0])

		if 'PEAKMJD' in sn.__dict__.keys(): sn.SEARCH_PEAKMJD = sn.PEAKMJD
		zHel = float(sn.REDSHIFT_HELIO.split('+-')[0])
		if estimate_tpk:
			if 'B' in sn.FLT:
				tpk,tpkmsg = estimate_tpk_bazin(
					sn.MJD[sn.FLT == 'B'],sn.FLUXCAL[sn.FLT == 'B'],sn.FLUXCALERR[sn.FLT == 'B'],max_nfev=100000,t0=sn.SEARCH_PEAKMJD)
			elif 'g' in sn.FLT:
				tpk,tpkmsg = estimate_tpk_bazin(
					sn.MJD[sn.FLT == 'g'],sn.FLUXCAL[sn.FLT == 'g'],sn.FLUXCALERR[sn.FLT == 'g'],max_nfev=100000,t0=sn.SEARCH_PEAKMJD)
			elif 'c' in sn.FLT:
				tpk,tpkmsg = estimate_tpk_bazin(
					sn.MJD[sn.FLT == 'c'],sn.FLUXCAL[sn.FLT == 'c'],sn.FLUXCALERR[sn.FLT == 'c'],max_nfev=100000,t0=sn.SEARCH_PEAKMJD)
			else:
				raise RuntimeError('need a blue filter to estimate tmax')
		else:
			tpk = sn.SEARCH_PEAKMJD
			tpkmsg = 'termination condition is satisfied'

		# at least one epoch 3 days before max
		if not len(sn.MJD[sn.MJD < tpk-3]):
			addwarning('skipping SN %s; no epochs 3 days pre-max'%sn.SNID)
			continue

		if 'termination condition is satisfied' not in tpkmsg:
			addwarning('skipping SN %s; can\'t estimate t_max'%sn.SNID)
			continue

		datadict[sn.SNID] = {'snfile':f,
							 'zHelio':zHel,
							 'survey':sn.SURVEY,
							 'tpk':tpk}
		#datadict[snid]['zHelio'] = zHel
		
		# TODO: flux errors
		datadict[sn.SNID]['specdata'] = {} 				
		datadict[sn.SNID]['photdata'] = {}
		datadict[sn.SNID]['photdata']['tobs'] = sn.MJD - tpk
		datadict[sn.SNID]['photdata']['mjd'] = sn.MJD
		datadict[sn.SNID]['photdata']['fluxcal'] = sn.FLUXCAL
		datadict[sn.SNID]['photdata']['fluxcalerr'] = sn.FLUXCALERR
		datadict[sn.SNID]['photdata']['filt'] = sn.FLT

	if not len(datadict.keys()):
		raise RuntimeError('no light curve data to train on!!')
		
	if speclist:
		datadict = rdSpecData(datadict,speclist)
		
	return datadict
	
