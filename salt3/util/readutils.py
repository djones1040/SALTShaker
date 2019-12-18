import os
import numpy as np
from salt3.util import snana
from salt3.util.estimate_tpk_bazin import estimate_tpk_bazin
from astropy.io import fits
from salt3.initfiles import init_rootdir
from salt3.data import data_rootdir
from astroquery.irsa_dust import IrsaDust
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
from time import time

def rdkcor(surveylist,options,addwarning=None):

	kcordict = {}
	for survey in surveylist:
		kcorfile = options.__dict__['%s_kcorfile'%survey]
		subsurveys = options.__dict__['%s_subsurveylist'%survey].split(',')
		kcorfile = os.path.expandvars(kcorfile)
		if not os.path.exists(kcorfile):
			print('kcor file %s does not exist.	 Checking %s/kcor'%(kcorfile,data_rootdir))
			kcorfile = '%s/kcor/%s'%(data_rootdir,kcorfile)
			if not os.path.exists(kcorfile):
				raise RuntimeError('kcor file %s does not exist'%kcorfile)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			try:
				hdu = fits.open(kcorfile)
				zpoff = hdu[1].data
				snsed = hdu[2].data
				filtertrans = hdu[5].data
				primarysed = hdu[6].data
				hdu.close()
			except:
				raise RuntimeError('kcor file format is non-standard for kcor file %s'%kcorfile)

		for subsurvey in subsurveys:
			kcorkey = '%s(%s)'%(survey,subsurvey)
			if not subsurvey: kcorkey = survey[:]
			kcordict[kcorkey] = {}
			kcordict[kcorkey]['filtwave'] = filtertrans['wavelength (A)']
			kcordict[kcorkey]['primarywave'] = primarysed['wavelength (A)']
			kcordict[kcorkey]['snflux'] = snsed['SN Flux (erg/s/cm^2/A)']

			if 'AB' in primarysed.names:
				kcordict[kcorkey]['AB'] = primarysed['AB']
			if 'Vega' in primarysed.names:
				kcordict[kcorkey]['Vega'] = primarysed['Vega']
			if 'VEGA' in primarysed.names:
				kcordict[kcorkey]['Vega'] = primarysed['VEGA']
			if 'BD17' in primarysed.names:
				kcordict[kcorkey]['BD17'] = primarysed['BD17']
			for filt in zpoff['Filter Name']:
				kcordict[kcorkey][filt.split('-')[-1].split('/')[-1]] = {}
				kcordict[kcorkey][filt.split('-')[-1].split('/')[-1]]['fullname'] = filt.split('/')[0][1:]
				kcordict[kcorkey][filt.split('-')[-1].split('/')[-1]]['filttrans'] = filtertrans[filt]
				lambdaeff = np.sum(kcordict[kcorkey]['filtwave']*filtertrans[filt])/np.sum(filtertrans[filt])
				kcordict[kcorkey][filt.split('-')[-1].split('/')[-1]]['lambdaeff'] = lambdaeff
				kcordict[kcorkey][filt.split('-')[-1].split('/')[-1]]['zpoff'] = \
					zpoff['ZPOff(Primary)'][zpoff['Filter Name'] == filt][0]
				kcordict[kcorkey][filt.split('-')[-1].split('/')[-1]]['magsys'] = \
					zpoff['Primary Name'][zpoff['Filter Name'] == filt][0]
				kcordict[kcorkey][filt.split('-')[-1].split('/')[-1]]['primarymag'] = \
					zpoff['Primary Mag'][zpoff['Filter Name'] == filt][0]

	initBfilt = '%s/Bessell90_B.dat'%init_rootdir
	filtwave,filttp = np.genfromtxt(initBfilt,unpack=True)
		
	primarywave,primarysed = np.genfromtxt('%s/flatnu.dat'%init_rootdir,unpack=True)
	
	kcordict['default'] = {}
	kcordict['default']['Bwave'] = filtwave
	kcordict['default']['Btp'] = filttp
	kcordict['default']['AB']=primarysed
	kcordict['default']['primarywave']=primarywave
	return kcordict
			
def rdSpecData(datadict,speclist,KeepOnlySpec=False,waverange=[2000,9200]):
	if not os.path.exists(speclist):
		raise RuntimeError('speclist %s does not exist')
	
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
					print('warning: File {} contains no supernova spectra'.format(sf))
					if KeepOnlySpec: 
						print('KeepOnlySpec (debug) flag is set, removing SN')
						datadict.pop(s)
					continue
				
					#raise ValueError('File {} contains no supernova spectra'.format(sf))
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

					z = datadict[s]['zHelio']
					iGood = ((datadict[s]['specdata'][speccount]['wavelength']/(1+z) > waverange[0]) &
							 (datadict[s]['specdata'][speccount]['wavelength']/(1+z) < waverange[1]))
					datadict[s]['specdata'][speccount]['flux'] = datadict[s]['specdata'][speccount]['flux'][iGood]
					datadict[s]['specdata'][speccount]['wavelength'] = datadict[s]['specdata'][speccount]['wavelength'][iGood]
					datadict[s]['specdata'][speccount]['fluxerr'] = datadict[s]['specdata'][speccount]['fluxerr'][iGood]
					speccount+=1
			else:
				print('SNID %s has no photometry so I\'m ignoring it'%s)

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

def rdAllData(snlists,estimate_tpk,kcordict,addwarning,
			  dospec=False,KeepOnlySpec=False,peakmjdlist=None,waverange=[2000,9200]):
	datadict = {}
	if peakmjdlist:
		pksnid,pkmjd,pkmjderr = np.loadtxt(peakmjdlist,unpack=True,dtype=str)
		pkmjd,pkmjderr = pkmjd.astype('float'),pkmjderr.astype('float')
	rdtime = 0
	for snlist in snlists.split(','):
		tsn = time()
		snlist = os.path.expandvars(snlist)
		if not os.path.exists(snlist):
			print('SN list file %s does not exist.	Checking %s/trainingdata/%s'%(snlist,data_rootdir,snlist))
			snlist = '%s/trainingdata/%s'%(data_rootdir,snlist)
		if not os.path.exists(snlist):
			raise RuntimeError('SN list file %s does not exist'%snlist)

		snfiles = np.genfromtxt(snlist,dtype='str')
		snfiles = np.atleast_1d(snfiles)

		for f in snfiles:
			if f.lower().endswith('.fits'):
				raise RuntimeError('FITS extensions are not supported yet')

			if '/' not in f:
				f = '%s/%s'%(os.path.dirname(snlist),f)
			rdstart = time()
			sn = snana.SuperNova(f)
			rdtime += time()-rdstart
			
			if sn.SNID in datadict.keys():
				addwarning('SNID %s is a duplicate!	 Skipping'%sn.SNID)
				continue

			if not 'SURVEY' in sn.__dict__.keys():
				raise RuntimeError('File %s has no SURVEY key, which is needed to find the filter transmission curves'%PhotSNID[0])
			if not 'REDSHIFT_HELIO' in sn.__dict__.keys():
				raise RuntimeError('File %s has no heliocentric redshift information in the header'%sn.SNID)

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
			elif peakmjdlist:
				if sn.SNID in pksnid:
					tpk = pkmjd[sn.SNID == pksnid][0]
					tpkerr = pkmjderr[sn.SNID == pksnid][0]
					if tpkerr < 2: tpkmsg = 'termination condition is satisfied'
					else: tpkmsg = 'time of max uncertainty of +/- %.1f days is too uncertain!'%tpkerr
				else:
					tpkmsg = 'can\'t fint tmax in file %s'%peakmjdlist
					addwarning(tpkmsg)
					#raise RuntimeError('SN ID %s not found in peak MJD list'%sn.SNID)
			else:
				tpk = sn.SEARCH_PEAKMJD
				if type(tpk) == str:
					tpk = float(sn.SEARCH_PEAKMJD.split()[0])
				tpkmsg = 'termination condition is satisfied'

			# at least one epoch 3 days before max
			#try:
			#	if not len(sn.MJD[sn.MJD < tpk-3]):
			#		addwarning('skipping SN %s; no epochs 3 days pre-max'%sn.SNID)
			#		continue
			#except: import pdb; pdb.set_trace()

			if 'termination condition is satisfied' not in tpkmsg:
				addwarning('skipping SN %s; can\'t estimate t_max'%sn.SNID)
				continue

			if not (kcordict is None ) and sn.SURVEY not in kcordict.keys():
				raise RuntimeError('survey %s not in kcor file'%(sn.SURVEY))

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
			if 'MWEBV' in sn.__dict__.keys():
				try: datadict[sn.SNID]['MWEBV'] = float(sn.MWEBV.split()[0])
				except: datadict[sn.SNID]['MWEBV'] = float(sn.MWEBV)
			elif 'RA' in sn.__dict__.keys() and 'DEC' in sn.__dict__.keys():
				print('determining MW E(B-V) from IRSA for SN %s using RA/Dec in file'%sn.SNID)
				sc = SkyCoord(sn.RA,sn.DEC,frame="fk5",unit=u.deg)
				datadict[sn.SNID]['MWEBV'] = IrsaDust.get_query_table(sc)['ext SandF mean'][0]
			else:
				raise RuntimeError('Could not determine E(B-V) from files.	Set MWEBV keyword in input file header for SN %s'%sn.SNID)

		if dospec:
			tspec = time()
			datadict = rdSpecData(datadict,snlist,KeepOnlySpec=KeepOnlySpec,waverange=waverange)

	print('reading data files took %.1f'%(rdtime))
	if not len(datadict.keys()):
		raise RuntimeError('no light curve data to train on!!')
		
	return datadict
	
