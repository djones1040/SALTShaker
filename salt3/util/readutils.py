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
import scipy.stats as ss
import logging
log=logging.getLogger(__name__)

def rdkcor(surveylist,options):

	kcordict = {}
	for survey in surveylist:
		kcorfile = options.__dict__['%s_kcorfile'%survey]
		subsurveys = options.__dict__['%s_subsurveylist'%survey].split(',')
		kcorfile = os.path.expandvars(kcorfile)
		if not os.path.exists(kcorfile):
			log.info('kcor file %s does not exist.	 Checking %s/kcor'%(kcorfile,data_rootdir))
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

		
	primarywave,primarysed = np.genfromtxt('%s/flatnu.dat'%init_rootdir,unpack=True)
	
	kcordict['default'] = {}
	initBfilt = '%s/Bessell90_B.dat'%init_rootdir
	filtwave,filttp = np.genfromtxt(initBfilt,unpack=True)
	kcordict['default']['Bwave'] = filtwave
	kcordict['default']['Btp'] = filttp
	
	initVfilt = '%s/Bessell90_V.dat'%init_rootdir
	filtwave,filttp = np.genfromtxt(initVfilt,unpack=True)
	kcordict['default']['Vwave'] = filtwave
	kcordict['default']['Vtp'] = filttp
	
	kcordict['default']['AB']=primarysed
	kcordict['default']['primarywave']=primarywave
	return kcordict
			
def rdSpecData(datadict,speclist,KeepOnlySpec=False,waverange=[2000,9200],binspecres=None):
	if not os.path.exists(speclist):
		raise RuntimeError('speclist %s does not exist')
	
	specfiles=np.genfromtxt(speclist,dtype='str')
	specfiles=np.atleast_1d(specfiles)
	snanaSpec=True
	
	if KeepOnlySpec: log.info('KeepOnlySpec (debug) flag is set, removing any supernova without spectra')

	if snanaSpec:
		for sf in specfiles:
		
			if '/' not in sf:
				sf = '%s/%s'%(os.path.dirname(speclist),sf)
			if not os.path.exists(sf):
				raise RuntimeError('specfile %s does not exist'%sf)
			sn=snana.SuperNova(sf)
			s=str(sn.name)
			if s in datadict.keys():
				tpk=datadict[s]['tpk']
				if 'specdata' not in datadict[s].keys():
					datadict[s]['specdata'] = {}
					speccount = 0
				else:
					speccount = len(datadict[s]['specdata'].keys())
				
				if len(sn.SPECTRA)==0:
					log.debug('File {} contains no supernova spectra'.format(sf))
					if KeepOnlySpec: 
						datadict.pop(s)
					continue
				
					#raise ValueError('File {} contains no supernova spectra'.format(sf))
				for k in sn.SPECTRA:
					spec=sn.SPECTRA[k]
					m=spec['SPECTRUM_MJD']
					#if '95ac' in sf:
					#	import pdb; pdb.set_trace()
					if m-tpk < -19 or m-tpk > 49:
						speccount += 1
						continue

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
							 (datadict[s]['specdata'][speccount]['wavelength']/(1+z) < waverange[1])) # &
							 #(datadict[s]['specdata'][speccount]['flux']/datadict[s]['specdata'][speccount]['fluxerr'] > 3))
					#if s == '05D2ci':
					#	import pdb; pdb.set_trace()
					#if len(iGood) <= 5:
					#	datadict[s]['specdata'].pop(speccount)
					#	continue
					
					#import pdb; pdb.set_trace()
					if binspecres is not None:
						flux = datadict[s]['specdata'][speccount]['flux'][iGood]
						wavelength = datadict[s]['specdata'][speccount]['wavelength'][iGood]
						fluxerr = datadict[s]['specdata'][speccount]['fluxerr'][iGood]
						weights = 1/fluxerr**2.
						
						def weighted_avg(values):
							"""
							Return the weighted average and standard deviation.
							values, weights -- Numpy ndarrays with the same shape.
							"""
							#try:
							average = np.average(flux[values], weights=weights[values])
							variance = np.average((flux[values]-average)**2, weights=weights[values])  # Fast and numerically precise
							#except:
							#	import pdb; pdb.set_trace()
								
							return average

						def weighted_err(values):
							"""
							Return the weighted average and standard deviation.
							values, weights -- Numpy ndarrays with the same shape.
							"""
							average = np.average(flux[values], weights=weights[values])
							variance = np.average((flux[values]-average)**2, weights=weights[values])  # Fast and numerically precise
							return np.sqrt(variance) #/np.sqrt(len(values))


						
						#wavebins = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/binspecres)
						wavebins = np.linspace(np.min(wavelength),np.max(wavelength),(np.max(wavelength)-np.min(wavelength))/(binspecres))#*(1+z)))
						binned_flux = ss.binned_statistic(wavelength,range(len(flux)),bins=wavebins,statistic=weighted_avg).statistic
						binned_fluxerr = ss.binned_statistic(wavelength,range(len(flux)),bins=wavebins,statistic=weighted_err).statistic
						iGood = (binned_flux == binned_flux) # & (binned_flux/binned_fluxerr > 3)
						datadict[s]['specdata'][speccount]['flux'] = binned_flux[iGood]
						datadict[s]['specdata'][speccount]['wavelength'] = (wavebins[1:][iGood]+wavebins[:-1][iGood])/2.
						datadict[s]['specdata'][speccount]['fluxerr'] = binned_fluxerr[iGood]
					else:
						
						datadict[s]['specdata'][speccount]['flux'] = datadict[s]['specdata'][speccount]['flux'][iGood]
						datadict[s]['specdata'][speccount]['wavelength'] = datadict[s]['specdata'][speccount]['wavelength'][iGood]
						datadict[s]['specdata'][speccount]['fluxerr'] = datadict[s]['specdata'][speccount]['fluxerr'][iGood]
					# error floor
					try:
						datadict[s]['specdata'][speccount]['fluxerr'] = np.sqrt(datadict[s]['specdata'][speccount]['fluxerr']**2. + \
																				(0.005*np.max(datadict[s]['specdata'][speccount]['flux']))**2.)
					except:
						import pdb; pdb.set_trace()

					speccount+=1

			else:
				log.debug('SNID %s has no photometry so I\'m ignoring it'%s)

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

				if binspecres is not None:
					weights = 1/fluxerr**2.
						
					def weighted_avg(values):
						"""
						Return the weighted average and standard deviation.
						values, weights -- Numpy ndarrays with the same shape.
						"""
						average = np.average(values, weights=weights)
						variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise

						return average

					def weighted_err(values):
						"""
						Return the weighted average and standard deviation.
						values, weights -- Numpy ndarrays with the same shape.
						"""
						average = np.average(values, weights=weights)
						variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
						return np.sqrt(variance)/np.sqrt(len(values))
					
					wavebins = np.linspace(waverange[0],waverange[1],(waverange[1]-waverange[0])/binspecres)

					binned_flux = ss.binned_statistic(wave,flux,bins=wavebins,statistic=weighted_avg).statistic
					binned_fluxerr = ss.binned_statistic(wave,flux,bins=wavebins,statistic=weighted_err).statistic
					
					datadict[s]['specdata'][speccount] = {}
					datadict[s]['specdata'][speccount]['fluxerr'] = binned_fluxerr
					datadict[s]['specdata'][speccount]['wavelength'] = (wavebins[1:]+wavebins[0:])/2.
					datadict[s]['specdata'][speccount]['flux'] = binned_flux
					datadict[s]['specdata'][speccount]['tobs'] = m - tpk
					datadict[s]['specdata'][speccount]['mjd'] = m

				else:
					datadict[s]['specdata'][speccount] = {}
					datadict[s]['specdata'][speccount]['fluxerr'] = fluxerr
					datadict[s]['specdata'][speccount]['wavelength'] = wave
					datadict[s]['specdata'][speccount]['flux'] = flux
					datadict[s]['specdata'][speccount]['tobs'] = m - tpk
					datadict[s]['specdata'][speccount]['mjd'] = m

				
			else:
				log.debug('SNID %s has no photometry so I\'m ignoring it'%s)

	return datadict

def rdAllData(snlists,estimate_tpk,kcordict,
			  dospec=False,KeepOnlySpec=False,peakmjdlist=None,waverange=[2000,9200],binspecres=None):
	datadict = {}
	if peakmjdlist:
		pksnid,pkmjd = np.loadtxt(peakmjdlist,unpack=True,dtype=str,usecols=[0,1])
		pkmjd = pkmjd.astype('float')
	rdtime = 0
	for snlist in snlists.split(','):
		tsn = time()
		snlist = os.path.expandvars(snlist)
		if not os.path.exists(snlist):
			log.info('SN list file %s does not exist.	Checking %s/trainingdata/%s'%(snlist,data_rootdir,snlist))
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
			sn = snana.SuperNova(f,readspec=False)
			rdtime += time()-rdstart
			sn.SNID=str(sn.SNID)
			if sn.SNID in datadict.keys():
				log.warning('SNID %s is a duplicate!	 Skipping'%sn.SNID)
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
				if str(sn.SNID) in pksnid:
					tpk = pkmjd[str(sn.SNID) == pksnid][0]
					tpkmsg = 'termination condition is satisfied'
					#tpkerr = pkmjderr[str(sn.SNID) == pksnid][0]
					#if tpkerr < 2: tpkmsg = 'termination condition is satisfied'
					#else: tpkmsg = 'time of max uncertainty of +/- %.1f days is too uncertain!'%tpkerr
				else:
					log.warning('can\'t find tmax in file %s'%peakmjdlist)
					tpkmsg = 'can\'t find tmax in file %s'%peakmjdlist
					#raise RuntimeError('SN ID %s not found in peak MJD list'%sn.SNID)
			else:
				tpk = sn.SEARCH_PEAKMJD
				if type(tpk) == str:
					tpk = float(sn.SEARCH_PEAKMJD.split()[0])
				tpkmsg = 'termination condition is satisfied'

			# at least one epoch 3 days before max
			#try:
			#	if not len(sn.MJD[sn.MJD < tpk-3]):
			#		log.warning('skipping SN %s; no epochs 3 days pre-max'%sn.SNID)
			#		continue
			#except: import pdb; pdb.set_trace()

			if 'termination condition is satisfied' not in tpkmsg:
				log.warning('skipping SN %s; can\'t estimate t_max'%sn.SNID)
				continue

			if not (kcordict is None ) and sn.SURVEY not in kcordict.keys():
				raise RuntimeError('survey %s not in kcor file'%(sn.SURVEY))

			datadict[sn.SNID] = {'snfile':f,
								 'zHelio':zHel,
								 'survey':sn.SURVEY,
								 'tpk':tpk}
			#datadict[snid]['zHelio'] = zHel
			iGood = np.array([],dtype=int)
			for i,f in enumerate(sn.FLT):
				if sn.MJD[i]-tpk > -19 and sn.MJD[i]-tpk < 49:
					iGood = np.append(iGood,i)
				#try: kcordict[sn.SURVEY][f]['fullname']
				#except: import pdb; pdb.set_trace()
			
			# TODO: flux errors
			datadict[sn.SNID]['specdata'] = {}
			datadict[sn.SNID]['photdata'] = {}
			datadict[sn.SNID]['photdata']['tobs'] = sn.MJD[iGood] - tpk
			datadict[sn.SNID]['photdata']['mjd'] = sn.MJD[iGood]
			datadict[sn.SNID]['photdata']['fluxcal'] = sn.FLUXCAL[iGood]
			datadict[sn.SNID]['photdata']['fluxcalerr'] = sn.FLUXCALERR[iGood]
			datadict[sn.SNID]['photdata']['filt'] = sn.FLT[iGood]
			if 'MWEBV' in sn.__dict__.keys():
				try: datadict[sn.SNID]['MWEBV'] = float(sn.MWEBV.split()[0])
				except: datadict[sn.SNID]['MWEBV'] = float(sn.MWEBV)
			elif 'RA' in sn.__dict__.keys() and 'DEC' in sn.__dict__.keys():
				log.warning('determining MW E(B-V) from IRSA for SN %s using RA/Dec in file'%sn.SNID)
				sc = SkyCoord(sn.RA,sn.DEC,frame="fk5",unit=u.deg)
				datadict[sn.SNID]['MWEBV'] = IrsaDust.get_query_table(sc)['ext SandF mean'][0]
			else:
				raise RuntimeError('Could not determine E(B-V) from files.	Set MWEBV keyword in input file header for SN %s'%sn.SNID)

		if dospec:
			tspec = time()
			datadict = rdSpecData(datadict,snlist,KeepOnlySpec=KeepOnlySpec,waverange=waverange,binspecres=binspecres)

	log.info('reading data files took %.1f'%(rdtime))
	if not len(datadict.keys()):
		raise RuntimeError('no light curve data to train on!!')
		
	return datadict
	
