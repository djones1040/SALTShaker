#!/usr/bin/env python
import numpy as np
from saltshaker.data import data_rootdir
import glob

survey_filters = {'SNLS3_LC':'griz',
				  'Hicken2009_LC':'UBVRIfhjklabcde',
				  'OTHER_LOWZ_LC':'UBVRIfhjkl',
				  'Hamuy1996_LC':'UBVRIfhjkl',
				  'Jha2006_LC':'UBVRIfhjkl',
				  'SDSS':'tvywxABCfhjklugriz',
				  'Riess1999_LC':'UBVRIfhjkl'}

filtdict = {'SDSS::u':'u',
			'SDSS::g':'g',
			'SDSS::r':'r',
			'SDSS::i':'i',
			'SDSS::z':'z',
			'SWOPE2::u':'t',
			'SWOPE2::B':'v',
			'SWOPE2::V':'y',
			'SWOPE2::g':'A',
			'SWOPE2::r':'B',
			'SWOPE2::i':'C',
			'MEGACAMPSF::g':'g',
			'MEGACAMPSF::r':'r',
			'MEGACAMPSF::i':'i',
			'MEGACAMPSF::z':'z',
			'4SHOOTER2::Us':'a',
			'4SHOOTER2::B':'b',
			'4SHOOTER2::V':'c',
			'4SHOOTER2::R':'d',
			'4SHOOTER2::I':'e',
			'KEPLERCAM::Us':'f',
			'KEPLERCAM::B':'h',
			'KEPLERCAM::V':'j',
			'KEPLERCAM::r':'k',
			'KEPLERCAM::i':'l',
			'STANDARD::U':'U',
			'STANDARD::B':'B',
			'STANDARD::V':'V',
			'STANDARD::R':'R',
			'STANDARD::I':'I'}

_jladir = f'{data_rootdir}/trainingdata/jla'

def main(residsfile='/Users/David/Dropbox/research/SALT2/trainsalt/trainingdir/lcresiduals.list'):

	z,x0,x1,c,snid = np.loadtxt(
		residsfile,unpack=True,usecols=[1,2,3,4,-1],dtype=str)

	with open('snpca_training_v6_resids.txt','w') as fout:
		print('# SNID zHelio x0 x1 c',file=fout)

		for xs,x0s,x1s,cs,snids in zip(z,x0,x1,c,snid):
			print(f'{snids} {xs} {x0s} {x1s} {cs}',file=fout)

def lcfiles_from_training(trainingdir='',outdir='JLA_training_origlc'):

	lcfiles = np.append(glob.glob(f'{_jladir}/lc-*.list'),glob.glob(f'{_jladir}/SDSS*DAT'))
	surveylist = []
	#['SNLS3_LC', 'Hicken2009_LC', 'OTHER_LOWZ_LC', 'Hamuy1996_LC', 'Jha2006_LC', 'SDSS', 'Riess1999_LC']

	for l in lcfiles:
		print(l)
		header = {}
		mjdlist,fluxlist,fluxerrlist,zptlist,fltlist,zpsyslist = \
			[],[],[],[],[],[]
		with open(l) as fin:
			for line in fin:
				line = line.replace('\n','')
				if line.startswith('@'):
					header[line.split()[0][1:]] = line.split()[1]
				elif line.startswith('#') or not line:
					continue
				else:
					mjd,flux,fluxerr,zpt,flt,zpsys = line.split()
					mjdlist += [mjd]
					fluxlist += [flux]
					fluxerrlist += [fluxerr]
					zptlist += [zpt]
					fltlist += [flt]
					zpsyslist += [zpsys]
		mjdlist,fluxlist,fluxerrlist,zptlist,fltlist,zpsyslist = \
			np.array(mjdlist).astype(float),np.array(fluxlist).astype(float),\
			np.array(fluxerrlist).astype(float),np.array(zptlist).astype(float),\
			np.array(fltlist),np.array(zpsyslist)

		if header['SURVEY'] not in surveylist:
			surveylist += [header['SURVEY']]


		# dru run to get filter list
		fullfilt,abbrevfilt = np.array([]),np.array([])
		for m,f,fe,zp,flt in zip(
				mjdlist,fluxlist,fluxerrlist,zptlist,fltlist):
			if flt in filtdict.keys():
				snana_flt = filtdict[flt]
				fullfilt = np.append(fullfilt,flt)
				abbrevfilt = np.append(abbrevfilt,snana_flt)
			else:
				print(flt)
				raise RuntimeError('filter not in dictionary!')


		#zfinal = get_vpec(header['Z_HELIO'])
		with open(f"{outdir}/{header['SN']}.dat",'w') as fout:
			headertxt = f"""SURVEY: {header['SURVEY']}
SNID: {header['SN']}
IAUC: UNKNOWN
RA: 0.0
DECL: 0.0
PHOTOMETRY_VERSION: {header['SURVEY']}
MWEBV: {header['MWEBV']} MW E(B-V)
REDSHIFT_HELIO: {header['Z_HELIO']} +- 0.001    (HELIO)
REDSHIFT_Status: warning: redshift err fudged
SEARCH_PEAKMJD: {header['DayMax']}
FILTERS: {survey_filters[header['SURVEY']]}

NOBS: {len(fluxlist)}
NVAR: 5
VARLIST:  MJD   FLT       FIELD    FLUXCAL   FLUXCALERR
"""
			print(headertxt,file=fout)

			fullfilt,abbrevfilt = np.array([]),np.array([])
			for m,f,fe,zp,flt in zip(
					mjdlist,fluxlist,fluxerrlist,zptlist,fltlist):
				if flt in filtdict.keys():
					snana_flt = filtdict[flt]
					fullfilt = np.append(fullfilt,flt)
					abbrevfilt = np.append(abbrevfilt,snana_flt)
				else:
					print(flt)
					raise RuntimeError('filter not in dictionary!')
				ft = f*10**(-0.4*(zp-27.5))
				fte = fe*10**(-0.4*(zp-27.5))
				print(f"OBS: {m:.3f} {snana_flt}  NULL    {ft:.4f} {fte:.4f}",file=fout)
			for a in np.unique(abbrevfilt):
				if len(np.unique(fullfilt[abbrevfilt == a])) > 1:
					raise RuntimeError(f'multiple matches in file {l}')
			print("END_PHOTOMETRY:",file=fout)

			# now just copy over the spectra from previous directory
			# as usual this is ugly
			previous_file = glob.glob(f"/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/JLA_training_orig/*_{header['SN'].replace('sn','').replace('.0','')}.*")
			if not len(previous_file):
				previous_file = glob.glob(f"/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/JLA_training_orig/*_{header['SN'].replace('sn',''.replace('.0','')).upper()}*")
			if not len(previous_file):
				previous_file = glob.glob(f"/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/JLA_training_orig/*_SN{int(header['SN'].replace('sn','').replace('.0','')):06.0f}.*")
			if len(previous_file) > 1 or not len(previous_file): import pdb; pdb.set_trace()

			nlamlist = []
			with open(previous_file[0]) as fin:
				start = False
				nlam = 0
				for line in fin:
					if start and line.startswith('SPEC:'):
						nlam += 1
					if line.startswith('END_PHOTOMETRY:'):
						start = True
					if line.startswith('SPECTRUM_END:'):
						nlamlist += [nlam]
						nlam = 0

			listidx = 0
			with open(previous_file[0]) as fin:
				start = False
				for line in fin:
					line = line.replace('\n','')
					if start and line.startswith('SPECTRUM_MJD:'):
						print(line,file=fout)
						print(f'SPECTRUM_NLAM:     {nlamlist[listidx]:.0f}',file=fout)
					elif start:
						print(line,file=fout)
					if line.startswith('END_PHOTOMETRY:'):
						start = True
					if line.startswith('SPECTRUM_END'):
						listidx += 1
			
	print(surveylist)

if __name__ == "__main__":
	#main()
	lcfiles_from_training()
