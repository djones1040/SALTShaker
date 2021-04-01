#!/usr/bin/env python

import numpy as np
import os
import glob
from saltshaker.util import jla, snana
from saltshaker.data import data_rootdir
from astropy.coordinates import SkyCoord
import astropy.units as u

import logging
log=logging.getLogger(__name__)

#_cfa_early_dir = "$SNDATA_ROOT/lcmerge/
_lowz_dir = "$SNDATA_ROOT/lcmerge/Pantheon_LOWZ_TEXT"
_pantheon_lowz_dir = "$SNDATA_ROOT/lcmerge/Pantheon_LOWZ_TEXT"
_jla_lowz_dir = "$SNDATA_ROOT/lcmerge/SNLS3year_JRK07"
_foundation_dir = "$SNDATA_ROOT/lcmerge/Foundation_DJ17"
#_lowz_dir = "$SNDATA_ROOT/lcmerge/02-DATA_PHOTOMETRY/DES-SN3YR_LOWZ"
_des_dir = "$SNDATA_ROOT/lcmerge/02-DATA_PHOTOMETRY/DES-SN3YR_DES"
_snls_dir = "$SNDATA_ROOT/lcmerge/JLA2014_SNLS"
_snls_orig_dir = "$SNDATA_ROOT/lcmerge/SNLS3year_MEGACAM"
_sdss_dir1 = "$SNDATA_ROOT/lcmerge/SMPv8+BOSS/SMPv8+BOSS_2004"
_sdss_dir2 = "$SNDATA_ROOT/lcmerge/SMPv8+BOSS/SMPv8+BOSS_2005"
_sdss_dir3 = "$SNDATA_ROOT/lcmerge/SMPv8+BOSS/SMPv8+BOSS_2006"
_sdss_dir4 = "$SNDATA_ROOT/lcmerge/SMPv8+BOSS/SMPv8+BOSS_2007"
_ps1_dir = "$SNDATA_ROOT/lcmerge/Pantheon_PS1MD"

_training_dirs = [_lowz_dir,_snls_dir,
				  _sdss_dir1,_sdss_dir2,_sdss_dir3,_sdss_dir4,
				  _des_dir,_foundation_dir,_ps1_dir]
_training_dirs_orig = [_jla_lowz_dir,_pantheon_lowz_dir,_snls_dir,_snls_orig_dir,
					   _sdss_dir1,_sdss_dir2,_sdss_dir3,_sdss_dir4]

#_outdir = '%s/trainingdata/snana'%(data_rootdir)
_jladir = '%s/trainingdata/jla'%(data_rootdir)
#_outdir = '%s/trainingdata/Pantheon_noPS1'%(data_rootdir)
_outdir = '%s/trainingdata/Pantheon_Found_DES'%(data_rootdir)
_outdir_orig = '%s/trainingdata/JLA_training_orig'%(data_rootdir)

_jlaspecdir = '%s/trainingdata/jla'%(data_rootdir)
_ps1specdir = '%s/trainingdata/ps1spec_formatted'%(data_rootdir)
_foundspecdir = '%s/trainingdata/foundationspec_formatted'%(data_rootdir)
_foundoldspecdir = '%s/trainingdata/FoundModSpec'%(data_rootdir)
_foundpubspecdir = '%s/trainingdata/FoundationSpeccopy'%(data_rootdir)
_ps1oldspecdir = '%s/trainingdata/PS1Spec'%(data_rootdir)


def orig_training_data():

	snidlist_out = []
	for t in _training_dirs_orig:
		version = t.split('/')[-1]
		listfile = os.path.expandvars('%s/%s.LIST'%(t,version))
		if not os.path.exists(listfile):
			raise RuntimeError('listfile %s does not exist'%listfile)

		lcfiles = np.genfromtxt(listfile,unpack=True,dtype='str')
		for l in lcfiles:
			if '05ir' in l:
				import pdb; pdb.set_trace()
			
			try:
				sn = snana.SuperNova(os.path.expandvars('%s/%s'%(t,l)))
				if sn.SNID in snidlist_out: continue
				snidlist_out += [sn.SNID]
			except: print(os.path.expandvars('%s/%s'%(t,l)))
			
			if t in [_sdss_dir1,_sdss_dir2,_sdss_dir3,_sdss_dir4]:
				sdss_lcfile = glob.glob('%s/SDSS3_%06i.DAT'%(_jladir,sn.SNID))
				if not len(sdss_lcfile): continue

			#if isinstance(sn.SNID,str) and '2004dt' in sn.SNID: import pdb; pdb.set_trace()
			if 'REDSHIFT_HELIO' not in sn.__dict__.keys():
				zhel = vold(float(sn.RA.split()[0]),float(sn.DECL.split()[0]),float(sn.REDSHIFT_FINAL.split()[0]))
				sn.REDSHIFT_HELIO = '%.7f +- 0.000'%zhel
			#import pdb; pdb.set_trace()
			if 'hi': #try:
				#if t == _foundation_dir: sn.appendspec2snanafile('%s/%s'%(_outdir,l),_foundspecdir,verbose=True)
				#elif t == _ps1_dir: sn.appendspec2snanafile('%s/%s'%(_outdir,l),_ps1specdir,verbose=True,ps=True)
				sn.appendspec2snanafile('%s/%s'%(_outdir_orig,l),_jlaspecdir,verbose=False)
			else: pass
			#except: import pdb; pdb.set_trace()

def main():

	for t in _training_dirs:
		version = t.split('/')[-1]
		listfile = os.path.expandvars('%s/%s.LIST'%(t,version))
		if not os.path.exists(listfile):
			raise RuntimeError('listfile %s does not exist'%listfile)

		lcfiles = np.genfromtxt(listfile,unpack=True,dtype='str')
		for l in lcfiles:
			try: sn = snana.SuperNova(os.path.expandvars('%s/%s'%(t,l)))
			except: print(os.path.expandvars('%s/%s'%(t,l)))

			if t in [_sdss_dir1,_sdss_dir2,_sdss_dir3,_sdss_dir4]:
				sdss_lcfile = glob.glob('%s/SDSS3_%06i.DAT'%(_jladir,sn.SNID))
				if not len(sdss_lcfile): continue

			#if isinstance(sn.SNID,str) and '2004dt' in sn.SNID: import pdb; pdb.set_trace()
			if 'REDSHIFT_HELIO' not in sn.__dict__.keys():
				zhel = vold(float(sn.RA.split()[0]),float(sn.DECL.split()[0]),float(sn.REDSHIFT_FINAL.split()[0]))
				sn.REDSHIFT_HELIO = '%.7f +- 0.000'%zhel
			#import pdb; pdb.set_trace()
			if 'hi': #try:
				if t == _foundation_dir: sn.appendspec2snanafile('%s/%s'%(_outdir,l),_foundspecdir,verbose=True)
				elif t == _ps1_dir: sn.appendspec2snanafile('%s/%s'%(_outdir,l),_ps1specdir,verbose=True,ps=True)
				else: sn.appendspec2snanafile('%s/%s'%(_outdir,l),_jlaspecdir,verbose=False)
			else: pass
			#except: import pdb; pdb.set_trace()

			
def vnew(ra, dec, z):
	c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame = 'icrs')
	c_icrs = c_icrs.galactic
	b = c_icrs.b.degree
	l = c_icrs.l.degree

	b = np.radians(b)
	l = np.radians(l)
	l_0 = np.radians(264.14)
	b_0 = np.radians(48.26)


	v = (float(z)*3*10**5 + 371 * (np.sin(b) * np.sin(b_0) + np.cos(b) * np.cos(b_0) * np.cos(l-l_0)))/(3*10**5)

	return v

def vold(ra, dec, z):
	c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame = 'icrs')
	c_icrs = c_icrs.galactic
	b = c_icrs.b.degree
	l = c_icrs.l.degree

	b = np.radians(b)
	l = np.radians(l)
	l_0 = np.radians(264.14)
	b_0 = np.radians(48.26)

	v = (float(z)*(3*10**5) - 371 * (np.sin(b) * np.sin(b_0) - np.cos(b) * np.cos(b_0) * np.cos(l-l_0)))/(3*10**5)

	return v

def formatFoundSpec(outdir='/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/foundationspec_formatted'):
	from astropy.time import Time
	
	fileset = glob.glob('%s/*flm'%_foundoldspecdir)
	lc_name,found_name = np.loadtxt('%s/trainingdata/foundnames.txt'%data_rootdir,unpack=True,dtype=str)
	
	for f in fileset:

		snid = f.split('/')[-1].split('_')[0]
		try: snid = lc_name[found_name == snid][0]
		except: pass
		
		datestr = f.split('_')[1]
		t = Time('%s-%s-%s 00:00:00'%(datestr[:4],datestr[4:6],datestr[6:8]),format='iso')
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%t.mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True,skiprows=7)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

def formatFoundSpecNew(outdir='/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/foundationspec_formatted'):
	from astropy.time import Time
	
	fileset = glob.glob('%s/*flm'%_foundoldspecdir)
	lc_name,iau_name = np.loadtxt('%s/trainingdata/tns_names.list'%data_rootdir,unpack=True,dtype=str)
	#lc_name,found_name = np.loadtxt('%s/trainingdata/foundnames.txt'%data_rootdir,unpack=True,dtype=str)
	
	for f in fileset:

		snid = f.split('/')[-1].split('_')[0]
		try: snid = lc_name[lc_name == snid][0]
		except:
			snid = lc_name[iau_name == snid][0]
		
		datestr = f.split('_')[1]
		try: t = Time('%s-%s-%s 00:00:00'%(datestr[:4],datestr[4:6],datestr[6:8]),format='iso')
		except:
			import pdb; pdb.set_trace()
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%t.mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True,skiprows=7)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

		
def formatFoundPubSpec(outdir='/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/foundationspec_pub_formatted'):
	from astropy.time import Time
	
	fileset = glob.glob('%s/*/*flm'%_foundoldspecdir)
	lc_name,found_name = np.loadtxt('%s/trainingdata/foundnames.txt'%data_rootdir,unpack=True,dtype=str)
	
	for f in fileset:

		snid = f.split('/')[-1].split('_')[0]
		try: snid = lc_name[found_name == snid][0]
		except: pass
		
		datestr = f.split('_')[1]
		t = Time('%s-%s-%s 00:00:00'%(datestr[:4],datestr[4:6],datestr[6:8]),format='iso')
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%t.mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True,skiprows=7)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

		
def formatPS1Spec(outdir='/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/ps1spec_formatted'):
	from astropy.time import Time
	
	fileset1 = glob.glob('%s/ps1-??????-????????.*flm'%_ps1oldspecdir)
	fileset2 = glob.glob('%s/checto*.????m*m??????-????????.*.flm'%_ps1oldspecdir)
	fileset3 = glob.glob('%s/PS1-????-*-??????-*-????????.*flm'%_ps1oldspecdir)
	fileset4 = glob.glob('%s/???????-*-????????.*flm'%_ps1oldspecdir)
	fileset5 = glob.glob('%s/???????-????????.flm'%_ps1oldspecdir)
	fileset6 = glob.glob('%s/ps1-????-?-??????-????????*flm'%_ps1oldspecdir)
	fileset7 = glob.glob('%s/ps1-??????-????????-*flm'%_ps1oldspecdir)
	fileset8 = glob.glob('%s/??????-????????.flm'%_ps1oldspecdir)
	#import pdb; pdb.set_trace()
	#fileset1 = glob.glob('%s/*flm'%_ps1specdir)
	for f in fileset1:

		snid = '%06i'%int(f.split('-')[1])
		datestr = f.split('-')[2].split('.')[0]
		t = Time('%s-%s-%s 00:00:00'%(datestr[:4],datestr[4:6],datestr[6:8]),format='iso')
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%t.mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

	for f in fileset2:
		#print(f)
		snid = '%06i'%int(f.split('-')[0].split('m')[-1])
		datestr = f.split('-')[1].split('.')[0]
		t = Time('%s-%s-%s 00:00:00'%(datestr[:4],datestr[4:6],datestr[6:8]),format='iso')
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%t.mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

	for f in fileset3:
		#print(f)
		snid = '%06i'%int(f.split('-')[3])
		datestr = f.split('-')[5].split('.')[0]
		t = Time('%s-%s-%s 00:00:00'%(datestr[:4],datestr[4:6],datestr[6:8]),format='iso')
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%t.mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True)
			for w,f,fe in zip(wave,flux,fluxerr):
				if f == f and fe == fe:print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				if f == f: print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

	for f in fileset4:
		#print(f)
		snid = '%06i'%int(f.split('/')[-1][1:7])
		datestr = f.split('-')[2].split('.')[0]
		t = Time('%s-%s-%s 00:00:00'%(datestr[:4],datestr[4:6],datestr[6:8]),format='iso')
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%t.mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

	for f in fileset5:
		#print(f)
		snid = '%06i'%int(f.split('/')[-1][1:7])
		datestr = f.split('-')[1].split('.')[0]
		t = Time('%s-%s-%s 00:00:00'%(datestr[:4],datestr[4:6],datestr[6:8]),format='iso')
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%t.mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

	for f in fileset6:
		#print(f)
		snid = '%06i'%int(f.split('-')[3])
		datestr = f.split('-')[4] #.split('.')[0]
		t = Time('%s-%s-%s 00:00:00'%(datestr[:4],datestr[4:6],datestr[6:8]),format='iso')
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%t.mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

	for f in fileset7:

		snid = '%06i'%int(f.split('-')[1])
		datestr = f.split('-')[2].split('.')[0]
		t = Time('%s-%s-%s 00:00:00'%(datestr[:4],datestr[4:6],datestr[6:8]),format='iso')
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%t.mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

	for f in fileset8:
		#print(f)
		snid = '%06i'%int(f.split('/')[-1][1:6])
		datestr = f.split('-')[1].split('.')[0]
		t = Time('%s-%s-%s 00:00:00'%(datestr[:4],datestr[4:6],datestr[6:8]),format='iso')
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%t.mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()
		
	# now the annoying ones
	from txtobj import txtobj
	rst = txtobj('/Users/David/Dropbox/research/hostspec/snspec/rest14.lst')
	rstz = txtobj('/Users/David/Dropbox/research/hostspec/snspec/rest14.table')
	
	fileset9 = glob.glob('%s/L-??????_*flm'%_ps1specdir)
	for f in fileset9:
		print(f)
		snid = '%06i'%int(f.split('-')[1].split('_')[0])
		#import pdb; pdb.set_trace()
		try:
			mjd = float(rstz.MJD[rstz.ID == rst.PS1name[rst.PS1ID == 'PSc%s'%snid]][0])
		except: continue
		#datestr = f.split('-')[1].split('.')[0]
		#t = Time('%s-%s-%s 00:00:00'%(datestr[:4],datestr[4:6],datestr[6:8]),format='iso')
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

	yc = txtobj('/Users/David/Dropbox/research/hostspec/snspec/snspec_yc.txt')
	yc2 = txtobj('/Users/David/Dropbox/research/hostspec/snspec/snspec.txt')
	fileset10 = glob.glob('%s/PS1-??????_*'%_ps1specdir)
	for f in fileset10:
		print(f)
		snid = '%06i'%int(f.split('-')[1].split('_')[0])
		datestr = yc.specdate[(yc.id == 'ps1-%s'%snid) | (yc.id == 'PS1-%s'%snid)][0]

		t = Time('%s 00:00:00'%(datestr),format='iso')

		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
		print('@Date %i'%t.mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

	fileset12 = np.concatenate((glob.glob('%s/f?????.flm'%_ps1specdir),
								glob.glob('%s/f?????_*flm'%_ps1specdir),
								glob.glob('%s/e?????_*flm'%_ps1specdir),
								glob.glob('%s/e?????.flm'%_ps1specdir),
								glob.glob('%s/f??????-*flm'%_ps1specdir)))
	for f in fileset12:
		if '50203' in f: continue
		print(f)
		snid = '%06i'%int(f.split('/')[-1][1:].split('.')[0].split('_')[0].split('-')[0])
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')

		try:
			datestr = yc.specdate[(yc.id == 'ps1-%s'%snid) | (yc.id == 'PS1-%s'%snid)][0]
			t = Time('%s 00:00:00'%(datestr),format='iso')
			print('@Date %i'%t.mjd,file=fout)
		except:
			try:
				mjd = rstz.MJD[rstz.ID == rst.PS1name[rst.PS1ID == 'PSc%s'%snid]][0]
				if ',' in mjd: mjd = float(mjd.split(',')[0])
				else: mjd = float(mjd)
				print('@Date %i'%mjd,file=fout)
			except:
				datestr = yc2.specdate[(yc2.id == 'ps1-%s'%snid) | (yc2.id == 'PS1-%s'%snid)][0]
				t = Time('%s 00:00:00'%(datestr),format='iso')
				print('@Date %i'%t.mjd,file=fout)

				
		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

	fileset11 = glob.glob('%s/ps1-??????.dat'%_ps1specdir)
	for f in fileset11:
		print(f)
		snid = '%06i'%int(f.split('-')[1].split('.')[0])
		fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')

		try:
			datestr = yc.specdate[(yc.id == 'ps1-%s'%snid) | (yc.id == 'PS1-%s'%snid)][0]
			t = Time('%s 00:00:00'%(datestr),format='iso')
			print('@Date %i'%t.mjd,file=fout)
		except:
			mjd = rstz.MJD[rstz.ID == rst.PS1name[rst.PS1ID == 'PSc%s'%snid]][0]
			if ',' in mjd: mjd = float(mjd.split(',')[0])
			else: mjd = float(mjd)
			print('@Date %i'%mjd,file=fout)

		try:
			wave,flux,fluxerr = np.loadtxt(f,unpack=True)
			for w,f,fe in zip(wave,flux,fluxerr):
				print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
		except:
			wave,flux = np.loadtxt(f,unpack=True)
			for w,f in zip(wave,flux):
				print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

		fout.close()

		
		
	
	fileset11 = np.concatenate((glob.glob('%s/PS1-????.flm'%_ps1specdir),
								glob.glob('%s/PS1-?????_*.flm'%_ps1specdir),
								glob.glob('%s/PS1-?????-*.flm'%_ps1specdir)))
						  
	#for f in fileset11:
	#	print(f)
	#	snid = (f.split('-')[1].split('_')[-1])
	#	mjd = rstz.MJD[rstz.ID == rst.PS1name[rst.PS1ID == 'PSc%s'%snid]][0]
	#	fout = open('%s/spectrum-%s.list'%(outdir,snid),'w')
	#	print('@Date %i'%mjd,file=fout)

	#	try:
	#		wave,flux,fluxerr = np.loadtxt(f,unpack=True)
	#		for w,f,fe in zip(wave,flux,fluxerr):
	#			print('%.1f %8.5e %8.5e'%(w,f,fe),file=fout)
	#	except:
	#		wave,flux = np.loadtxt(f,unpack=True)
	#		for w,f in zip(wave,flux):
	#			print('%.1f %8.5e %8.5e'%(w,f,f*0.01),file=fout)

	#	fout.close()

		
	# 49
		
if __name__ == "__main__":
	#formatPS1Spec()
	#formatFoundSpecNew()
	#main()
	orig_training_data()
