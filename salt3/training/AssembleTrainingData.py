#!/usr/bin/env python

import numpy as np
import os
import glob
from salt3.util import jla, snana
from salt3.data import data_rootdir

#_cfa_early_dir = "$SNDATA_ROOT/lcmerge/
_lowz_dir = "$SNDATA_ROOT/lcmerge/Pantheon_LOWZ_TEXT"
#_foundation_dir = "$SNDATA_ROOT/lcmerge/Pantheon_LOWZ_TEXT"
#_lowz_dir = "$SNDATA_ROOT/lcmerge/02-DATA_PHOTOMETRY/DES-SN3YR_LOWZ"
#_des_dir = "$SNDATA_ROOT/lcmerge/02-DATA_PHOTOMETRY/DES-SN3YR_DES"
_snls_dir = "$SNDATA_ROOT/lcmerge/JLA2014_SNLS"
_sdss_dir1 = "$SNDATA_ROOT/lcmerge/SMPv8+BOSS/SMPv8+BOSS_2004"
_sdss_dir2 = "$SNDATA_ROOT/lcmerge/SMPv8+BOSS/SMPv8+BOSS_2005"
_sdss_dir3 = "$SNDATA_ROOT/lcmerge/SMPv8+BOSS/SMPv8+BOSS_2006"
_sdss_dir4 = "$SNDATA_ROOT/lcmerge/SMPv8+BOSS/SMPv8+BOSS_2007"

_training_dirs = [_lowz_dir,_snls_dir,
				  _sdss_dir1,_sdss_dir2,_sdss_dir3,_sdss_dir4]#,_des_dir]
#_outdir = '%s/trainingdata/snana'%(data_rootdir)
_jladir = '%s/trainingdata/jla'%(data_rootdir)
_outdir = '%s/trainingdata/Pantheon_noPS1'%(data_rootdir)

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
				
			sn.appendspec2snanafile('%s/%s'%(_outdir,l),_jladir)
			#import pdb; pdb.set_trace()

if __name__ == "__main__":
	main()
