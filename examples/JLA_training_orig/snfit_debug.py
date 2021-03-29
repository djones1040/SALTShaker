#!/usr/bin/env python

import numpy as np
import pylab as plt
import sncosmo
from astropy.table import Table
import os
import snana
import astropy.units as u
from saltshaker.initfiles import init_rootdir as salt2dir
print(salt2dir)

def main():

	sn = snana.SuperNova('JLA_training_orig/JLA2014_SNLS_04D4in_nospec.dat')	
	data = Table(rows=None,names=['mjd','band','flux','fluxerr','zp','zpsys'],
				 dtype=('f8','S1','f8','f8','f8','U5'),
				 meta={'t0':sn.SEARCH_PEAKMJD})

	for filtfile,filtname in zip(['$SNDATA_ROOT/filters/PS1/Pantheon/SNLS3-Megacam/effMEGACAM-g.dat',
								  '$SNDATA_ROOT/filters/PS1/Pantheon/SNLS3-Megacam/effMEGACAM-r.dat',
								  '$SNDATA_ROOT/filters/PS1/Pantheon/SNLS3-Megacam/effMEGACAM-i.dat',
								  '$SNDATA_ROOT/filters/PS1/Pantheon/SNLS3-Megacam/effMEGACAM-z.dat'],
								  'griz'):
		filtwave,filttrans = np.loadtxt(
			os.path.expandvars(
				filtfile),
			unpack=True)
		band = sncosmo.Bandpass(
			filtwave,
			filttrans,
			wave_unit=u.angstrom,name=filtname)
		sncosmo.register(band, force=True)

	for m,flt,flx,flxe in zip(sn.MJD,sn.FLT,sn.FLUXCAL,sn.FLUXCALERR):
		#if flt == 'g':
		data.add_row((m,flt,flx,flxe,
					  27.5-0.0067+0.0076605430,'ab'))
	salt2source = sncosmo.SALT2Source(modeldir=salt2dir)
	dust = sncosmo.F99Dust()
	salt2model = sncosmo.Model(salt2source,effects=[dust],effect_names=['mw'],effect_frames=['obs'])
	#salt2model.set(z=float(sn.REDSHIFT_HELIO.split()[0]),mwebv=sn.MWEBV.split()[0])
	salt2model.set(z=float(sn.REDSHIFT_HELIO.split()[0]),mwebv=sn.MWEBV.split()[0],
				   t0=5.32785123e+04,x0=1.17451904e-05,x1=1.55527370e+00,c=-6.67126317e-02)
	flux,cov = salt2model.bandfluxcov(
		'g', 5.32785123e+04+np.array([-16.487, -12.497,   6.513,   8.493,  14.453,  38.413]), zp=27.5, zpsys='ab')
	
	
	#fitparams = ['t0', 'x0', 'x1', 'c']
	#result, fitted_model = sncosmo.fit_lc(
	#	data, salt2model, fitparams,
	#	bounds={'t0':(sn.SEARCH_PEAKMJD-10, sn.SEARCH_PEAKMJD+10),
	#			'z':(0.0,0.7),'x1':(-3,3),'c':(-0.3,0.3)})
	import pdb; pdb.set_trace()
	
if __name__ == "__main__":
	main()
