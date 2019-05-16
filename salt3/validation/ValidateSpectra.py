import sys,os,sncosmo
import numpy as np
from salt3.util import readutils
from matplotlib import pyplot as plt
from scipy.special import factorial

def compareSpectra(speclist,salt3dir,outdir=None,parfile='salt3_parameters.dat',
		 m0file='salt3_template_0.dat',
		 m1file='salt3_template_1.dat',
		 clfile='salt2_color_correction.dat',
		 cdfile='salt2_color_dispersion.dat',
		 errscalefile='salt2_lc_dispersion_scaling.dat',
		 lcrv00file='salt2_lc_relative_variance_0.dat',
		 lcrv11file='salt2_lc_relative_variance_1.dat',
		 lcrv01file='salt2_lc_relative_covariance_01.dat'):

	datadict=readutils.rdAllData(speclist,False,None,lambda x: None,speclist)
	salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,
							m1file=m1file,
							clfile=clfile,cdfile=cdfile,
							errscalefile=errscalefile,
							lcrv00file=lcrv00file,
							lcrv11file=lcrv11file,
							lcrv01file=lcrv01file)
	model=sncosmo.Model(source=salt3)
	parlist,pars=np.loadtxt(os.path.join(salt3dir,parfile),skiprows=1,unpack=True,dtype=[('a','U40'),('b',float)])
	if outdir is None: outdir=salt3dir	
	for sn in datadict.keys():
		specdata=datadict[sn]['specdata']
		snPars={'z':datadict[sn]['zHelio']}
		try:
			for par in ['x0','x1','c','t0']:
				if par=='t0':
					snPars['t0']=pars[parlist=='tpkoff_{}'.format(sn)][0]
				else:
					snPars[par]=pars[parlist== '{}_{}'.format(par,sn)][0]
		except:
			print('SN {} is not in parameters, skipping'.format(sn))
			continue
		model.update(snPars)
		for k in specdata.keys():
			
			coeffs=pars[parlist=='specrecal_{}_{}'.format(sn,k)]
			coeffs/=factorial(np.arange(len(coeffs)))
			wave=specdata[k]['wavelength']
			print(coeffs)
			modelFlux = model.flux(specdata[k]['tobs'],wave)*np.exp(np.poly1d(coeffs)((wave-np.mean(wave))/2500))
			unncalledModel=model.flux(specdata[k]['tobs'],wave)
			#import pdb; pdb.set_trace()
			plt.clf()
			plt.plot(wave,modelFlux,'r-','Model spectrum')
			plt.plot(wave,specdata[k]['flux'],'b-','Spectral data')
			plt.plot(wave,unncalledModel,'g-','Model spectrum (no calibration)')
			plt.xlim(wave.min(),wave.max())
			plt.ylim(0,max(modelFlux.max(),specdata[k]['flux'].max())*1.25)
			plt.xlabel('Wavelength $\AA$')
			plt.ylabel('Flux')
			plt.savefig('{}/speccomp_{}_{}.png'.format(outdir,sn,k),dpi=288)

if __name__ == "__main__":
	usagestring = """ Compares a SALT3 model to spectra

	usage: python ValidateSpectra.py <speclist> <salt3dir>

	Dependencies: sncosmo?
	"""
	compareSpectra(*sys.argv[1:])