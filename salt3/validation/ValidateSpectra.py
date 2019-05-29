import sys,os,sncosmo
import numpy as np
from salt3.util import readutils
from matplotlib import pyplot as plt
from scipy.special import factorial
from matplotlib.backends.backend_pdf import PdfPages


def compareSpectra(speclist,salt3dir,outdir=None,parfile='salt3_parameters.dat',
				   m0file='salt3_template_0.dat',
				   m1file='salt3_template_1.dat',
				   clfile='salt3_color_correction.dat',
				   cdfile='salt3_color_dispersion.dat',
				   errscalefile='salt3_lc_dispersion_scaling.dat',
				   lcrv00file='salt3_lc_relative_variance_0.dat',
				   lcrv11file='salt3_lc_relative_variance_1.dat',
				   lcrv01file='salt3_lc_relative_covariance_01.dat',
				   ax=None):

	plt.close('all')
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
	pdf_pages = PdfPages('%s/speccomp.pdf'%outdir)

	axcount = 0
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

			if not axcount % 3:
				fig = plt.figure()
			ax = plt.subplot(3,1,axcount % 3 + 1)
			
			#ax.clf()
			ax.plot(wave,modelFlux,'r-',label='Model spectrum')
			ax.plot(wave,specdata[k]['flux'],'b-',label='Spectral data')
			ax.plot(wave,unncalledModel,'g-',label='Model spectrum (no calibration)')
			ax.set_xlim(wave.min(),wave.max())
			ax.set_ylim(0,max(modelFlux.max(),specdata[k]['flux'].max())*1.25)
			ax.set_xlabel('Wavelength $\AA$')
			ax.set_ylabel('Flux')
			ax.legend()
			#plt.savefig('{}/speccomp_{}_{}.png'.format(outdir,sn,k),dpi=288)
			axcount += 1

			if not axcount % 3:
				pdf_pages.savefig()

	pdf_pages.savefig()
	pdf_pages.close()
			
if __name__ == "__main__":
	usagestring = """ Compares a SALT3 model to spectra

	usage: python ValidateSpectra.py <speclist> <salt3dir>

	Dependencies: sncosmo?
	"""
	compareSpectra(*sys.argv[1:])
