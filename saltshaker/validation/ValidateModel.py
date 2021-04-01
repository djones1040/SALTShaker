#!/usr/bin/env python

import numpy as np
import pylab as plt
import sncosmo
from saltshaker.util import snana
from astropy.table import Table
import argparse

def main(outfile,salt3dir,
		 m0file='salt3_template_0.dat',
		 m1file='salt3_template_1.dat',
		 clfile='salt3_color_correction.dat',
		 cdfile='salt3_color_dispersion.dat',
		 errscalefile='salt3_lc_dispersion_scaling.dat',
		 lcrv00file='salt3_lc_relative_variance_0.dat',
		 lcrv11file='salt3_lc_relative_variance_1.dat',
		 lcrv01file='salt3_lc_relative_covariance_01.dat'):
	plt.clf()
	
	hsiao_model = sncosmo.Model(source='hsiao')
	hsiao_model.set(amplitude=1)
	salt2_model = sncosmo.Model(source='salt2')
	salt2_model.set_source_peakmag(
		hsiao_model.source_peakmag('bessellv', 'vega'),'bessellv', 'vega')

	salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,m1file=m0file,
								clfile=clfile,cdfile=cdfile,
								errscalefile=errscalefile,
								lcrv00file=lcrv00file,
								lcrv11file=lcrv11file,
								lcrv01file=lcrv01file)
	salt3_model =  sncosmo.Model(salt3)
	salt3_model.set_source_peakmag(
		hsiao_model.source_peakmag('bessellv', 'vega'),'bessellv', 'vega')
	p=np.linspace(-10,10,100)

	wave_array=np.linspace(3000.0,8000.0,1000.0)
	flux_salt2=salt2_model.flux(p, wave_array)
	flux_salt3=salt3_model.flux(p, wave_array)
	flux_hsiao=hsiao_model.flux(p, wave_array)
	salt3_hsiao_comp=np.sqrt((flux_salt3-flux_hsiao)**2)
	salt3_salt2_comp=np.sqrt((flux_salt3-flux_salt2)**2)
	max_salt3_hsiao_comp=np.max(salt3_hsiao_comp)
	max_salt3_salt2_comp=np.max(salt3_salt2_comp)
	fig = plt.figure(figsize=(8, 5))

	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

	ax1.set_ylabel('Phase (days)')
	#ax1.set_xlabel('Wavelength ($\AA$)')
	ax1.set_xticklabels([])
	ax1.set_title('SALT3-Hsiao')
	i = ax1.imshow(salt3_hsiao_comp/max_salt3_hsiao_comp,
				   interpolation='none',aspect='auto',cmap = plt.cm.get_cmap("viridis"),
				   origin='lower',extent=[min(wave_array),max(wave_array), min(p),max(p)])
	clb = plt.colorbar(i,ax=ax1)
	clb.ax.set_title('% 100')
	ax2.set_ylabel('Phase (days)')
	ax2.set_xlabel('Wavelength ($\AA$)')
	ax2.set_title('SALT3-SALT2')
	i = ax2.imshow(salt3_salt2_comp/max_salt3_salt2_comp,
				   interpolation='none',aspect='auto',cmap = plt.cm.get_cmap("viridis")
				   ,origin='lower',extent=[min(wave_array),max(wave_array), min(p),max(p)])
	clb = plt.colorbar(i,ax=ax2)
	clb.ax.set_title('% 100')
	plt.savefig(outfile)
	#plt.show()

def m0m1_chi2(outfile,salt3dir,
			  m0file='salt3_template_0.dat',
			  m1file='salt3_template_1.dat',
			  clfile='salt3_color_correction.dat',
			  cdfile='salt3_color_dispersion.dat',
			  errscalefile='salt3_lc_dispersion_scaling.dat',
			  lcrv00file='salt3_lc_relative_variance_0.dat',
			  lcrv11file='salt3_lc_relative_variance_1.dat',
			  lcrv01file='salt3_lc_relative_covariance_01.dat'):
	plt.clf()

	hsiao_model = sncosmo.Model(source='hsiao')
	hsiao_model.set(amplitude=1)

	salt2_model = sncosmo.Model(source='salt2')
	salt2_model.set_source_peakmag(
		hsiao_model.source_peakmag('bessellv', 'vega'),'bessellv', 'vega')

	salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,m1file=m0file,
								clfile=clfile,cdfile=cdfile,
								errscalefile=errscalefile,
								lcrv00file=lcrv00file,
								lcrv11file=lcrv11file,
								lcrv01file=lcrv01file)
	salt3_model =  sncosmo.Model(salt3)
	salt3_model.set_source_peakmag(
		hsiao_model.source_peakmag('bessellv', 'vega'),'bessellv', 'vega')
	p=np.linspace(-10,10,100)

	wave_array=np.linspace(3000.0,8000.0,1000.0)
	flux_salt2=salt2_model._source._model['M0'](p, wave_array)
	fluxerr_salt2 = salt2_model._source._model['LCRV00'](p,wave_array)*1e-12
	flux_salt3=salt3_model._source._model['M0'](p, wave_array)
	fluxerr_salt3 = salt3_model._source._model['LCRV00'](p,wave_array)*1e-12

	flux_salt2_m1=salt2_model._source._model['M1'](p, wave_array)
	fluxerr_salt2_m1 = salt2_model._source._model['LCRV01'](p,wave_array)*1e-12
	flux_salt3_m1=salt3_model._source._model['M1'](p, wave_array)
	fluxerr_salt3_m1 = salt3_model._source._model['LCRV01'](p,wave_array)*1e-12

	
	flux_hsiao=hsiao_model.flux(p, wave_array)
	salt3_hsiao_comp=np.sqrt((flux_salt3-flux_hsiao)**2)
	salt3_salt2_comp=np.sqrt((flux_salt3-flux_salt2)**2)
	max_salt3_hsiao_comp=np.max(salt3_hsiao_comp)
	max_salt3_salt2_comp=np.max(salt3_salt2_comp)
	fig = plt.figure(figsize=(8, 5))
	
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

	ax1.set_ylabel('Phase (days)')
	ax1.set_xlabel('Wavelength ($\AA$)')
	ax1.text(0.5,0.9,'M$_0$ (SALT3-SALT2) $\chi^2$',ha='center',va='center',color='1.0',transform=ax1.transAxes,bbox={'facecolor':'0.0','edgecolor':'0.0','alpha':0.2})
	i = ax1.imshow((flux_salt3 - flux_salt2)**2./(fluxerr_salt2**2.+fluxerr_salt3**2.+(0.02*flux_salt3)**2. + (0.02*flux_salt2)**2.),
				   interpolation='none',aspect='auto',cmap = plt.cm.get_cmap("viridis")
				   ,origin='lower',extent=[min(wave_array),max(wave_array), min(p),max(p)],vmin=0,vmax=100)
	clb = plt.colorbar(i,ax=ax1)
	clb.ax.set_ylabel('$\chi^2$ (2% error floor)')


	ax2.set_ylabel('Phase (days)')
	ax2.set_xlabel('Wavelength ($\AA$)')
	ax2.text(0.5,0.9,'M$_1$ (SALT3-SALT2) $\chi^2$',ha='center',va='center',transform=ax2.transAxes,bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.2})
	i = ax2.imshow((flux_salt3_m1 - flux_salt2_m1)**2./(fluxerr_salt2_m1**2.+fluxerr_salt3_m1**2.+(0.02*flux_salt3_m1)**2. + (0.02*flux_salt2_m1)**2.),
				   interpolation='none',aspect='auto',cmap = plt.cm.get_cmap("viridis")
				   ,origin='lower',extent=[min(wave_array),max(wave_array), min(p),max(p)],vmin=0,vmax=5000)

	clb = plt.colorbar(i,ax=ax2)
	clb.ax.set_ylabel('$\chi^2$ (2% error floor)')
	plt.savefig(outfile)
	#plt.show()

	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Plot differences between SALT3 model and the SALT2/Hsiao models')
	parser.add_argument('salt3dir',type=str,help='File with supernova fit parameters')
	parser.add_argument('outfile',type=str,nargs='?',default=None,help='File with supernova fit parameters')
	parser=parser.parse_args()
	args=vars(parser)
	if parser.outfile is None:
		args['outfile']='modelcomp.png'
	m0m1_chi2(**args)
	#main(**args)
