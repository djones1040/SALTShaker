import numpy as np
import pylab as plt
import sncosmo
import argparse
from salt3.util import snana
from astropy.table import Table
import astropy.units as u
from salt3.util.synphot import synphot
from scipy.interpolate import interp1d
from sncosmo.constants import HC_ERG_AA
from salt3.initfiles import init_rootdir
from salt3.training.init_hsiao import synphotB
_SCALE_FACTOR = 1e-12

filtdict = {'b':'cspb','c':'cspv3014','d':'cspr','e':'cspi'}

def main(outfile,lcfile,salt3dir,
		 m0file='salt3_template_0.dat',
		 m1file='salt3_template_1.dat',
		 clfile='salt2_color_correction.dat',
		 cdfile='salt2_color_dispersion.dat',
		 errscalefile='salt2_lc_dispersion_scaling.dat',
		 lcrv00file='salt2_lc_relative_variance_0.dat',
		 lcrv11file='salt2_lc_relative_variance_1.dat',
		 lcrv01file='salt2_lc_relative_covariance_01.dat',
		 x0 = None, x1 = None, c = None, t0 = None,
		 fitx1=False,fitc=False,bandpassdict=None):
	bandpassdict = None
	
	plt.clf()

	fitparams_salt3 = []
	if not t0: fitparams_salt3 += ['t0']
	if not x0: fitparams_salt3 += ['x0']
	if not x1 and fitx1: fitparams_salt3 += ['x1']
	if not c and fitc: fitparams_salt3 += ['c']

	sn = snana.SuperNova(lcfile)
	sn.FLT = sn.FLT.astype('U20')

	if bandpassdict:
		bandlist = []
		for k in bandpassdict.keys():
			band = sncosmo.Bandpass(
				bandpassdict[k]['filtwave'],
				bandpassdict[k]['filttrans'],
				wave_unit=u.angstrom,name=k)
			sncosmo.register(band, k, force=True)
	else:
		for i in range(len(sn.FLT)):
			if sn.FLT[i] in filtdict.keys():
				sn.FLT[i] = filtdict[sn.FLT[i]]
			elif sn.FLT[i] in 'griz':
				sn.FLT[i] = 'sdss%s'%sn.FLT[i]
			elif sn.FLT[i].lower() == 'v':
				sn.FLT[i] = 'swope2::v'
			else:
				sn.FLT[i] = 'csp%s'%sn.FLT[i].lower()

	zpsys='AB'
	data = Table([sn.MJD,sn.FLT,sn.FLUXCAL,sn.FLUXCALERR,
				  np.array([27.5]*len(sn.MJD)),np.array([zpsys]*len(sn.MJD))],
				 names=['mjd','band','flux','fluxerr','zp','zpsys'],
				 meta={'t0':sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]})
	
	flux = sn.FLUXCAL
	salt2model = sncosmo.Model(source='salt2')
	hsiaomodel = sncosmo.Model(source='hsiao')
	salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,
								m1file=m1file,
								clfile=clfile,cdfile=cdfile,
								errscalefile=errscalefile,
								lcrv00file=lcrv00file,
								lcrv11file=lcrv11file,
								lcrv01file=lcrv01file)
	salt3model =  sncosmo.Model(salt3)
	salt3model.set(z=sn.REDSHIFT_HELIO[0:5])
	fitparams_salt2=['t0', 'x0', 'x1', 'c']
	salt2model.set(z=sn.REDSHIFT_HELIO[0:5])
	result_salt2, fitted_salt2_model = sncosmo.fit_lc(data, salt2model, fitparams_salt2)
	fitparams_hsiao = ['t0','amplitude']
	hsiaomodel.set(z=sn.REDSHIFT_HELIO[0:5])
	result_hsiao, fitted_hsiao_model = sncosmo.fit_lc(data, hsiaomodel, fitparams_hsiao)

	salt3model.set(z=sn.REDSHIFT_HELIO[0:5])
	if x0: salt3model.set(x0=x0)
	if t0: salt3model.set(t0=t0)
	if x1: salt3model.set(x1=x1)
	if c: salt3model.set(c=c)
	if len(fitparams_salt3):
		result_salt3, fitted_salt3_model = sncosmo.fit_lc(data, salt3model, fitparams_salt3)
	else:
		fitted_salt3_model = salt3model
	plotmjd = np.linspace(sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]-20,
						  sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]+55,100)
	
	fig = plt.figure(figsize=(15, 5))
	ax1 = fig.add_subplot(131)
	ax2 = fig.add_subplot(132)
	ax3 = fig.add_subplot(133)
	
	for flt,i,ax in zip(np.unique(sn.FLT),range(3),[ax1,ax2,ax3]):
		try:
			hsiaoflux = fitted_hsiao_model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
			salt2flux = fitted_salt2_model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
			salt3flux = fitted_salt3_model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')#*\
				#10**(-0.4*bandpassdict[flt]['zpoff'])*10**(0.4*bandpassdict[flt]['stdmag'])
		except:
			print('Warning : error for band %s'%flt)
			continue
		ax.plot(plotmjd,hsiaoflux,color='C0',
				label='Hsiao, $\chi^2_{red} = %.1f$'%(
					result_hsiao['chisq']/result_hsiao['ndof']))
		ax.plot(plotmjd,salt2flux,color='C1',
				label='SALT2, $\chi^2_{red} = %.1f$'%(
					result_salt2['chisq']/result_salt2['ndof']))
		if len(fitparams_salt3):
			ax.plot(plotmjd,salt3flux,color='C2',
					label='SALT3, $\chi^2_{red} = %.1f$'%(
						result_salt3['chisq']/result_salt3['ndof']))
		else:
			ax.plot(plotmjd,salt3flux*1e10,color='C2',
					label='SALT3')

		ax.errorbar(sn.MJD[sn.FLT == flt],sn.FLUXCAL[sn.FLT == flt],
					yerr=sn.FLUXCALERR[sn.FLT == flt],
					fmt='o',label=sn.SNID,color='k')
		ax.set_title(flt)
		ax.set_xlim([sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]-30,
					 sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]+55])
		ax.set_ylim([-np.max(sn.FLUXCAL)*1/20.,np.max(sn.FLUXCAL)*1.1])
#		import pdb; pdb.set_trace()
	ax1.legend()
	plt.savefig(outfile)
	plt.show()

def customfilt(outfile,lcfile,salt3dir,
			   m0file='salt3_template_0.dat',
			   m1file='salt3_template_1.dat',
			   clfile='salt2_color_correction.dat',
			   cdfile='salt2_color_dispersion.dat',
			   errscalefile='salt2_lc_dispersion_scaling.dat',
			   lcrv00file='salt2_lc_relative_variance_0.dat',
			   lcrv11file='salt2_lc_relative_variance_1.dat',
			   lcrv01file='salt2_lc_relative_covariance_01.dat',
			   Bfilt='Bessell90_B.dat',
			   flatnu='flatnu.dat',
			   x0 = None, x1 = None, c = None, t0 = None,
			   fitx1=False,fitc=False,bandpassdict=None, n_components=1):
	
	plt.clf()

	refWave,refFlux=np.loadtxt('%s/%s'%(init_rootdir,flatnu),unpack=True)
	Bfilt = '%s/%s'%(init_rootdir,Bfilt)
	
	fitparams_salt3 = []
	if not t0: fitparams_salt3 += ['t0']
	if not x0: fitparams_salt3 += ['x0']
	if not x1 and fitx1: fitparams_salt3 += ['x1']
	if not c and fitc: fitparams_salt3 += ['c']

	sn = snana.SuperNova(lcfile)
	sn.FLT = sn.FLT.astype('U20')

	if bandpassdict:
		bandlist = []
		for k in bandpassdict.keys():
			band = sncosmo.Bandpass(
				bandpassdict[k]['filtwave'],
				bandpassdict[k]['filttrans'],
				wave_unit=u.angstrom,name=k)
			sncosmo.register(band, k, force=True)
	else:
		for i in range(len(sn.FLT)):
			if sn.FLT[i] in filtdict.keys():
				sn.FLT[i] = filtdict[sn.FLT[i]]
			elif sn.FLT[i] in 'griz':
				sn.FLT[i] = 'sdss%s'%sn.FLT[i]
			elif sn.FLT[i].lower() == 'v':
				sn.FLT[i] = 'swope2::v'
			else:
				sn.FLT[i] = 'csp%s'%sn.FLT[i].lower()

	zpsys='AB'
	data = Table([sn.MJD,sn.FLT,sn.FLUXCAL,sn.FLUXCALERR,
				  np.array([27.5]*len(sn.MJD)),np.array([zpsys]*len(sn.MJD))],
				 names=['mjd','band','flux','fluxerr','zp','zpsys'],
				 meta={'t0':sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]})
	
	flux = sn.FLUXCAL
	salt2model = sncosmo.Model(source='salt2')
	hsiaomodel = sncosmo.Model(source='hsiao')
	salt3phase,salt3wave,salt3flux = np.genfromtxt('%s/%s'%(salt3dir,m0file),unpack=True)
	#salt3flux *= 10**(-0.4*(-19.36+(synphotB(refWave,refFlux,0,0,Bfilt)-synphotB(salt3wave[salt3phase==0],salt3flux[salt3phase==0],0,0,Bfilt))))
	salt3m1phase,salt3m1wave,salt3m1flux = np.genfromtxt('%s/%s'%(salt3dir,m1file),unpack=True)
	#salt3m1flux *= 10**(-0.4*(-19.36+(synphotB(refWave,refFlux,0,0,Bfilt)-synphotB(salt3m1wave[salt3m1phase==0],salt3m1flux[salt3m1phase==0],0,0,Bfilt))))

	salt3flux = salt3flux.reshape([len(np.unique(salt3phase)),len(np.unique(salt3wave))])#*10**(0.4*27.5)
	salt3m1flux = salt3m1flux.reshape([len(np.unique(salt3phase)),len(np.unique(salt3wave))])#*10**(0.4*27.5)
	salt3phase = np.unique(salt3phase)*(1+float(sn.REDSHIFT_HELIO[0:5]))
	salt3wave = np.unique(salt3wave)*(1+float(sn.REDSHIFT_HELIO[0:5]))

	
	if n_components == 1: salt3flux = x0*salt3flux
	elif n_components == 2: salt3flux = x0*(salt3flux + x1*salt3m1flux)
	salt3flux *= _SCALE_FACTOR
	#import pdb; pdb.set_trace()
	
	salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,
								m1file=m1file,
								clfile=clfile,cdfile=cdfile,
								errscalefile=errscalefile,
								lcrv00file=lcrv00file,
								lcrv11file=lcrv11file,
								lcrv01file=lcrv01file)
	salt3model =  sncosmo.Model(salt3)
	salt3model.set(z=sn.REDSHIFT_HELIO[0:5])
	fitparams_salt2=['t0', 'x0', 'x1', 'c']
	salt2model.set(z=sn.REDSHIFT_HELIO[0:5])
	#result_salt2, fitted_salt2_model = sncosmo.fit_lc(data, salt2model, fitparams_salt2)
	#fitparams_hsiao = ['t0','amplitude']
	#hsiaomodel.set(z=sn.REDSHIFT_HELIO[0:5])
	#result_hsiao, fitted_hsiao_model = sncosmo.fit_lc(data, hsiaomodel, fitparams_hsiao)

	salt3model.set(z=sn.REDSHIFT_HELIO[0:5])
	if x0: salt3model.set(x0=x0)
	if t0: salt3model.set(t0=t0)
	if x1: salt3model.set(x1=x1)
	if c: salt3model.set(c=c)
	if len(fitparams_salt3):
		result_salt3, fitted_salt3_model = sncosmo.fit_lc(data, salt3model, fitparams_salt3)
	else:
		fitted_salt3_model = salt3model
	plotmjd = np.linspace(sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)][0]-20,
						  sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)][0]+55,200)
	
	fig = plt.figure(figsize=(15, 5))
	ax1 = fig.add_subplot(131)
	ax2 = fig.add_subplot(132)
	ax3 = fig.add_subplot(133)

	int1d = interp1d(salt3phase,salt3flux,axis=0,fill_value='extrapolate')
	for flt,i,ax in zip(np.unique(sn.FLT),range(3),[ax1,ax2,ax3]):
		try:
			hsiaoflux = fitted_hsiao_model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
			salt2flux = fitted_salt2_model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
			
			ax.plot(plotmjd,hsiaoflux,color='C0',
					label='Hsiao, $\chi^2_{red} = %.1f$'%(
						result_hsiao['chisq']/result_hsiao['ndof']))
			ax.plot(plotmjd,salt2flux,color='C1',
					label='SALT2, $\chi^2_{red} = %.1f$'%(
						result_salt2['chisq']/result_salt2['ndof']))
		except:
			print('Warning : error for band %s'%flt)
			#continue

			#salt3flux = fitted_salt3_model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')#*\

		phase=plotmjd-t0
		salt3fluxnew = int1d(phase)
		#phase=(photdata['tobs']+tpkoff)/1+z
		filtwave = bandpassdict[flt]['filtwave']
		filttrans = bandpassdict[flt]['filttrans']

		g = (salt3wave >= filtwave[0]) & (salt3wave <= filtwave[-1])  # overlap range

		pbspl = np.interp(salt3wave[g],filtwave,filttrans)
		pbspl *= salt3wave[g]
		denom = np.trapz(pbspl,salt3wave[g])
		salt3synflux=np.trapz(pbspl[np.newaxis,:]*salt3fluxnew[:,g]/HC_ERG_AA,salt3wave[g],axis=1)/denom
		salt3synflux *= 10**(-0.4*bandpassdict[flt]['zpoff'])*10**(0.4*bandpassdict[flt]['stdmag'])*10**(0.4*27.5)
		#int1d = interp1d(salt3phase,salt3synflux,axis=0,fill_value='extrapolate')
	
		#import pdb; pdb.set_trace()
		#salt3synflux = int1d(plotmjd-t0)
		
		if len(fitparams_salt3):
			ax.plot(plotmjd,salt3synflux,color='C2',
					label='SALT3, $\chi^2_{red} = %.1f$'%(
						result_salt3['chisq']/result_salt3['ndof']))
		else:
			ax.plot(plotmjd,salt3synflux,color='C2',
					label='SALT3, x1=%.2f, z=%.3f'%(x1,float(sn.REDSHIFT_HELIO[0:5])))
			
		ax.errorbar(sn.MJD[sn.FLT == flt],sn.FLUXCAL[sn.FLT == flt],
					yerr=sn.FLUXCALERR[sn.FLT == flt],
					fmt='o',label=sn.SNID,color='k')
		ax.set_title(flt)
		try:
			ax.set_xlim([sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]-30,
						 sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]+55])
		except:
			import pdb; pdb.set_trace()
		ax.set_ylim([-np.max(sn.FLUXCAL)*1/20.,np.max(sn.FLUXCAL)*1.1])

		#if flt == 'c': import pdb; pdb.set_trace()
	ax1.legend()
	plt.savefig(outfile)
	plt.show()

	
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Plot lightcurves from SALT3 model against SALT2 model, Hsiao model, and data')
	parser.add_argument('lcfile',type=str,help='File with supernova fit parameters')
	parser.add_argument('salt3dir',type=str,help='File with supernova fit parameters')
	parser.add_argument('outfile',type=str,nargs='?',default=None,help='File with supernova fit parameters')
	parser=parser.parse_args()
	args=vars(parser)
	if parser.outfile is None:
		sn = snana.SuperNova(parser.lcfile)
		args['outfile']='lccomp_%s.png'%sn.SNID
	main(**args)
