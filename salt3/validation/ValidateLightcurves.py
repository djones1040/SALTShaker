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
from sncosmo.salt2utils import SALT2ColorLaw
_SCALE_FACTOR = 1e-12

#filtdict = {'b':'cspb','c':'cspv3014','d':'cspr','e':'cspi'}
filtdict = {'J':'J','H':'H',
			'Y':'Y',
			'a':'Jrc2',
			'b':'Jrc1',
			'c':'Ydw',
			'd':'Jdw',
			'e':'Hdw',
			'f':'J2m',
			'g':'H2m',
			'l':'Ks2m',
			'm':'JANDI',
			'n':'HANDI',
			'o':'F125W',
			'p':'F160W'}

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
			   clfile='salt3_color_correction.dat',
			   cdfile='salt3_color_dispersion.dat',
			   errscalefile='salt3_lc_dispersion_scaling.dat',
			   lcrv00file='salt3_lc_relative_variance_0.dat',
			   lcrv11file='salt3_lc_relative_variance_1.dat',
			   lcrv01file='salt3_lc_relative_covariance_01.dat',
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
	if 'PEAKMJD' in sn.__dict__.keys():
		data = Table(rows=None,names=['mjd','band','flux','fluxerr','zp','zpsys'],
					 dtype=('f8','S1','f8','f8','f8','U5'),
					 meta={'t0':sn.PEAKMJD})
	else:
		data = Table(rows=None,names=['mjd','band','flux','fluxerr','zp','zpsys'],
					 dtype=('f8','S1','f8','f8','f8','U5'),
					 meta={'t0':sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]})

	for m,flt,flx,flxe in zip(sn.MJD,sn.FLT,sn.FLUXCAL,sn.FLUXCALERR):
		data.add_row((m,flt,flx,flxe,
					  27.5,'bd17'))

	#data = Table([sn.MJD,sn.FLT,sn.FLUXCAL,sn.FLUXCALERR,
	#			  np.array([27.5]*len(sn.MJD)),np.array(['bd17']*len(sn.MJD))],
	#			 names=['mjd','band','flux','fluxerr','zp','zpsys'],
	#			 meta={'t0':sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]})
		
	flux = sn.FLUXCAL
	from salt3.initfiles import init_rootdir as salt2dir
	salt2source = sncosmo.SALT2Source(modeldir=salt2dir)
	salt2model = sncosmo.Model(salt2source)
	hsiaomodel = sncosmo.Model(source='hsiao')
	salt3phase,salt3wave,salt3flux = np.genfromtxt('%s/%s'%(salt3dir,m0file),unpack=True)
	salt3m1phase,salt3m1wave,salt3m1flux = np.genfromtxt('%s/%s'%(salt3dir,m1file),unpack=True)
	#salt2phase,salt2wave,salt2flux = np.genfromtxt('/usr/local/SNDATA_ROOT/models/SALT2/SALT2.JLA-B14/salt2_template_0.dat',unpack=True)
	#salt2m1phase,salt2m1wave,salt2m1flux = np.genfromtxt('/usr/local/SNDATA_ROOT/models/SALT2/SALT2.JLA-B14/salt2_template_1.dat',unpack=True)
	salt2phase,salt2wave,salt2flux = np.genfromtxt('{}/salt2_template_0.dat'.format(salt2dir),unpack=True)
	salt2m1phase,salt2m1wave,salt2m1flux = np.genfromtxt('{}/salt2_template_1.dat'.format(salt2dir),unpack=True)


	# color laws
	with open('%s/%s'%(salt3dir,clfile)) as fin:
		lines = fin.readlines()
	if len(lines):
		for i in range(len(lines)):
			lines[i] = lines[i].replace('\n','')
		colorlaw_salt3_coeffs = np.array(lines[1:5]).astype('float')
		salt3_colormin = float(lines[6].split()[1])
		salt3_colormax = float(lines[7].split()[1])

		salt3colorlaw = SALT2ColorLaw([2800,7000],colorlaw_salt3_coeffs)

	salt2colorlaw = SALT2ColorLaw([2800,7000], [-0.504294,0.787691,-0.461715,0.0815619])

	
	
	salt3flux = salt3flux.reshape([len(np.unique(salt3phase)),len(np.unique(salt3wave))])#*10**(0.4*27.5)
	salt3m1flux = salt3m1flux.reshape([len(np.unique(salt3phase)),len(np.unique(salt3wave))])#*10**(0.4*27.5)
	salt3phase = np.unique(salt3phase)*(1+float(sn.REDSHIFT_HELIO[0:5]))
	salt3wave = np.unique(salt3wave)*(1+float(sn.REDSHIFT_HELIO[0:5]))

	salt2m0flux = salt2flux.reshape([len(np.unique(salt2phase)),len(np.unique(salt2wave))])
	salt2flux = salt2flux.reshape([len(np.unique(salt2phase)),len(np.unique(salt2wave))])
	salt2m1flux = salt2m1flux.reshape([len(np.unique(salt2phase)),len(np.unique(salt2wave))])
	#int1d_salt2m1 = interp1d(salt2m1phase,salt2m1flux,axis=0,fill_value='extrapolate')
	#salt2m1flux = int1d_salt2m1(salt2phase)
	if 'SIM_SALT2x0' in sn.__dict__.keys():
		salt2phase = np.unique(salt2phase)*(1+float(sn.SIM_REDSHIFT_HELIO))
		salt2wave = np.unique(salt2wave)*(1+float(sn.SIM_REDSHIFT_HELIO))

	if n_components == 1: salt3flux = x0*salt3flux
	elif n_components == 2: salt3flux = x0*(salt3flux + x1*salt3m1flux)
	if c:
		salt3flux *= 10. ** (-0.4 * salt3colorlaw(salt3wave/(1+float(sn.SIM_REDSHIFT_HELIO))) * c)
	salt3flux *= _SCALE_FACTOR

	if 'SIM_SALT2x0' in sn.__dict__.keys():
		salt2flux = sn.SIM_SALT2x0*(salt2m0flux*_SCALE_FACTOR + (sn.SIM_SALT2x1)*salt2m1flux*_SCALE_FACTOR) * \
					10. ** (-0.4 * salt2colorlaw(salt2wave/(1+float(sn.SIM_REDSHIFT_HELIO))) * float(sn.SIM_SALT2c))
		
	salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,
								m1file=m1file,
								clfile=clfile,cdfile=cdfile,
								errscalefile=errscalefile,
								lcrv00file=lcrv00file,
								lcrv11file=lcrv11file,
								lcrv01file=lcrv01file)
	
	if 'PEAKMJD' in sn.__dict__.keys():
		plotmjd = np.linspace(sn.PEAKMJD-20,
							  sn.PEAKMJD+55,200)
	else:
		print('BLAH!')
		plotmjd = np.linspace(sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)][0]-20,
							  sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)][0]+55,200)
	
	fig = plt.figure(figsize=(15, 5))
	ax1 = fig.add_subplot(131)
	ax2 = fig.add_subplot(132)
	ax3 = fig.add_subplot(133)

	int1d = interp1d(salt3phase,salt3flux,axis=0,fill_value='extrapolate')
	if 'SIM_SALT2x0' in sn.__dict__.keys():
		int1d_salt2 = interp1d(salt2phase,salt2flux,axis=0,fill_value='extrapolate')
	for flt,i,ax in zip(np.unique(sn.FLT),range(3),[ax1,ax2,ax3]):
		#if 'Y' not in filtdict[flt] and 'J' not in filtdict[flt]:
		#	continue
		#print('HACK')
		phase=plotmjd-sn.PEAKMJD #t0
		salt3fluxnew = int1d(phase)
		if 'SIM_SALT2x0' in sn.__dict__.keys():
			phase_salt2 = plotmjd-float(sn.SIM_PEAKMJD.split()[0])
			salt2fluxnew = int1d_salt2(phase_salt2)

		#phase=(photdata['tobs']+tpkoff)/1+z
		filtwave = bandpassdict[flt]['filtwave']
		filttrans = bandpassdict[flt]['filttrans']

		g = (salt3wave >= filtwave[0]) & (salt3wave <= filtwave[-1])  # overlap range
		pbspl = np.interp(salt3wave[g],filtwave,filttrans)
		pbspl *= salt3wave[g]
		denom = np.trapz(pbspl,salt3wave[g])
		salt3synflux=np.trapz(pbspl[np.newaxis,:]*salt3fluxnew[:,g]/HC_ERG_AA,salt3wave[g],axis=1)/denom
		salt3synflux *= 10**(0.4*bandpassdict[flt]['stdmag'])*10**(0.4*27.5)/(1+float(sn.REDSHIFT_HELIO[0:5]))
		#10**(-0.4*bandpassdict[flt]['zpoff'])

		if 'SIM_SALT2x0' in sn.__dict__.keys():
			g = (salt2wave >= filtwave[0]) & (salt2wave <= filtwave[-1])  # overlap range
			pbspl = np.interp(salt2wave[g],filtwave,filttrans)
			pbspl *= salt2wave[g]
			denom = np.trapz(pbspl,salt2wave[g])
			salt2synflux=np.trapz(pbspl[np.newaxis,:]*salt2fluxnew[:,g]/HC_ERG_AA,salt2wave[g],axis=1)/denom
			salt2synflux *= 10**(0.4*bandpassdict[flt]['stdmag'])*10**(0.4*27.5)*10**(-0.4*0.27)/(1+float(sn.REDSHIFT_HELIO[0:5]))

		
		#if len(fitparams_salt3):
		#	ax.plot(plotmjd,salt3synflux,color='C2',
		#			label='SALT3, $\chi^2_{red} = %.1f$'%(
		#				result_salt3['chisq']/result_salt3['ndof']))
		#else:
		ax.plot(plotmjd,salt3synflux,color='C2',
				label='SALT3, $x_0$ = %8.5e, x1=%.2f, z=%.3f'%(x0,x1,float(sn.REDSHIFT_HELIO[0:5])))
		if 'SIM_SALT2x0' in sn.__dict__.keys():
			ax.plot(plotmjd,salt2synflux,color='C1',
					label='SALT2, x0=%8.5e, x1=%.2f, z=%.3f'%(sn.SIM_SALT2x0,sn.SIM_SALT2x1,float(sn.REDSHIFT_HELIO[0:5])))
			
		ax.errorbar(sn.MJD[sn.FLT == flt],sn.FLUXCAL[sn.FLT == flt],
					yerr=sn.FLUXCALERR[sn.FLT == flt],
					fmt='o',label=sn.SNID,color='k')
		#print('HACK')
		#ax.set_title(filtdict[flt])
		ax.set_title(flt)
		try:
			if 'PEAKMJD' in sn.__dict__.keys():
				ax.set_xlim([sn.PEAKMJD-30,
							 sn.PEAKMJD+55])
			else:
				iMax = np.where(sn.FLUXCAL == np.max(sn.FLUXCAL))[0]
				if len(iMax) > 1:
					iMax = iMax[0]

				ax.set_xlim([sn.MJD[iMax]-30,
							 sn.MJD[iMax]+55])
		except:
			import pdb; pdb.set_trace()
		ax.set_ylim([-np.max(sn.FLUXCAL)*1/20.,np.max(sn.FLUXCAL)*1.1])

		#if flt == 'c': import pdb; pdb.set_trace()
	ax1.legend()
	if 'SIM_SALT2x0' in sn.__dict__.keys():
		ax2.set_title('$x_0$ = %8.5e, $x_1$ = %.2f,\n$c$ = %.2f, $z$ = %.2f'%(
			sn.SIM_SALT2x0,sn.SIM_SALT2x1,sn.SIM_SALT2c,sn.SIM_REDSHIFT_HELIO))
	plt.savefig(outfile)

	#plt.ion()
	#plt.show()
	#import pdb; pdb.set_trace()
	#plt.close('all')
	
	
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
