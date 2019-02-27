#!/usr/bin/env python
# D. Jones, R. Kessler - 8/31/18
from __future__ import print_function

import os
import argparse
import configparser
import numpy as np
import sys
import multiprocessing
from scipy.linalg import lstsq

from salt3.util import snana
from salt3.util.estimate_tpk_bazin import estimate_tpk_bazin
from scipy.optimize import minimize, least_squares, differential_evolution
from salt3.training.fitting import fitting
from salt3.training.init_hsiao import init_hsiao
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo

class TrainSALT:
	def __init__(self):
		self.warnings = []
	
	def addwarning(self,warning):
		print(warning)
		self.warnings.append(warning)
		
	def add_options(self, parser=None, usage=None, config=None):
		if parser == None:
			parser = argparse.ArgumentParser(usage=usage, conflict_handler="resolve")

		# The basics
		parser.add_argument('-v', '--verbose', action="count", dest="verbose",
							default=1,help='verbosity level')
		parser.add_argument('--debug', default=False, action="store_true",
							help='debug mode: more output and debug files')
		parser.add_argument('--clobber', default=False, action="store_true",
							help='clobber')
		parser.add_argument('-c','--configfile', default=None, type=str,
							help='configuration file')
		parser.add_argument('-s','--stage', default='all', type=str,
							help='stage - options are train and validate')

		
		# input/output files
		parser.add_argument('--snlist', default=config.get('iodata','snlist'), type=str,
							help="""list of SNANA-formatted SN data files, including both photometry and spectroscopy. (default=%default)""")
		parser.add_argument('--speclist', default=config.get('iodata','speclist'), type=str,
							help="""optional list of ascii spectra, which will be written to the 
							SNANA-formatted SN light curve files provided by the snlist argument.
							List format should be space-delimited SNID, MJD-OBS (or DATE-OBS), spectrum filename (default=%default)""")
		parser.add_argument('--outputdir', default=config.get('iodata','outputdir'), type=str,
							help="""data directory for spectroscopy, format should be ASCII 
							with columns wavelength, flux, fluxerr (optional) (default=%default)""")
		parser.add_argument('--initmodelfile', default=config.get('iodata','initmodelfile'), type=str,
							help="""initial model to begin training, ASCII with columns
							phase, wavelength, flux (default=%default)""")
		parser.add_argument('--initbfilt', default=config.get('iodata','initbfilt'), type=str,
							help="""initial B-filter to get the normalization of the initial model (default=%default)""")
		parser.add_argument('--kcor_path', default=config.get('iodata','kcor_path'), type=str, action='append',
							help="""kcor_path gives survey,kcorfile for each survey used in the training data (default=%default)""")

		# training model parameters
		parser.add_argument('--waverange', default=list(map(int,config.get('trainparams','waverange').split(','))), type=int, nargs=2,
							help='wavelength range over which the model is defined (default=%default)')
		parser.add_argument('--colorwaverange', default=list(map(int,config.get('trainparams','colorwaverange').split(','))), type=int, nargs=2,
							help='wavelength range over which the color law is fit to data (default=%default)')
		parser.add_argument('--interpfunc', default=config.get('trainparams','interpfunc'), type=str,
							help='function to interpolate between control points in the fitting (default=%default)')
		parser.add_argument('--interporder', default=config.get('trainparams','interporder'), type=int,
							help='for splines/polynomial funcs, order of the function (default=%default)')
		parser.add_argument('--wavesplineres', default=config.get('trainparams','wavesplineres'), type=float,
							help='number of angstroms between each wavelength spline knot (default=%default)')
		parser.add_argument('--phasesplineres', default=config.get('trainparams','phasesplineres'), type=float,
							help='number of angstroms between each phase spline knot (default=%default)')
		parser.add_argument('--waveoutres', default=config.get('trainparams','waveoutres'), type=float,
							help='wavelength resolution in angstroms of the output file (default=%default)')
		parser.add_argument('--phaseoutres', default=config.get('trainparams','phaseoutres'), type=float,
							help='phase resolution in angstroms of the output file (default=%default)')
		parser.add_argument('--phaserange', default=list(map(int,config.get('trainparams','phaserange').split(','))), type=int, nargs=2,
							help='phase range over which model is trained (default=%default)')
		parser.add_argument('--fitmethod', default=config.get('trainparams','fitmethod'), type=str,
							help='fitting algorithm, passed to the fitter (default=%default)')
		parser.add_argument('--fitstrategy', default=config.get('trainparams','fitstrategy'), type=str,
							help='fitting strategy, one of leastsquares, minimize, emcee, hyperopt, simplemcmc or diffevol (default=%default)')
		parser.add_argument('--fititer', default=config.get('trainparams','fititer'), type=int,
							help='fitting iterations (default=%default)')
		parser.add_argument('--n_components', default=config.get('trainparams','n_components'), type=int,
							help='number of principal components of the SALT model to fit for (default=%default)')
		parser.add_argument('--n_colorpars', default=config.get('trainparams','n_colorpars'), type=int,
							help='number of degrees of the phase-independent color law polynomial (default=%default)')
		parser.add_argument('--n_specrecal', default=config.get('trainparams','n_specrecal'), type=int,
							help='number of parameters defining the spectral recalibration (default=%default)')
		parser.add_argument('--n_processes', default=config.get('trainparams','n_processes'), type=int,
							help='number of processes to use in calculating chi2 (default=%default)')
		parser.add_argument('--n_iter', default=config.get('trainparams','n_iter'), type=int,
							help='number of fitting iterations (default=%default)')
		parser.add_argument('--regulargradientphase', default=config.get('trainparams','regulargradientphase'), type=float,
							help='Weighting of phase gradient chi^2 regularization during training of model parameters (default=%default)')
		parser.add_argument('--regulargradientwave', default=config.get('trainparams','regulargradientwave'), type=float,
							help='Weighting of wave gradient chi^2 regularization during training of model parameters (default=%default)')
		parser.add_argument('--regulardyad', default=config.get('trainparams','regulardyad'), type=float,
							help='Weighting of dyadic chi^2 regularization during training of model parameters (default=%default)')

				

		# mcmc parameters
		parser.add_argument('--n_steps_mcmc', default=config.get('mcmcparams','n_steps_mcmc'), type=int,
							help='number of accepted MCMC steps (default=%default)')
		parser.add_argument('--n_burnin_mcmc', default=config.get('mcmcparams','n_burnin_mcmc'), type=int,
							help='number of burn-in MCMC steps  (default=%default)')
		parser.add_argument('--n_init_steps_mcmc', default=config.get('mcmcparams','n_init_steps_mcmc'), type=int,
							help='number of accepted MCMC steps, initialization stage (default=%default)')
		parser.add_argument('--n_init_burnin_mcmc', default=config.get('mcmcparams','n_init_burnin_mcmc'), type=int,
							help='number of burn-in MCMC steps, initialization stage  (default=%default)')
		parser.add_argument('--stepsize_M0', default=config.get('mcmcparams','stepsize_M0'), type=float,
							help='initial MCMC step size for M0, in mag  (default=%default)')
		parser.add_argument('--stepsize_mag_M1', default=config.get('mcmcparams','stepsize_mag_M1'), type=float,
							help='initial MCMC step size for M1, in mag - need both mag and flux steps because M1 can be negative (default=%default)')
		parser.add_argument('--stepsize_flux_M1', default=config.get('mcmcparams','stepsize_flux_M1'), type=float,
							help='initial MCMC step size for M1, in flux - need both mag and flux steps because M1 can be negative (default=%default)')
		parser.add_argument('--stepsize_cl', default=config.get('mcmcparams','stepsize_cl'), type=float,
							help='initial MCMC step size for color law  (default=%default)')
		parser.add_argument('--stepsize_specrecal', default=config.get('mcmcparams','stepsize_specrecal'), type=float,
							help='initial MCMC step size for spec recal. params  (default=%default)')
		parser.add_argument('--stepsize_x0', default=config.get('mcmcparams','stepsize_x0'), type=float,
							help='initial MCMC step size for x0, in mag  (default=%default)')
		parser.add_argument('--stepsize_x1', default=config.get('mcmcparams','stepsize_x1'), type=float,
							help='initial MCMC step size for x1  (default=%default)')
		parser.add_argument('--stepsize_c', default=config.get('mcmcparams','stepsize_c'), type=float,
							help='initial MCMC step size for c  (default=%default)')
		parser.add_argument('--stepsize_tpk', default=config.get('mcmcparams','stepsize_tpk'), type=float,
							help='initial MCMC step size for tpk  (default=%default)')

		
		return parser

	def rdkcor(self,kcorpath):

		self.kcordict = {}
		for k in kcorpath:
			survey,kcorfile = k.split(',')
			kcorfile = os.path.expandvars(kcorfile)
			if not os.path.exists(kcorfile):
				raise RuntimeError('kcor file %s does not exist'%kcorfile)
			self.kcordict[survey] = {}

			try:
				hdu = fits.open(kcorfile)
				zpoff = hdu[1].data
				snsed = hdu[2].data
				filtertrans = hdu[5].data
				primarysed = hdu[6].data
				hdu.close()
			except:
				raise RuntimeError('kcor file format is non-standard')

			self.kcordict[survey]['filtwave'] = filtertrans['wavelength (A)']
			self.kcordict[survey]['primarywave'] = primarysed['wavelength (A)']
			self.kcordict[survey]['snflux'] = snsed['SN Flux (erg/s/cm^2/A)']
			if 'AB' in primarysed.names:
				self.kcordict[survey]['AB'] = primarysed['AB']
			if 'Vega' in primarysed.names:
				self.kcordict[survey]['Vega'] = primarysed['Vega']
			if 'BD17' in primarysed.names:
				self.kcordict[survey]['BD17'] = primarysed['BD17']
			for filt in zpoff['Filter Name']:
				self.kcordict[survey][filt.split('-')[-1].split('/')[-1]] = {}
				self.kcordict[survey][filt.split('-')[-1].split('/')[-1]]['filttrans'] = filtertrans[filt]
				self.kcordict[survey][filt.split('-')[-1].split('/')[-1]]['zpoff'] = \
					zpoff['ZPOff(Primary)'][zpoff['Filter Name'] == filt][0]
				self.kcordict[survey][filt.split('-')[-1].split('/')[-1]]['magsys'] = \
					zpoff['Primary Name'][zpoff['Filter Name'] == filt][0]
				self.kcordict[survey][filt.split('-')[-1].split('/')[-1]]['primarymag'] = \
					zpoff['Primary Mag'][zpoff['Filter Name'] == filt][0]

	def rdSpecData(self,datadict,speclist,tpk):

		if not os.path.exists(speclist):
			raise RuntimeError('speclist %s does not exist')
		
		snid,mjd,specfiles = np.genfromtxt(speclist,unpack=True,dtype='str')
		snid,mjd,specfiles = np.atleast_1d(snid),np.atleast_1d(mjd),np.atleast_1d(specfiles)
		for s,m,sf in zip(snid,mjd,specfiles):
			try: m = float(m)
			except: m = snana.date_to_mjd(m)

			if '/' not in sf:
				sf = '%s/%s'%(os.path.dirname(speclist),sf)
			if not os.path.exists(sf):
				raise RuntimeError('specfile %s does not exist'%sf)
				
			if s in datadict.keys():
				if 'specdata' not in datadict[s].keys():
					datadict[s]['specdata'] = {}
					speccount = 0
				else:
					speccount = len(datadict[s]['specdata'].keys())
				datadict[s]['specdata'][speccount] = {}
				try:
					wave,flux,fluxerr = np.genfromtxt(sf,unpack=True,usecols=[0,1,2])
					datadict[s]['specdata'][speccount]['fluxerr'] = fluxerr
				except:
					wave,flux = np.genfromtxt(sf,unpack=True,usecols=[0,1])
				datadict[s]['specdata'][speccount]['wavelength'] = wave
				datadict[s]['specdata'][speccount]['flux'] = flux
				datadict[s]['specdata'][speccount]['tobs'] = m - tpk
				datadict[s]['specdata'][speccount]['mjd'] = m
			else:
				print('SNID %s has no photometry so I\'m ignoring it')

		return datadict

	def rdAllData(self,snlist,speclist=None):
		datadict = {}

		if not os.path.exists(snlist):
			raise RuntimeError('SN list %s doesn\'t exist'%snlist)
		snfiles = np.genfromtxt(snlist,dtype='str')
		snfiles = np.atleast_1d(snfiles)

		for f in snfiles:
			if f.lower().endswith('.fits'):
				raise RuntimeError('FITS extensions are not supported yet')

			if '/' not in f:
				f = '%s/%s'%(os.path.dirname(snlist),f)
			sn = snana.SuperNova(f)

			if sn.SNID in datadict.keys():
				self.addwarning('SNID %s is a duplicate!  Skipping'%sn.SNID)
				continue

			if not 'SURVEY' in sn.__dict__.keys():
				raise RuntimeError('File %s has no SURVEY key, which is needed to find the filter transmission curves'%PhotSNID[0])
			if not 'REDSHIFT_HELIO' in sn.__dict__.keys():
				raise RuntimeError('File %s has no heliocentric redshift information in the header'%PhotSNID[0])

			if 'PEAKMJD' in sn.__dict__.keys(): sn.SEARCH_PEAKMJD = sn.PEAKMJD
			zHel = float(sn.REDSHIFT_HELIO.split('+-')[0])
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

			# at least one epoch 3 days before max
			if not len(sn.MJD[sn.MJD < tpk-3]):
				self.addwarning('skipping SN %s; no epochs 3 days pre-max'%sn.SNID)
				continue

			if 'termination condition is satisfied' not in tpkmsg:
				self.addwarning('skipping SN %s; can\'t estimate t_max'%sn.SNID)
				continue

			datadict[sn.SNID] = {'snfile':f,
								 'zHelio':zHel,
								 'survey':sn.SURVEY}
			#datadict[snid]['zHelio'] = zHel
			
			# TODO: flux errors
			datadict[sn.SNID]['specdata'] = {}
			for k in sn.SPECTRA.keys():
				datadict[sn.SNID]['specdata'][k] = {}
				datadict[sn.SNID]['specdata'][k]['specphase'] = sn.SPECTRA[k]['SPECTRUM_MJD']
				datadict[sn.SNID]['specdata'][k]['tobs'] = sn.SPECTRA[k]['SPECTRUM_MJD'] - tpk
				datadict[sn.SNID]['specdata'][k]['mjd'] = sn.SPECTRA[k]['SPECTRUM_MJD']
				if 'LAMAVG' in sn.SPECTRA[k].keys():
					datadict[sn.SNID]['specdata'][k]['wavelength'] = sn.SPECTRA[k]['LAMAVG']
				elif 'LAMMIN' in sn.SPECTRA[k].keys() and 'LAMMAX' in sn.SPECTRA[k].keys():
					datadict[sn.SNID]['specdata'][k]['wavelength'] = np.mean([[sn.SPECTRA[k]['LAMMIN']],
																			  [sn.SPECTRA[k]['LAMMAX']]],axis=0)
				else:
					raise RuntimeError('couldn\t find wavelength data in photometry file')
				datadict[sn.SNID]['specdata'][k]['flux'] = sn.SPECTRA[k]['FLAM']
				datadict[sn.SNID]['specdata'][k]['fluxerr'] = sn.SPECTRA[k]['FLAMERR']
				
			datadict[sn.SNID]['photdata'] = {}
			datadict[sn.SNID]['photdata']['tobs'] = sn.MJD - tpk
			datadict[sn.SNID]['photdata']['mjd'] = sn.MJD
			datadict[sn.SNID]['photdata']['fluxcal'] = sn.FLUXCAL
			datadict[sn.SNID]['photdata']['fluxcalerr'] = sn.FLUXCALERR
			datadict[sn.SNID]['photdata']['filt'] = sn.FLT

		if not len(datadict.keys()):
			raise RuntimeError('no light curve data to train on!!')
			
		if speclist:
			datadict = self.rdSpecData(datadict,speclist,tpk)
			
		return datadict
		
	def fitSALTModel(self,datadict,phaserange,phaseres,waverange,waveres,
					 colorwaverange,fitmethod,fitstrategy,fititer,kcordict,initmodelfile,initBfilt,
					 phaseoutres,waveoutres,regulargradientphase, regulargradientwave, regulardyad ,n_components=1,n_colorpars=0,n_processes=1):

		if self.options.fitstrategy == 'multinest' or self.options.fitstrategy == 'simplemcmc':
			from salt3.training import saltfit_mcmc as saltfit
		else:
			from salt3.training import saltfit
		
		if not os.path.exists(initmodelfile):
			from salt3.initfiles import init_rootdir
			initmodelfile = '%s/%s'%(init_rootdir,initmodelfile)
			initBfilt = '%s/%s'%(init_rootdir,initBfilt)
			salt2file = '%s/salt2_template_0.dat.gz'%init_rootdir
		if not os.path.exists(initmodelfile):
			raise RuntimeError('model initialization file not found in local directory or %s'%init_rootdir)
		
		phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_hsiao(
			initmodelfile,salt2file,initBfilt,phaserange=phaserange,waverange=waverange,
			phasesplineres=phaseres,wavesplineres=waveres,
			phaseinterpres=phaseoutres,waveinterpres=waveoutres)
		n_phaseknots,n_waveknots = len(phaseknotloc)-4,len(waveknotloc)-4
		n_sn = len(datadict.keys())
		# x1,x0,c for each SN
		# phase/wavelength spline knots for M0, M1 (ignoring color for now)
		# TODO: spectral recalibration
		n_params = n_components*n_phaseknots*n_waveknots + 4*n_sn
		
		if n_colorpars: n_params += n_colorpars
		print(n_params)
		guess = np.zeros(n_params)

		parlist = np.array(['m0']*(n_phaseknots*n_waveknots))
		if n_components == 2:
			parlist = np.append(parlist,['m1']*(n_phaseknots*n_waveknots))
		if n_colorpars:
			parlist = np.append(parlist,['cl']*n_colorpars)

		for k in datadict.keys():
			parlist = np.append(parlist,['x0_%s'%k,'x1_%s'%k,'c_%s'%k,'tpkoff_%s'%k])
		parlist = np.array(parlist)

		
		guess[parlist == 'm0'] = m0knots
		if n_components == 2:
			guess[parlist == 'm1'] = m1knots
		if n_colorpars:
			guess[parlist == 'cl'] = [0.]*n_colorpars
		guess[(parlist == 'm0') & (guess < 0)] = 0

		saltfitter = saltfit.chi2(guess,datadict,parlist,
								  phaseknotloc,waveknotloc,phaserange,
								  waverange,phaseres,waveres,phaseoutres,waveoutres,
								  colorwaverange,
								  kcordict,initmodelfile,initBfilt,regulargradientphase, regulargradientwave, regulardyad ,n_components,n_colorpars)

		saltfitter.stepsize_M0 = self.options.stepsize_M0
		saltfitter.stepsize_mag_M1 = self.options.stepsize_mag_M1
		saltfitter.stepsize_flux_M1 = self.options.stepsize_flux_M1
		saltfitter.stepsize_cl = self.options.stepsize_cl
		saltfitter.stepsize_specrecal = self.options.stepsize_specrecal
		saltfitter.stepsize_x0 = self.options.stepsize_x0
		saltfitter.stepsize_x1 = self.options.stepsize_x1
		saltfitter.stepsize_c = self.options.stepsize_c
		saltfitter.stepsize_tpk = self.options.stepsize_tpk

		# first pass - estimate x0 so we can bound it to w/i an order of mag
		initbounds = ([0,-np.inf,-np.inf,-5]*n_sn,[np.inf,np.inf,np.inf,5]*n_sn)
		initparlist = []
		initguess = ()
		for k in datadict.keys():
			initparlist += ['x0_%s'%k,'x1_%s'%k,'c_%s'%k,'tpkoff_%s'%k]
			initguess += (10**(-0.4*(cosmo.distmod(datadict[k]['zHelio']).value-19.36)),0,0,0)
		#initparlist += ['cl']*n_colorpars
		initparlist = np.array(initparlist)

		
		for i in range(self.options.n_iter):
			saltfitter.onlySNpars = True
			if i > 0:
				initguess = SNpars.transpose()[0]
				saltfitter.components = saltfitter.SALTModel(x)
			saltfitter.parlist = initparlist

			fitter = fitting(n_components,n_colorpars,
							 n_phaseknots,n_waveknots,
							 datadict,initguess,
							 initparlist,parlist)

			initX,phase,wave,M0,M1,clpars,SNParams,message = fitter.mcmc(
				saltfitter,initguess,(),(),n_processes,
				self.options.n_init_steps_mcmc,
				self.options.n_init_burnin_mcmc,init=True)
			
			try:
				if 'condition is satisfied' not in message.decode('utf-8'):
					self.addwarning('Initialization MCMC message: %s'%message)
			except:
				if 'condition is satisfied' not in message:
					self.addwarning('Initialization MCMC message: %s'%message)
			if self.verbose:
				print('SN guesses initialized successfully')
			saltfitter.updateEffectivePoints(initX)
			#saltfitter.plotEffectivePoints()
			# 2nd pass - let the SALT model spline knots float			
			SNpars,SNparlist = [],[]
			for k in datadict.keys():
				tpk_init = datadict[k]['photdata']['mjd'][0] - datadict[k]['photdata']['tobs'][0]
				guess[parlist == 'x0_%s'%k] = SNParams[k]['x0']
				SNpars += [SNParams[k]['x0'],SNParams[k]['x1'],SNParams[k]['c'],SNParams[k]['tpkoff']]
				SNparlist += ['x0_%s'%k,'x1_%s'%k,'c_%s'%k,'tpkoff_%s'%k]
			SNparlist = np.array(SNparlist); SNpars = np.array(SNpars)
				#guess[parlist == 'cl'] = md_init.x[initparlist == 'cl']
			
			if i > 0:
				guess[parlist == 'm0'] = x[parlist == 'm0']
				if n_components == 2:
					guess[parlist == 'm1'] = x[parlist == 'm1']
				if n_colorpars:
					guess[parlist == 'cl'] = clpars
				
			saltfitter.parlist = parlist
			saltfitter.onlySNpars = False

			fitter = fitting(n_components,n_colorpars,
							 n_phaseknots,n_waveknots,
							 datadict,guess,
							 initparlist,parlist)

			x,phase,wave,M0,M1,clpars,SNParams,message = fitter.mcmc(
				saltfitter,guess,SNpars,SNparlist,n_processes,
				self.options.n_steps_mcmc,self.options.n_burnin_mcmc)
			for k in datadict.keys():
				tpk_init = datadict[k]['photdata']['mjd'][0] - datadict[k]['photdata']['tobs'][0]
				SNParams[k]['t0'] = -SNParams[k]['tpkoff'] + tpk_init
			
			try:
				if 'condition is satisfied' not in message.decode('utf-8'):
					self.addwarning('MCMC message on iter %i: %s'%(i,message))
			except:
				if 'condition is satisfied' not in message:
					self.addwarning('MCMC message on iter %i: %s'%(i,message))
					self.addwarning('Minimizer message on iter %i: %s'%(i,message))
			print('Individual components of final regularization chi^2'); saltfitter.regularizationChi2(x,1,1,1)
			print('Final chi^2'); saltfitter.chi2fit(x,None,False,False)

		return phase,wave,M0,M1,clpars,SNParams

	def wrtoutput(self,outdir,phase,wave,M0,M1,clpars,SNParams):

		if not os.path.exists(outdir):
			raise RuntimeError('desired output directory %s doesn\'t exist'%outdir)

		# principal components and color law
		foutm0 = open('%s/salt3_template_0.dat'%outdir,'w')
		foutm1 = open('%s/salt3_template_1.dat'%outdir,'w')
		foutcl = open('%s/salt3_color_correction.dat'%outdir,'w')
		
		for p,i in zip(phase,range(len(phase))):
			for w,j in zip(wave,range(len(wave))):
				print('%.1f %.2f %8.5e'%(p,w,M0[i,j]),file=foutm0)
				print('%.1f %.2f %8.5e'%(p,w,M1[i,j]),file=foutm1)

		foutm0.close()
		foutm1.close()

		print('%i'%len(clpars),file=foutcl)
		for c in clpars:
			print('%8.5e'%c,file=foutcl)
		print("""Salt2ExtinctionLaw.version 1
Salt2ExtinctionLaw.min_lambda %i
Salt2ExtinctionLaw.max_lambda %i"""%(
	self.options.colorwaverange[0],
	self.options.colorwaverange[1]),file=foutcl)
		foutcl.close()

		# best-fit SN params
		foutsn = open('%s/salt3train_snparams.txt'%outdir,'w')
		print('# SN x0 x1 c t0',file=foutsn)
		for k in SNParams.keys():
			print('%s %8.5e %.4f %.4f %.2f'%(k,SNParams[k]['x0'],SNParams[k]['x1'],SNParams[k]['c'],SNParams[k]['t0']),file=foutsn)
		foutsn.close()
			
		return

	def validate(self,outputdir):

		import pylab as plt
		plt.ion()
		
		from salt3.validation import ValidateLightcurves
		from salt3.validation import ValidateModel

		x0,x1,c,t0 = np.loadtxt('%s/salt3train_snparams.txt'%outputdir,unpack=True,usecols=[1,2,3,4])
		snid = np.genfromtxt('%s/salt3train_snparams.txt'%outputdir,unpack=True,dtype='str',usecols=[0])
		
		ValidateModel.main(
			'%s/spectralcomp.png'%outputdir,
			outputdir)
		
		snfiles = np.genfromtxt(self.options.snlist,dtype='str')
		snfiles = np.atleast_1d(snfiles)
		fitx1,fitc = False,False
		if self.options.n_components == 2:
			fitx1 = True
		if self.options.n_colorpars > 0:
			fitc = True

		from salt3.util.synphot import synphot
		kcordict = {}
		for k in self.kcordict.keys():
			for k2 in self.kcordict[k].keys():
				if k2 not in ['primarywave','snflux','BD17','filtwave','AB']:
					if self.kcordict[k][k2]['magsys'] == 'AB': primarykey = 'AB'
					elif self.kcordict[k][k2]['magsys'] == 'Vega': primarykey = 'Vega'
					elif self.kcordict[k][k2]['magsys'] == 'BD17': primarykey = 'BD17'

					kcordict[k2] = self.kcordict[k][k2]
					kcordict[k2]['filtwave'] = self.kcordict[k]['filtwave']
					kcordict[k2]['stdmag'] = synphot(self.kcordict[k]['primarywave'],self.kcordict[k][primarykey],filtwave=self.kcordict[k]['filtwave'],
													 filttp=self.kcordict[k][k2]['filttrans'],
													 zpoff=0) - self.kcordict[k][k2]['primarymag']
#		import pdb; pdb.set_trace()
				
		for l in snfiles:
			plt.clf()
			if '/' not in l:
				l = '%s/%s'%(os.path.dirname(self.options.snlist),l)
			sn = snana.SuperNova(l)
			sn.SNID = str(sn.SNID)

			if sn.SNID not in snid:
				self.addwarning('sn %s not in output files'%sn.SNID)
				continue
			x0sn,x1sn,csn,t0sn = \
				x0[snid == sn.SNID][0],x1[snid == sn.SNID][0],\
				c[snid == sn.SNID][0],t0[snid == sn.SNID][0]
			if not fitc: csn = 0
			if not fitx1: x1sn = 0
			
			ValidateLightcurves.customfilt(
				'%s/lccomp_%s.png'%(outputdir,sn.SNID),l,outputdir,
				t0=t0sn,x0=x0sn,x1=x1sn,c=csn,fitx1=fitx1,fitc=fitc,
				bandpassdict=kcordict)

		
	def main(self):

		if not self.options.kcor_path:
			raise RuntimeError('kcor_path variable must be defined!')
		self.rdkcor(self.options.kcor_path)
		# TODO: ASCII filter files
		
		# read the data
		datadict = self.rdAllData(self.options.snlist,speclist=self.options.speclist)
		
		if not os.path.exists(self.options.outputdir):
			os.makedirs(self.options.outputdir)
		
		# Eliminate all data outside phase range
		numSpecElimmed,numSpec=0,0
		numPhotElimmed,numPhot=0,0
		for sn in datadict.keys():
			photdata = datadict[sn]['photdata']
			specdata = datadict[sn]['specdata']
			z = datadict[sn]['zHelio']
			#Remove spectra outside phase range
			for k in list(specdata.keys()):
				if ((specdata[k]['tobs'])/(1+z)<self.options.phaserange[0]) or ((specdata[k]['tobs'])/(1+z)>self.options.phaserange[1]):
					specdata.pop(k)
					numSpecElimmed+=1
				else:
					numSpec+=1
			#Remove photometric data outside phase range
			phase=(photdata['tobs'])/(1+z)
			phaseFilter=(phase>self.options.phaserange[0]) & (phase<self.options.phaserange[1])
			numPhotElimmed+=(~phaseFilter).sum()
			numPhot+=phaseFilter.sum()
			datadict[sn]['photdata'] ={key:photdata[key][phaseFilter] for key in photdata}
		print('{} spectra and {} photometric observations removed for being outside phase range'.format(numSpecElimmed,numPhotElimmed))
		print('{} spectra and {} photometric observations remaining'.format(numSpec,numPhot))
		# fit the model - initial pass
		if self.options.stage == "all" or self.options.stage == "train":
			phase,wave,M0,M1,clpars,SNParams = self.fitSALTModel(
				datadict,self.options.phaserange,self.options.phasesplineres,
				self.options.waverange,self.options.wavesplineres,
				self.options.colorwaverange,
				self.options.fitmethod,
				self.options.fitstrategy,
				self.options.fititer,
				self.kcordict,
				self.options.initmodelfile,
				self.options.initbfilt,
				self.options.phaseoutres,
				self.options.waveoutres,
				self.options.regulargradientphase, 
				self.options.regulargradientwave, 
				self.options.regulardyad,
				self.options.n_components,
				self.options.n_colorpars)
		
			# write the output model - M0, M1, c
			self.wrtoutput(self.options.outputdir,phase,wave,M0,M1,clpars,SNParams)

		if self.options.stage == "all" or self.options.stage == "validate":
			self.validate(self.options.outputdir)
		
		print('successful SALT2 training!  Output files written to %s'%self.options.outputdir)
		
	
if __name__ == "__main__":
	usagestring = """SALT3 Training

usage: python TrainSALT.py -c <configfile> <options>

config file options can be overwridden at the command line

Dependencies: sncosmo?
"""

	if sys.version_info < (3,0):
		sys.exit('Sorry, Python 2 is not supported')
	
	salt = TrainSALT()

	parser = argparse.ArgumentParser(usage=usagestring, conflict_handler="resolve")
	parser.add_argument('-c','--configfile', default=None, type=str,
					  help='configuration file')
	options, args = parser.parse_known_args()

	if options.configfile:
		config = configparser.ConfigParser()
		if not os.path.exists(options.configfile):
			raise RuntimeError('Configfile doesn\'t exist!')
		config.read(options.configfile)
	else:
		parser.print_help()
		raise RuntimeError('Configuration file must be specified at command line')

	parser = salt.add_options(usage=usagestring,config=config)
	options = parser.parse_args()

	options.kcor_path = (options.kcor_path,)
	with open(options.configfile) as fin:
		for line in fin:
			if line.startswith('kcor_path+'):
				options.kcor_path += (line.replace('\n','').split('=')[-1],)
	
	salt.options = options
	salt.verbose = options.verbose
	salt.clobber = options.clobber
	
	salt.main()

	if len(salt.warnings):
		print('There were warnings!!')
		print(salt.warnings)

