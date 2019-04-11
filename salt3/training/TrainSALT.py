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

from salt3.util import snana,readutils
from salt3.util.estimate_tpk_bazin import estimate_tpk_bazin
from scipy.optimize import minimize, least_squares, differential_evolution
from salt3.training.fitting import fitting
from salt3.training.init_hsiao import init_hsiao, init_errs
from salt3.initfiles import init_rootdir
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo

testx1dict = {5999387:-2.3919,
			  5999388: 0.8029,
			  5999389: 0.6971,
			  5999390:-2.9930,
			  5999391:-1.0716,
			  5999392:-2.2737,
			  5999393: 1.7047,
			  5999394: 1.1291,
			  5999395: 2.0568,
			  5999396: 1.6647,
			  5999397:-2.1002,
			  5999398: 2.5913,
			  5999399:-1.2756,
			  5999400:-1.8371,
			  5999401:-2.3846,
			  5999402: 2.9826,
			  5999403:-0.7240,
			  5999404:-2.6362,
			  5999405: 1.9302,
			  5999406:-1.5747}

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
		parser.add_argument('--n_colorscatpars', default=config.get('trainparams','n_colorscatpars'), type=int,
							help='number of parameters in the broadband scatter model (default=%default)')
		parser.add_argument('--n_specrecal', default=config.get('trainparams','n_specrecal'), type=int,
							help='number of parameters defining the spectral recalibration (default=%default)')
		parser.add_argument('--n_processes', default=config.get('trainparams','n_processes'), type=int,
							help='number of processes to use in calculating chi2 (default=%default)')
		parser.add_argument('--n_iter', default=config.get('trainparams','n_iter'), type=int,
							help='number of fitting iterations (default=%default)')
		parser.add_argument('--estimate_tpk', default=config.get('trainparams','estimate_tpk'), type=bool,
							help='if set, estimate time of max with quick least squares fitting (default=%default)')
		parser.add_argument('--regulargradientphase', default=config.get('trainparams','regulargradientphase'), type=float,
							help='Weighting of phase gradient chi^2 regularization during training of model parameters (default=%default)')
		parser.add_argument('--regulargradientwave', default=config.get('trainparams','regulargradientwave'), type=float,
							help='Weighting of wave gradient chi^2 regularization during training of model parameters (default=%default)')
		parser.add_argument('--regulardyad', default=config.get('trainparams','regulardyad'), type=float,
							help='Weighting of dyadic chi^2 regularization during training of model parameters (default=%default)')
		parser.add_argument('--n_min_specrecal', default=config.get('trainparams','n_min_specrecal'), type=int,
							help='Minimum order of spectral recalibration polynomials (default=%default)')
		parser.add_argument('--specrange_wavescale_specrecal', default=config.get('trainparams','specrange_wavescale_specrecal'), type=float,
							help='Wavelength scale (in angstroms) for determining additional orders of spectral recalibration from wavelength range of spectrum (default=%default)')
		parser.add_argument('--n_specrecal_per_lightcurve', default=config.get('trainparams','n_specrecal_per_lightcurve'), type=float,
							help='Number of additional spectral recalibration orders per lightcurve (default=%default)')
		parser.add_argument('--filter_mass_tolerance', default=config.get('trainparams','filter_mass_tolerance'), type=float,
							help='Mass of filter transmission allowed outside of model wavelength range (default=%default)')
		parser.add_argument('--error_snake_phase_binsize', default=config.get('trainparams','error_snake_phase_binsize'), type=float,
							help='number of days over which to compute scaling of error model (default=%default)')
		parser.add_argument('--error_snake_wave_binsize', default=config.get('trainparams','error_snake_wave_binsize'), type=float,
							help='number of angstroms over which to compute scaling of error model (default=%default)')


		# mcmc parameters
		parser.add_argument('--n_steps_mcmc', default=config.get('mcmcparams','n_steps_mcmc'), type=int,
							help='number of accepted MCMC steps (default=%default)')
		parser.add_argument('--n_burnin_mcmc', default=config.get('mcmcparams','n_burnin_mcmc'), type=int,
							help='number of burn-in MCMC steps  (default=%default)')
		parser.add_argument('--n_init_steps_mcmc', default=config.get('mcmcparams','n_init_steps_mcmc'), type=int,
							help='number of accepted MCMC steps, initialization stage (default=%default)')
		parser.add_argument('--n_init_burnin_mcmc', default=config.get('mcmcparams','n_init_burnin_mcmc'), type=int,
							help='number of burn-in MCMC steps, initialization stage  (default=%default)')
		parser.add_argument('--stepsize_magscale_M0', default=config.get('mcmcparams','stepsize_magscale_M0'), type=float,
							help='initial MCMC step size for M0, in mag  (default=%default)')
		parser.add_argument('--stepsize_magadd_M0', default=config.get('mcmcparams','stepsize_magadd_M0'), type=float,
							help='initial MCMC step size for M0, in mag  (default=%default)')
		parser.add_argument('--stepsize_magscale_err', default=config.get('mcmcparams','stepsize_magscale_err'), type=float,
							help='initial MCMC step size for the model err spline knots, in mag  (default=%default)')
		parser.add_argument('--stepsize_magscale_M1', default=config.get('mcmcparams','stepsize_magscale_M1'), type=float,
							help='initial MCMC step size for M1, in mag - need both mag and flux steps because M1 can be negative (default=%default)')
		parser.add_argument('--stepsize_magadd_M1', default=config.get('mcmcparams','stepsize_magadd_M1'), type=float,
							help='initial MCMC step size for M1, in flux - need both mag and flux steps because M1 can be negative (default=%default)')
		parser.add_argument('--stepsize_cl', default=config.get('mcmcparams','stepsize_cl'), type=float,
							help='initial MCMC step size for color law  (default=%default)')
		parser.add_argument('--stepsize_magscale_clscat', default=config.get('mcmcparams','stepsize_magscale_clscat'), type=float,
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

		# adaptive MCMC parameters
		parser.add_argument('--nsteps_before_adaptive', default=config.get('mcmcparams','nsteps_before_adaptive'), type=float,
							help='number of steps before starting adaptive step sizes (default=%default)')
		parser.add_argument('--nsteps_adaptive_memory', default=config.get('mcmcparams','nsteps_adaptive_memory'), type=float,
							help='number of steps to use to estimate adaptive steps (default=%default)')
		parser.add_argument('--adaptive_sigma_opt_scale', default=config.get('mcmcparams','adaptive_sigma_opt_scale'), type=float,
							help='scaling the adaptive step sizes (default=%default)')
		
		return parser

		
	def fitSALTModel(self,datadict,phaserange,phaseres,waverange,waveres,
					 colorwaverange,fitmethod,fitstrategy,fititer,kcordict,initmodelfile,initBfilt,
					 phaseoutres,waveoutres,regulargradientphase, regulargradientwave, regulardyad,n_min_specrecal,specrange_wavescale_specrecal,n_specrecal_per_lightcurve,filter_mass_tolerance ,n_components=1,n_colorpars=0,n_processes=1):

		if self.options.fitstrategy == 'multinest' or self.options.fitstrategy == 'simplemcmc':
			from salt3.training import saltfit_mcmc as saltfit
		else:
			from salt3.training import saltfit
		
		if not os.path.exists(initmodelfile):
			from salt3.initfiles import init_rootdir
			initmodelfile = '%s/%s'%(init_rootdir,initmodelfile)
			flatnu='%s/flatnu.dat'%(init_rootdir)
			initBfilt = '%s/%s'%(init_rootdir,initBfilt)
			salt2file = '%s/salt2_template_0.dat.gz'%init_rootdir
		if not os.path.exists(initmodelfile):
			raise RuntimeError('model initialization file not found in local directory or %s'%init_rootdir)
		
		phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_hsiao(
			initmodelfile,salt2file,initBfilt,flatnu,phaserange=phaserange,waverange=waverange,
			phasesplineres=phaseres,wavesplineres=waveres,
			phaseinterpres=phaseoutres,waveinterpres=waveoutres)
		errphaseknotloc,errwaveknotloc = init_errs(
			initmodelfile,salt2file,initBfilt,phaserange=phaserange,waverange=waverange,
			phasesplineres=self.options.error_snake_phase_binsize,
			wavesplineres=self.options.error_snake_wave_binsize,
			phaseinterpres=phaseoutres,waveinterpres=waveoutres)
		
		
		n_phaseknots,n_waveknots = len(phaseknotloc)-4,len(waveknotloc)-4
		n_errphaseknots,n_errwaveknots = len(errphaseknotloc)-4,len(errwaveknotloc)-4
		n_sn = len(datadict.keys())
		m1knots = m0knots*1e-2
		# x1,x0,c for each SN
		# phase/wavelength spline knots for M0, M1 (ignoring color for now)
		# TODO: spectral recalibration

		parlist = np.array(['m0']*(n_phaseknots*n_waveknots))
		if n_components == 2:
			parlist = np.append(parlist,['m1']*(n_phaseknots*n_waveknots))
		if n_colorpars:
			parlist = np.append(parlist,['cl']*n_colorpars)
		if self.options.error_snake_phase_binsize and self.options.error_snake_wave_binsize:
			parlist = np.append(parlist,['modelerr']*n_errphaseknots*n_errwaveknots)
		if self.options.n_colorscatpars:
			# four knots for the end points
			parlist = np.append(parlist,['clscat']*(self.options.n_colorscatpars+8))
			
		for k in datadict.keys():
			parlist = np.append(parlist,['x0_%s'%k,'x1_%s'%k,'c_%s'%k,'tpkoff_%s'%k])
			
		for sn in datadict.keys():
			specdata=datadict[sn]['specdata']
			photdata=datadict[sn]['photdata']
			for k in specdata.keys():
				order=n_min_specrecal+int(np.log((specdata[k]['wavelength'].max() - specdata[k]['wavelength'].min())/specrange_wavescale_specrecal) +np.unique(photdata['filt']).size* n_specrecal_per_lightcurve)
				parlist=np.append(parlist,['specrecal_{}_{}'.format(sn,k)]*order)
				
		parlist = np.array(parlist)
		
		n_params=parlist.size
		guess = np.zeros(parlist.size)
		guess[parlist == 'm0'] = m0knots
		guess[parlist == 'modelerr'] = 0.01
		if n_components == 2:
			guess[parlist == 'm1'] = m1knots
		if n_colorpars:
			guess[parlist == 'cl'] = [0.]*n_colorpars
		guess[(parlist == 'm0') & (guess < 0)] = 0
		
		
		saltfitter = saltfit.chi2(guess,datadict,parlist,
								  phaseknotloc,waveknotloc,
								  errphaseknotloc,errwaveknotloc,
								  phaserange,
								  waverange,phaseres,waveres,phaseoutres,waveoutres,
								  colorwaverange,
								  kcordict,initmodelfile,initBfilt,regulargradientphase,
								  regulargradientwave,regulardyad,filter_mass_tolerance,specrange_wavescale_specrecal,
								  n_components,n_colorpars,
								  n_iter=self.options.n_iter,
								  nsteps_before_adaptive=self.options.nsteps_before_adaptive,
								  nsteps_adaptive_memory=self.options.nsteps_adaptive_memory,
								  adaptive_sigma_opt_scale=self.options.adaptive_sigma_opt_scale)

		saltfitter.stepsize_magscale_M0 = self.options.stepsize_magscale_M0
		saltfitter.stepsize_magadd_M0 = self.options.stepsize_magadd_M0
		saltfitter.stepsize_magscale_err = self.options.stepsize_magscale_err
		saltfitter.stepsize_magscale_M1 = self.options.stepsize_magscale_M1
		saltfitter.stepsize_magadd_M1 = self.options.stepsize_magadd_M1
		saltfitter.stepsize_cl = self.options.stepsize_cl
		saltfitter.stepsize_magscale_clscat = self.options.stepsize_magscale_clscat
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
			initguess += (10**(-0.4*(cosmo.distmod(datadict[k]['zHelio']).value-19.24-10.635)),0,0,0)
		initparlist = np.array(initparlist)
		print('training on %i SNe!'%len(datadict.keys()))

		
		for k in datadict.keys():
			guess[parlist == 'x0_%s'%k] = 10**(-0.4*(cosmo.distmod(datadict[k]['zHelio']).value-19.36))

		saltfitter.parlist = parlist
		saltfitter.onlySNpars = False

		fitter = fitting(n_components,n_colorpars,
						 n_phaseknots,n_waveknots,
						 datadict)

		# do the fitting
		x_modelpars,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams,message = fitter.mcmc(
			saltfitter,guess,(),(),n_processes,
			self.options.n_steps_mcmc,self.options.n_burnin_mcmc)
		for k in datadict.keys():
			tpk_init = datadict[k]['photdata']['mjd'][0] - datadict[k]['photdata']['tobs'][0]
			SNParams[k]['t0'] = -SNParams[k]['tpkoff'] + tpk_init

		try:
			if 'condition is satisfied' not in message.decode('utf-8'):
				self.addwarning('MCMC message on iter 0: %s'%(i,message))
		except:
			if 'condition is satisfied' not in message:
				self.addwarning('MCMC message on iter 0: %s'%(message))
				self.addwarning('Minimizer message on iter 0: %s'%(message))
		print('Final regularization chi^2 terms:', saltfitter.regularizationChi2(x_modelpars,1,0,0),
			  saltfitter.regularizationChi2(x_modelpars,0,1,0),saltfitter.regularizationChi2(x_modelpars,0,0,1))
		print('Final chi^2'); saltfitter.chi2fit(x_modelpars,None,False,False)


		
		return phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams,x_modelpars,parlist

	def wrtoutput(self,outdir,phase,wave,
				  M0,M0err,M1,M1err,cov_M0_M1,
				  modelerr,clpars,
				  clerr,clscat,SNParams,pars,parlist):

		if not os.path.exists(outdir):
			raise RuntimeError('desired output directory %s doesn\'t exist'%outdir)

		#Save all model parameters
		
		with  open('{}/salt3_parameters.dat'.format(outdir),'w') as foutpars:
			foutpars.write('{: <30} {}\n'.format('Parameter Name','Value'))
			for name,par in zip(parlist,pars):
				foutpars.write('{: <30} {:.6e}\n'.format(name,par))
			
		# principal components and color law
		foutm0 = open('%s/salt3_template_0.dat'%outdir,'w')
		foutm1 = open('%s/salt3_template_1.dat'%outdir,'w')
		foutm0err = open('%s/salt3_lc_relative_variance_0.dat'%outdir,'w')
		foutm1err = open('%s/salt3_lc_relative_variance_1.dat'%outdir,'w')
		fouterrmod = open('%s/salt3_lc_dispersion_scaling.dat'%outdir,'w')
		foutcov = open('%s/salt3_lc_relative_covariance_01.dat'%outdir,'w')
		foutcl = open('%s/salt3_color_correction.dat'%outdir,'w')

		for p,i in zip(phase,range(len(phase))):
			for w,j in zip(wave,range(len(wave))):
				print('%.1f %.2f %8.5e'%(p,w,M0[i,j]),file=foutm0)
				print('%.1f %.2f %8.5e'%(p,w,M1[i,j]),file=foutm1)
				print('%.1f %.2f %8.5e'%(p,w,M0err[i,j]**2.),file=foutm0err)
				print('%.1f %.2f %8.5e'%(p,w,M1err[i,j]**2.),file=foutm1err)
				print('%.1f %.2f %8.5e'%(p,w,cov_M0_M1[i,j]),file=foutcov)
				print('%.1f %.2f %8.5e'%(p,w,modelerr[i,j]),file=fouterrmod)

		foutclscat = open('%s/salt3_color_dispersion.dat'%outdir,'w')
		for w,j in zip(wave,range(len(wave))):
			print('%.2f %8.5e'%(w,clscat[j]),file=foutclscat)
		foutclscat.close()
				
		foutm0.close()
		foutm1.close()
		foutm0err.close()
		foutm1err.close()
		foutcov.close()
		fouterrmod.close()
		
		print('%i'%len(clpars),file=foutcl)
		for c in clpars:
			print('%8.5e'%c,file=foutcl)
		print("""Salt2ExtinctionLaw.version 1
Salt2ExtinctionLaw.min_lambda %i
Salt2ExtinctionLaw.max_lambda %i"""%(
	self.options.colorwaverange[0],
	self.options.colorwaverange[1]),file=foutcl)
		foutcl.close()

		#for c in foutclscat
		foutclscat.close()
		
		# best-fit and simulated SN params
		snfiles = np.genfromtxt(self.options.snlist,dtype='str')
		snfiles = np.atleast_1d(snfiles)
		
		foutsn = open('%s/salt3train_snparams.txt'%outdir,'w')
		print('# SN x0 x1 c t0 tpkoff SIM_x0 SIM_x1 SIM_c SIM_t0',file=foutsn)
		for k in SNParams.keys():
			foundfile = False
			for l in snfiles:
				if str(k) not in l: continue
				foundfile = True
				if '/' not in l:
					l = '%s/%s'%(os.path.dirname(self.options.snlist),l)
				sn = snana.SuperNova(l)
				sn.SNID = str(sn.SNID)
				if 'SIM_SALT2x0' in sn.__dict__.keys(): SIM_x0 = sn.SIM_SALT2x0
				else: SIM_x0 = -99
				if 'SIM_SALT2x1' in sn.__dict__.keys(): SIM_x1 = sn.SIM_SALT2x1
				else: SIM_x1 = -99
				if 'SIM_SALT2c' in sn.__dict__.keys(): SIM_c = sn.SIM_SALT2c
				else: SIM_c = -99
				if 'SIM_PEAKMJD' in sn.__dict__.keys(): SIM_PEAKMJD = float(sn.SIM_PEAKMJD.split()[0])
				else: SIM_PEAKMJD = -99
			if not foundfile: SIM_x0,SIM_x1,SIM_c,SIM_PEAKMJD = -99,-99,-99,-99

			print('%s %8.5e %.4f %.4f %.2f %.2f %8.5e %.4f %.4f %.2f'%(
				k,SNParams[k]['x0'],SNParams[k]['x1'],SNParams[k]['c'],SNParams[k]['t0'],
				SNParams[k]['tpkoff'],SIM_x0,SIM_x1,SIM_c,SIM_PEAKMJD),file=foutsn)
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
			if k == 'default': continue
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
				bandpassdict=kcordict,n_components=self.options.n_components)

		
	def main(self):

		if not self.options.kcor_path:
			raise RuntimeError('kcor_path variable must be defined!')
		self.kcordict=readutils.rdkcor(self.options.kcor_path,self.addwarning)
		# TODO: ASCII filter files
		
		# read the data
		datadict = readutils.rdAllData(self.options.snlist,self.options.estimate_tpk,self.addwarning,speclist=self.options.speclist)
		
		if not os.path.exists(self.options.outputdir):
			os.makedirs(self.options.outputdir)
		
		# Eliminate all data outside wave/phase range
		numSpecElimmed,numSpec=0,0
		numPhotElimmed,numPhot=0,0
		numSpecPoints=0
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
					numSpecPoints+=((specdata[k]['wavelength']/(1+z)>self.options.waverange[0])&(specdata[k]['wavelength']/(1+z)<self.options.waverange[1])).sum()
					
			#Remove photometric data outside phase range
			phase=(photdata['tobs'])/(1+z)
			def checkFilterMass(flt):
				survey = datadict[sn]['survey']
				filtwave = self.kcordict[survey]['filtwave']
				filttrans = self.kcordict[survey][flt]['filttrans']
			
				#Check how much mass of the filter is inside the wavelength range
				filtRange=(filtwave/(1+z)>self.options.waverange[0]) &(filtwave/(1+z) <self.options.waverange[1])
				return np.trapz((filttrans*filtwave/(1+z))[filtRange],filtwave[filtRange]/(1+z))/np.trapz(filttrans*filtwave/(1+z),filtwave/(1+z)) > 1-self.options.filter_mass_tolerance
					
			filterInBounds=np.vectorize(checkFilterMass)(photdata['filt'])
			phaseInBounds=(phase>self.options.phaserange[0]) & (phase<self.options.phaserange[1])
			keepPhot=filterInBounds&phaseInBounds
			numPhotElimmed+=(~keepPhot).sum()
			numPhot+=keepPhot.sum()
			datadict[sn]['photdata'] ={key:photdata[key][keepPhot] for key in photdata}
			
		print('{} spectra and {} photometric observations removed for being outside phase range'.format(numSpecElimmed,numPhotElimmed))
		print('{} spectra and {} photometric observations remaining'.format(numSpec,numPhot))
		print('{} total spectroscopic data points'.format(numSpecPoints))
		# fit the model - initial pass
		if self.options.stage == "all" or self.options.stage == "train":
			phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
				modelerr,clpars,clerr,clscat,SNParams,pars,parlist = self.fitSALTModel(
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
				self.options.n_min_specrecal,
				self.options.specrange_wavescale_specrecal,
				self.options.n_specrecal_per_lightcurve,
				self.options.filter_mass_tolerance,
				self.options.n_components,
				self.options.n_colorpars)
		
			# write the output model - M0, M1, c
			self.wrtoutput(self.options.outputdir,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,
						   modelerr,clpars,clerr,clscat,SNParams,
						   pars,parlist)

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

