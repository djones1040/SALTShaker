import os
import argparse
import configparser
import numpy as np
from salt3.config import config_rootdir
from salt3.util.specSynPhot import getColorsForSN

import logging
log=logging.getLogger(__name__)

def boolean_string(s):
	if s not in {'False', 'True', 'false', 'true', '1', '0'}:
		raise ValueError('Not a valid boolean string')
	return (s == 'True') | (s == '1') | (s == 'true')

def nonetype_or_int(s):
	if s == 'None': return None
	else: return int(s)


class TrainSALTBase:
	def __init__(self):
		self.verbose = False
		

	def add_user_options(self, parser=None, usage=None, config=None):
		if parser == None:
			parser = argparse.ArgumentParser(usage=usage, conflict_handler="resolve")

		# The basics
		parser.add_argument('-v', '--verbose', action="count", dest="verbose",
							default=0,help='verbosity level')
		parser.add_argument('--debug', default=False, action="store_true",
							help='debug mode: more output and debug files')
		parser.add_argument('--clobber', default=False, action="store_true",
							help='clobber')
		parser.add_argument('-c','--configfile', default=None, type=str,
							help='configuration file')
		parser.add_argument('-s','--stage', default='all', type=str,
							help='stage - options are train and validate')

		
		# input/output files
		parser.add_argument('--loggingconfig', default=config.get('iodata','loggingconfig'), type=str,
							help='logging config file')
		parser.add_argument('--trainingconfig', default=config.get('iodata','trainingconfig'), type=str,
							help='training config file')
		parser.add_argument('--snlists', default=config.get('iodata','snlists'), type=str,
							help="""list of SNANA-formatted SN data files, including both photometry and spectroscopy. Can be multiple comma-separated lists. (default=%default)""")
		parser.add_argument('--snparlist', default=config.get('iodata','snparlist'), type=str,
							help="""optional list of initial SN parameters.  Needs columns SNID, zHelio, x0, x1, c""")
		parser.add_argument('--specrecallist', default=config.get('iodata','specrecallist'), type=str,
							help="""optional list giving number of spectral recalibration params.  Needs columns SNID, N, phase, ncalib where N is the spectrum number for a given SN, starting at 1""")
		parser.add_argument('--tmaxlist', default=config.get('iodata','tmaxlist'), type=str,
							help="""optional space-delimited list with SN ID, tmax, tmaxerr (default=%default)""")
		parser.add_argument('--dospec', default=config.get('iodata','dospec'), type=boolean_string,
							help="""if set, look for spectra in the snlist files (default=%default)""")
		parser.add_argument('--maxsn', default=config.get('iodata','maxsn'), type=nonetype_or_int,
							help="""sets maximum number of SNe to fit for debugging (default=%default)""")
		parser.add_argument('--keeponlyspec', default=config.get('iodata','keeponlyspec'), type=boolean_string,
							help="""if set, only train on SNe with spectra (default=%default)""")
		parser.add_argument('--outputdir', default=config.get('iodata','outputdir'), type=str,
							help="""data directory for spectroscopy, format should be ASCII 
							with columns wavelength, flux, fluxerr (optional) (default=%default)""")
		parser.add_argument('--initsalt2model', default=config.get('iodata','initsalt2model'), type=boolean_string,
							help="""If true, initialize model parameters from prior SALT2 model""")
		parser.add_argument('--initsalt2var', default=config.get('iodata','initsalt2var'), type=boolean_string,
							help="""If true, initialize model uncertainty parameters from prior SALT2 model""")
		parser.add_argument('--initm0modelfile', default=config.get('iodata','initm0modelfile'), type=str,
							help="""initial M0 model to begin training, ASCII with columns
							phase, wavelength, flux (default=%default)""")
		parser.add_argument('--initm1modelfile', default=config.get('iodata','initm1modelfile'), type=str,
							help="""initial M1 model with x1=1 to begin training, ASCII with columns
							phase, wavelength, flux (default=%default)""")
		parser.add_argument('--initbfilt', default=config.get('iodata','initbfilt'), type=str,
							help="""initial B-filter to get the normalization of the initial model (default=%default)""")
		parser.add_argument('--resume_from_outputdir', default=config.get('iodata','resume_from_outputdir'), type=str,
							help='if set, initialize using output parameters from previous run. If directory, initialize using ouptut parameters from specified directory')
		parser.add_argument('--filter_mass_tolerance', default=config.get('iodata','filter_mass_tolerance'), type=float,
							help='Mass of filter transmission allowed outside of model wavelength range (default=%default)')
		parser.add_argument('--trainingconfig', default=config.get('iodata','trainingconfig'), type=str,
							help='config file for the detailed training params')
		parser.add_argument('--fix_salt2modelpars', default=config.get('iodata','fix_salt2modelpars'), type=boolean_string,
							help="""if set, fix M0/M1 for wavelength/phase range of original SALT2 model (default=%default)""")


		parser.add_argument('--do_mcmc', default=config.get('trainparams','do_mcmc'), type=boolean_string,
							help='do MCMC fitting (default=%default)')
		parser.add_argument('--do_gaussnewton', default=config.get('trainparams','do_gaussnewton'), type=boolean_string,
							help='do Gauss-Newton least squares (default=%default)')
		parser.add_argument('--gaussnewton_maxiter', default=config.get('trainparams','gaussnewton_maxiter'), type=int,
							help='maximum iterations for Gauss-Newton (default=%default)')
		parser.add_argument('--regularize', default=config.get('trainparams','regularize'), type=boolean_string,
							help='turn on regularization if set (default=%default)')
		parser.add_argument('--fitsalt2', default=config.get('trainparams','fitsalt2'), type=boolean_string,
							help='fit SALT2 as a validation check (default=%default)')
		parser.add_argument('--n_repeat', default=config.get('trainparams','n_repeat'), type=int,
							help='repeat mcmc and/or gauss newton n times (default=%default)')
		parser.add_argument('--fit_model_err', default=config.get('trainparams','fit_model_err'), type=boolean_string,
							help='fit for model error if set (default=%default)')
		parser.add_argument('--fit_cdisp_only', default=config.get('trainparams','fit_cdisp_only'), type=boolean_string,
							help='fit for color dispersion component of model error if set (default=%default)')
		parser.add_argument('--fit_tpkoff', default=config.get('trainparams','fit_tpkoff'), type=boolean_string,
							help='fit for time of max in B-band if set (default=%default)')
		parser.add_argument('--fitting_sequence', default=config.get('trainparams','fitting_sequence'), type=str,
							help="Order in which parameters are fit, 'default' or empty string does the standard approach, otherwise should be comma-separated list with any of the following: all, pcaparams, color, colorlaw, spectralrecalibration, sn, tpk (default=%default)")

		# mcmc parameters
		parser.add_argument('--n_steps_mcmc', default=config.get('mcmcparams','n_steps_mcmc'), type=int,
							help='number of accepted MCMC steps (default=%default)')
		parser.add_argument('--n_burnin_mcmc', default=config.get('mcmcparams','n_burnin_mcmc'), type=int,
							help='number of burn-in MCMC steps	(default=%default)')


		# survey definitions
		self.surveylist = []
		for survey in config.sections():
			if not survey.startswith('survey_'): continue
			
			parser.add_argument("--%s_kcorfile"%survey.replace('survey_',''),default=config.get(survey,'kcorfile'),type=str,
								help="kcor file for survey %s"%survey)
			parser.add_argument("--%s_subsurveylist"%survey.replace('survey_',''),default=config.get(survey,'subsurveylist'),type=str,
								help="comma-separated list of subsurveys for survey %s"%survey)
			self.surveylist += [survey.replace('survey_','')]

		return parser


	def add_training_options(self, parser=None, usage=None, config=None):
		if parser == None:
			parser = argparse.ArgumentParser(usage=usage, conflict_handler="resolve")

		# training params
		parser.add_argument('--specrecal', default=config.get('trainingparams','specrecal'), type=int,
							help='number of parameters defining the spectral recalibration (default=%default)')
		parser.add_argument('--n_processes', default=config.get('trainingparams','n_processes'), type=int,
							help='number of processes to use in calculating chi2 (default=%default)')
		parser.add_argument('--estimate_tpk', default=config.get('trainingparams','estimate_tpk'), type=boolean_string,
							help='if set, estimate time of max with quick least squares fitting (default=%default)')
		parser.add_argument('--fix_t0', default=config.get('trainingparams','fix_t0'), type=boolean_string,
							help='if set, don\'t allow time of max to float (default=%default)')
		parser.add_argument('--regulargradientphase', default=config.get('trainingparams','regulargradientphase'), type=float,
							help='Weighting of phase gradient chi^2 regularization during training of model parameters (default=%default)')
		parser.add_argument('--regulargradientwave', default=config.get('trainingparams','regulargradientwave'), type=float,
							help='Weighting of wave gradient chi^2 regularization during training of model parameters (default=%default)')
		parser.add_argument('--regulardyad', default=config.get('trainingparams','regulardyad'), type=float,
							help='Weighting of dyadic chi^2 regularization during training of model parameters (default=%default)')
		parser.add_argument('--m1regularization', default=config.get('trainingparams','m1regularization'), type=float,
							help='Scales regularization weighting of M1 component relative to M0 weighting (>1 increases smoothing of M1)  (default=%default)')
		parser.add_argument('--spec_chi2_scaling', default=config.get('trainingparams','spec_chi2_scaling'), type=float,
							help='scaling of spectral chi^2 so it doesn\'t dominate the total chi^2 (default=%default)')
		parser.add_argument('--n_min_specrecal', default=config.get('trainingparams','n_min_specrecal'), type=int,
							help='Minimum order of spectral recalibration polynomials (default=%default)')
		parser.add_argument('--specrange_wavescale_specrecal', default=config.get('trainingparams','specrange_wavescale_specrecal'), type=float,
							help='Wavelength scale (in angstroms) for determining additional orders of spectral recalibration from wavelength range of spectrum (default=%default)')
		parser.add_argument('--n_specrecal_per_lightcurve', default=config.get('trainingparams','n_specrecal_per_lightcurve'), type=float,
							help='Number of additional spectral recalibration orders per lightcurve (default=%default)')
		parser.add_argument('--regularizationScaleMethod', default=config.get('trainingparams','regularizationScaleMethod'), type=str,
							help='Choose how scale for regularization is calculated (default=%default)')
		parser.add_argument('--binspec', default=config.get('trainingparams','binspec'), type=boolean_string,
							help='bin the spectra if set (default=%default)')
		parser.add_argument('--binspecres', default=config.get('trainingparams','binspecres'), type=int,
							help='binning resolution (default=%default)')
		
   		#neff parameters
		parser.add_argument('--wavesmoothingneff', default=config.get('trainingparams','wavesmoothingneff'), type=float,
							help='Smooth effective # of spectral points along wave axis (in units of waveoutres) (default=%default)')
		parser.add_argument('--phasesmoothingneff', default=config.get('trainingparams','phasesmoothingneff'), type=float,
							help='Smooth effective # of spectral points along phase axis (in units of phaseoutres) (default=%default)')
		parser.add_argument('--nefffloor', default=config.get('trainingparams','nefffloor'), type=float,
							help='Minimum number of effective points (has to be > 0 to prevent divide by zero errors).(default=%default)')
		parser.add_argument('--neffmax', default=config.get('trainingparams','neffmax'), type=float,
							help='Threshold for spectral coverage at which regularization will be turned off (default=%default)')

		# training model parameters
		parser.add_argument('--waverange', default=list(map(int,config.get('modelparams','waverange').split(','))), type=int, nargs=2,
							help='wavelength range over which the model is defined (default=%default)')
		parser.add_argument('--colorwaverange', default=list(map(int,config.get('modelparams','colorwaverange').split(','))), type=int, nargs=2,
							help='wavelength range over which the color law is fit to data (default=%default)')
		parser.add_argument('--interpfunc', default=config.get('modelparams','interpfunc'), type=str,
							help='function to interpolate between control points in the fitting (default=%default)')
		parser.add_argument('--errinterporder', default=config.get('modelparams','errinterporder'), type=int,
							help='for model uncertainty splines/polynomial funcs, order of the function (default=%default)')
		parser.add_argument('--interporder', default=config.get('modelparams','interporder'), type=int,
							help='for model splines/polynomial funcs, order of the function (default=%default)')
		parser.add_argument('--wavesplineres', default=config.get('modelparams','wavesplineres'), type=float,
							help='number of angstroms between each wavelength spline knot (default=%default)')
		parser.add_argument('--phasesplineres', default=config.get('modelparams','phasesplineres'), type=float,
							help='number of angstroms between each phase spline knot (default=%default)')
		parser.add_argument('--waveoutres', default=config.get('modelparams','waveoutres'), type=float,
							help='wavelength resolution in angstroms of the output file (default=%default)')
		parser.add_argument('--phaseoutres', default=config.get('modelparams','phaseoutres'), type=float,
							help='phase resolution in angstroms of the output file (default=%default)')
		parser.add_argument('--phaserange', default=list(map(int,config.get('modelparams','phaserange').split(','))), type=int, nargs=2,
							help='phase range over which model is trained (default=%default)')
		parser.add_argument('--n_components', default=config.get('modelparams','n_components'), type=int,
							help='number of principal components of the SALT model to fit for (default=%default)')
		parser.add_argument('--n_colorpars', default=config.get('modelparams','n_colorpars'), type=int,
							help='number of degrees of the phase-independent color law polynomial (default=%default)')
		parser.add_argument('--n_colorscatpars', default=config.get('modelparams','n_colorscatpars'), type=int,
							help='number of parameters in the broadband scatter model (default=%default)')
		parser.add_argument('--error_snake_phase_binsize', default=config.get('modelparams','error_snake_phase_binsize'), type=float,
							help='number of days over which to compute scaling of error model (default=%default)')
		parser.add_argument('--error_snake_wave_binsize', default=config.get('modelparams','error_snake_wave_binsize'), type=float,
							help='number of angstroms over which to compute scaling of error model (default=%default)')
		
		
		# mcmc parameters
		parser.add_argument('--stepsize_magscale_M0', default=config.get('mcmcparams','stepsize_magscale_M0'), type=float,
							help='initial MCMC step size for M0, in mag	 (default=%default)')
		parser.add_argument('--stepsize_magadd_M0', default=config.get('mcmcparams','stepsize_magadd_M0'), type=float,
							help='initial MCMC step size for M0, in mag	 (default=%default)')
		parser.add_argument('--stepsize_magscale_err', default=config.get('mcmcparams','stepsize_magscale_err'), type=float,
							help='initial MCMC step size for the model err spline knots, in mag  (default=%default)')
		parser.add_argument('--stepsize_errcorr', default=config.get('mcmcparams','stepsize_errcorr'), type=float,
							help='initial MCMC step size for the correlation between model error terms, in mag  (default=%default)')
		parser.add_argument('--stepsize_magscale_M1', default=config.get('mcmcparams','stepsize_magscale_M1'), type=float,
							help='initial MCMC step size for M1, in mag - need both mag and flux steps because M1 can be negative (default=%default)')
		parser.add_argument('--stepsize_magadd_M1', default=config.get('mcmcparams','stepsize_magadd_M1'), type=float,
							help='initial MCMC step size for M1, in flux - need both mag and flux steps because M1 can be negative (default=%default)')
		parser.add_argument('--stepsize_cl', default=config.get('mcmcparams','stepsize_cl'), type=float,
							help='initial MCMC step size for color law	(default=%default)')
		parser.add_argument('--stepsize_magscale_clscat', default=config.get('mcmcparams','stepsize_magscale_clscat'), type=float,
							help='initial MCMC step size for color law	(default=%default)')
		parser.add_argument('--stepsize_specrecal', default=config.get('mcmcparams','stepsize_specrecal'), type=float,
							help='initial MCMC step size for spec recal. params	 (default=%default)')
		parser.add_argument('--stepsize_x0', default=config.get('mcmcparams','stepsize_x0'), type=float,
							help='initial MCMC step size for x0, in mag	 (default=%default)')
		parser.add_argument('--stepsize_x1', default=config.get('mcmcparams','stepsize_x1'), type=float,
							help='initial MCMC step size for x1	 (default=%default)')
		parser.add_argument('--stepsize_c', default=config.get('mcmcparams','stepsize_c'), type=float,
							help='initial MCMC step size for c	(default=%default)')
		parser.add_argument('--stepsize_tpk', default=config.get('mcmcparams','stepsize_tpk'), type=float,
							help='initial MCMC step size for tpk  (default=%default)')

		# adaptive MCMC parameters
		parser.add_argument('--nsteps_before_adaptive', default=config.get('mcmcparams','nsteps_before_adaptive'), type=float,
							help='number of steps before starting adaptive step sizes (default=%default)')
		parser.add_argument('--nsteps_adaptive_memory', default=config.get('mcmcparams','nsteps_adaptive_memory'), type=float,
							help='number of steps to use to estimate adaptive steps (default=%default)')
		parser.add_argument('--modelpar_snpar_tradeoff_nstep', default=config.get('mcmcparams','modelpar_snpar_tradeoff_nstep'), type=float,
							help='number of steps when trading between adjusting model params and SN params (default=%default)')
		parser.add_argument('--nsteps_before_modelpar_tradeoff', default=config.get('mcmcparams','nsteps_before_modelpar_tradeoff'), type=float,
							help='number of steps when trading between adjusting model params and SN params (default=%default)')
		parser.add_argument('--nsteps_between_lsqfit', default=config.get('mcmcparams','nsteps_between_lsqfit'), type=float,
							help='every x number of steps, adjust the SN params via least squares fitting (default=%default)')
		parser.add_argument('--use_lsqfit', default=config.get('mcmcparams','use_lsqfit'), type=boolean_string,
							help='if set, periodically adjust the SN params via least squares fitting (default=%default)')
		parser.add_argument('--adaptive_sigma_opt_scale', default=config.get('mcmcparams','adaptive_sigma_opt_scale'), type=float,
							help='scaling the adaptive step sizes (default=%default)')
		parser.add_argument('--use_snpca_knots', default=config.get('modelparams','use_snpca_knots'), type=boolean_string,
							help='if set, define model on SNPCA knots (default=%default)')

		# priors
		for prior,val in config.items('priors'):
			parser.add_argument("--prior_%s"%prior,default=val,type=float,
								help="prior on %s"%prior)

		# bounds
		for bound,val in config.items('bounds'):
			parser.add_argument("--bound_%s"%bound,default=val.split(','),type=float,nargs=3,
								help="bound on %s"%bound)
		#import pdb; pdb.set_trace()

		return parser

	def get_saltkw(self,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc):


		saltfitkwargs = {'m1regularization':self.options.m1regularization,'bsorder':self.options.interporder,'errbsorder':self.options.errinterporder,
						 'waveSmoothingNeff':self.options.wavesmoothingneff,'phaseSmoothingNeff':self.options.phasesmoothingneff,
						 'neffFloor':self.options.nefffloor, 'neffMax':self.options.neffmax,
						 'specrecal':self.options.specrecal, 'regularizationScaleMethod':self.options.regularizationScaleMethod,
						 'phaseknotloc':phaseknotloc,'waveknotloc':waveknotloc,
						 'errphaseknotloc':errphaseknotloc,'errwaveknotloc':errwaveknotloc,
						 'phaserange':self.options.phaserange,
						 'waverange':self.options.waverange,'phaseres':self.options.phasesplineres,
						 'waveres':self.options.wavesplineres,'phaseoutres':self.options.phaseoutres,
						 'waveoutres':self.options.waveoutres,
						 'colorwaverange':self.options.colorwaverange,
						 'kcordict':self.kcordict,'initm0modelfile':self.options.initm0modelfile,
						 'initbfilt':self.options.initbfilt,'regulargradientphase':self.options.regulargradientphase,
						 'regulargradientwave':self.options.regulargradientwave,'regulardyad':self.options.regulardyad,
						 'filter_mass_tolerance':self.options.filter_mass_tolerance,
						 'specrange_wavescale_specrecal':self.options.specrange_wavescale_specrecal,
						 'n_components':self.options.n_components,'n_colorpars':self.options.n_colorpars,
						 'n_colorscatpars':self.options.n_colorscatpars,
						 'nsteps_before_adaptive':self.options.nsteps_before_adaptive,
						 'nsteps_adaptive_memory':self.options.nsteps_adaptive_memory,
						 'adaptive_sigma_opt_scale':self.options.adaptive_sigma_opt_scale,
						 'stepsize_magscale_M0':self.options.stepsize_magscale_M0,
						 'stepsize_magadd_M0':self.options.stepsize_magadd_M0,
						 'stepsize_magscale_err':self.options.stepsize_magscale_err,
						 'stepsize_errcorr':self.options.stepsize_errcorr,
						 'stepsize_magscale_M1':self.options.stepsize_magscale_M1,
						 'stepsize_magadd_M1':self.options.stepsize_magadd_M1,
						 'stepsize_cl':self.options.stepsize_cl,
						 'stepsize_magscale_clscat':self.options.stepsize_magscale_clscat,
						 'stepsize_specrecal':self.options.stepsize_specrecal,
						 'stepsize_x0':self.options.stepsize_x0,
						 'stepsize_x1':self.options.stepsize_x1,
						 'stepsize_c':self.options.stepsize_c,
						 'stepsize_tpk':self.options.stepsize_tpk,
						 'fix_t0':self.options.fix_t0,
						 'nsteps_before_modelpar_tradeoff':self.options.nsteps_before_modelpar_tradeoff,
						 'modelpar_snpar_tradeoff_nstep':self.options.modelpar_snpar_tradeoff_nstep,
						 'nsteps_between_lsqfit':self.options.nsteps_between_lsqfit,
						 'use_lsqfit':self.options.use_lsqfit,
						 'regularize':self.options.regularize,
						 'outputdir':self.options.outputdir,
						 'fit_model_err':self.options.fit_model_err,
						 'fit_cdisp_only':self.options.fit_cdisp_only,
						 'fitTpkOff':self.options.fit_tpkoff,
						 'spec_chi2_scaling':self.options.spec_chi2_scaling}
		
		for k in self.options.__dict__.keys():
			if k.startswith('prior') or k.startswith('bound'):
				saltfitkwargs[k] = self.options.__dict__[k]
		return saltfitkwargs

	def mkcuts(self,datadict,KeepOnlySpec=False):

		# Eliminate all data outside wave/phase range
		numSpecElimmed,numSpec=0,0
		numPhotElimmed,numPhot=0,0
		numSpecPoints=0
		failedlist = []
		log.info('hack! no spec color cut')
		#log.info('hack!  no cuts at all')
		#return datadict
		for sn in list(datadict.keys()):
			photdata = datadict[sn]['photdata']
			specdata = datadict[sn]['specdata']
			z = datadict[sn]['zHelio']

			# cuts
			# 4 epochs at -10 < phase < 35
			# 1 measurement near peak
			# 1 measurement at 5 < t < 20
			# 2 measurements at -8 < t < 10
			phase = (photdata['mjd'] - datadict[sn]['tpk'])/(1+z)
			iEpochsCut = np.where((phase > -10) & (phase < 35))[0]
			iPkCut = np.where((phase > -10) & (phase < 5))[0]
			iShapeCut = np.where((phase > 5) & (phase < 20))[0]
			iColorCut = np.where((phase > -8) & (phase < 10))[0]
			NFiltColorCut = len(np.unique(photdata['filt'][iColorCut]))
			iPreMaxCut = len(np.unique(photdata['filt'][np.where((phase > -10) & (phase < -2))[0]]))
			medSNR = np.median(photdata['fluxcal'][(phase > -10) & (phase < 10)]/photdata['fluxcalerr'][(phase > -10) & (phase < 10)])
			hasvalidfitprob=datadict[sn]['fitprob']!=-99
			iFitprob = (datadict[sn]['fitprob'] >= 1e-4)
			if not iFitprob:
				log.debug(f'SN {sn} failing fitprob cut!')
			if not hasvalidfitprob:
				log.warning(f'SN {sn} does not have a valid fitprob, including in sample')
				iFitprob=True
			if len(iEpochsCut) < 4 or not len(iPkCut) or not len(iShapeCut) or NFiltColorCut < 2 or not iFitprob: # or iPreMaxCut < 2 or medSNR < 10:
				datadict.pop(sn)
				failedlist += [sn]
				log.debug('SN %s fails cuts'%sn)
				log.debug('%i epochs, %i epochs near peak, %i epochs post-peak, %i filters near peak'%(
						len(iEpochsCut),len(iPkCut),len(iShapeCut),NFiltColorCut))
				continue

			#Remove spectra outside phase range
			for k in list(specdata.keys()):
				# remove spectra with bad colors
				colordiffs = getColorsForSN(
					specdata[k],photdata,self.kcordict,datadict[sn]['survey'])
				
				#if colordiffs is None or len(colordiffs[np.abs(colordiffs) > 0.1]):
				#	specdata.pop(k)
				#	numSpecElimmed += 1
				if ((specdata[k]['tobs'])/(1+z)<self.options.phaserange[0]) or \
				   ((specdata[k]['tobs'])/(1+z)>self.options.phaserange[1]-3):
					specdata.pop(k)
					numSpecElimmed+=1
				elif specdata[k]['mjd'] < np.min(photdata['mjd']) or \
					 specdata[k]['mjd'] > np.max(photdata['mjd']):
					specdata.pop(k)
					numSpecElimmed+=1
				else:
					numSpec+=1
					numSpecPoints+=((specdata[k]['wavelength']/(1+z)>self.options.waverange[0]) &
									(specdata[k]['wavelength']/(1+z)<self.options.waverange[1])).sum()
			if KeepOnlySpec and not len(specdata.keys()):
				datadict.pop(sn)
				continue
			
			#Remove photometric data outside phase range
			phase=(photdata['tobs'])/(1+z)
			def checkFilterMass(flt):
				survey = datadict[sn]['survey']
				filtwave = self.kcordict[survey]['filtwave']
				try:
					filttrans = self.kcordict[survey][flt]['filttrans']
				except:
					raise RuntimeError('filter %s not found in kcor file for SN %s'%(flt,sn))
					
				#Check how much mass of the filter is inside the wavelength range
				filtRange=(filtwave/(1+z) > self.options.waverange[0]) & \
						   (filtwave/(1+z) < self.options.waverange[1])
				return np.trapz((filttrans*filtwave/(1+z))[filtRange],
								filtwave[filtRange]/(1+z))/np.trapz(
									filttrans*filtwave/(1+z),
									filtwave/(1+z)) > 1-self.options.filter_mass_tolerance

			filterInBounds=np.vectorize(checkFilterMass)(photdata['filt'])
			phaseInBounds=(phase>self.options.phaserange[0]) & (phase<self.options.phaserange[1])
			keepPhot=filterInBounds&phaseInBounds
			numPhotElimmed+=(~keepPhot).sum()
			numPhot+=keepPhot.sum()
			datadict[sn]['photdata'] ={key:photdata[key][keepPhot] for key in photdata}

		if self.options.maxsn is not None:
			surveys = np.unique([datadict[k]['survey'].split('(')[0] for k in datadict.keys()])
			for s in surveys:
				count = 0
				for i,sn in enumerate(list(datadict.keys())):
					if datadict[sn]['survey'].split('(')[0] != s: continue
					if count >= self.options.maxsn/len(surveys):
						datadict.pop(sn)
					count += 1

		log.info('{} spectra and {} photometric observations removed for being outside phase range'.format(numSpecElimmed,numPhotElimmed))
		log.info('{} spectra and {} photometric observations remaining'.format(numSpec,numPhot))
		log.info('{} total spectroscopic data points'.format(numSpecPoints))
		log.info('Total number of supernovae: {}'.format(len(datadict)))
		return datadict
