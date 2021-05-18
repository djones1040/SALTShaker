import os
import argparse
import configparser
import numpy as np
from saltshaker.config import config_rootdir
from saltshaker.util.specSynPhot import getColorsForSN
from argparse import SUPPRESS
from saltshaker.training.priors import __priors__
import logging
log=logging.getLogger(__name__)


def expandvariablesandhomecommaseparated(paths):
	return ','.join([os.path.expanduser(os.path.expandvars(x)) for x in paths.split(',')])

class FullPaths(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		setattr(namespace, self.dest,expandvariablesandhomecommaseparated(values))

class EnvAwareArgumentParser(argparse.ArgumentParser):

	def add_argument(self,*args,**kwargs):
		if 'action' in kwargs:
			action=kwargs['action']
			
			if (action==FullPaths or action=='FullPaths') and 'default' in kwargs:
				kwargs['default']=expandvariablesandhomecommaseparated(kwargs['default'])
		return super().add_argument(*args,**kwargs)
		
class ConfigWithCommandLineOverrideParser(EnvAwareArgumentParser):
		
	def addhelp(self):
		default_prefix='-'
		self.add_argument(
				default_prefix+'h', default_prefix*2+'help',
				action='help', default=SUPPRESS,
				help=('show this help message and exit'))

	def add_argument_with_config_default(self,config,section,*keys,**kwargs):
		"""Given a ConfigParser and a section header, scans the section for matching keys and sets them as the default value for a command line argument added to self. If a default is provided, it is used if there is no matching key in the config, otherwise this method will raise a KeyError"""
		if 'clargformat' in kwargs:
			if kwargs['clargformat'] =='prependsection':
				kwargs['clargformat']='--{section}_{key}'
		else:
			kwargs['clargformat']='--{key}'
		
		clargformat=kwargs.pop('clargformat')

		clargs=[clargformat.format(section=section,key=key) for key in keys]
		def checkforflagsinconfig():
			for key in keys:
				if key in config[section]:
					return key,config[section][key]
			raise KeyError
		try:
			includedkey,kwargs['default']=checkforflagsinconfig()
		except KeyError:
			if 'default' in kwargs:
				pass
			else:
				raise KeyError(f'Key not found in {(config)}, section {section}; valid keys include: '+', '.join(keys))
		if 'nargs' in kwargs and ((type(kwargs['nargs']) is int and kwargs['nargs']>1) or (type(kwargs['nargs'] is str and (kwargs['nargs'] in ['+','*'])))):
			if not 'type' in kwargs:
				kwargs['default']=kwargs['default'].split(',')
			else:
				kwargs['default']=list(map(kwargs['type'],kwargs['default'].split(',')))
			if type(kwargs['nargs']) is int:
				try:
					assert(len(kwargs['default'])==kwargs['nargs'])
				except:
					nargs=kwargs['nargs']
					numfound=len(kwargs['default'])
					raise ValueError(f"Incorrect number of arguments in {(config)}, section {section}, key {includedkey}, {nargs} arguments required while {numfound} were found")
		return super().add_argument(*clargs,**kwargs)
		
def boolean_string(s):
	if s not in {'False', 'True', 'false', 'true', '1', '0'}:
		raise ValueError('Not a valid boolean string')
	return (s == 'True') | (s == '1') | (s == 'true')

def nonetype_or_int(s):
	if s == 'None': return None
	else: return int(s)

class SNCut:
	def __init__(self,description,requirement,valfunction):
		self.description=description
		self.requirement=requirement
		self.__valfunction__=valfunction
		
	def cutvalue(self,sn):
		return self.__valfunction__(sn)

	def passescut(self,sn):
		return self.cutvalue(sn)>=self.requirement

class TrainSALTBase:
	def __init__(self):
		self.verbose = False
		
		
	def add_user_options(self, parser=None, usage=None, config=None):
		if parser == None:
			parser = ConfigWithCommandLineOverrideParser(usage=usage, conflict_handler="resolve",add_help=False)

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
		parser.add_argument('--skip_validation', default=False, action="store_true",
							help='skip making validation plots')
		parser.add_argument('--fast', default=False, action="store_true",
							help='FAST option for debugging')

		
		# input files
		parser.add_argument_with_config_default(config,'iodata','calibrationshiftfile',	 type=str,action=FullPaths,default='',
							help='file containing a list of changes to zeropoint and central wavelength of filters by survey')
		parser.add_argument_with_config_default(config,'iodata','loggingconfig','loggingconfigfile',  type=str,action=FullPaths,default='',
							help='logging config file')
		parser.add_argument_with_config_default(config,'iodata','trainingconfig','trainingconfigfile','modelconfigfile',  type=str,action=FullPaths,
							help='Configuration file describing the construction of the model')
		parser.add_argument_with_config_default(config,'iodata','snlists','snlistfiles',  type=str,action=FullPaths,
							help="""list of SNANA-formatted SN data files, including both photometry and spectroscopy. Can be multiple comma-separated lists. (default=%(default)s)""")
		parser.add_argument_with_config_default(config,'iodata','snparlist', 'snparlistfile', type=str,action=FullPaths,
							help="""optional list of initial SN parameters.	 Needs columns SNID, zHelio, x0, x1, c""")
		parser.add_argument_with_config_default(config,'iodata','specrecallist','specrecallistfile',  type=str,action=FullPaths,
							help="""optional list giving number of spectral recalibration params.  Needs columns SNID, N, phase, ncalib where N is the spectrum number for a given SN, starting at 1""")
		parser.add_argument_with_config_default(config,'iodata','tmaxlist','tmaxlistfile',	type=str,action=FullPaths,
							help="""optional space-delimited list with SN ID, tmax, tmaxerr (default=%(default)s)""")
	
		#output files
		parser.add_argument_with_config_default(config,'iodata','outputdir',  type=str,action=FullPaths,
							help="""data directory for spectroscopy, format should be ASCII 
							with columns wavelength, flux, fluxerr (optional) (default=%(default)s)""")
		parser.add_argument_with_config_default(config,'iodata','yamloutputfile', default='/dev/null',type=str,action=FullPaths,
							help='File to which to output a summary of the fitting process')
		
		#options to configure cuts
		parser.add_argument_with_config_default(config,'iodata','dospec',  type=boolean_string,
							help="""if set, look for spectra in the snlist files (default=%(default)s)""")
		parser.add_argument_with_config_default(config,'iodata','maxsn',  type=nonetype_or_int,
							help="""sets maximum number of SNe to fit for debugging (default=%(default)s)""")
		parser.add_argument_with_config_default(config,'iodata','keeponlyspec',	 type=boolean_string,
							help="""if set, only train on SNe with spectra (default=%(default)s)""")
		parser.add_argument_with_config_default(config,'iodata','filter_mass_tolerance',  type=float,
							help='Mass of filter transmission allowed outside of model wavelength range (default=%(default)s)')

		#Initialize from SALT2.4		
		parser.add_argument_with_config_default(config,'iodata','initsalt2model',  type=boolean_string,
							help="""If true, initialize model parameters from prior SALT2 model""")
		parser.add_argument_with_config_default(config,'iodata','initsalt2var',	 type=boolean_string,
							help="""If true, initialize model uncertainty parameters from prior SALT2 model""")
		#Initialize from user defined files				
		parser.add_argument_with_config_default(config,'iodata','initm0modelfile',	type=str,action=FullPaths,
							help="""initial M0 model to begin training, ASCII with columns
							phase, wavelength, flux (default=%(default)s)""")
		parser.add_argument_with_config_default(config,'iodata','initm1modelfile',	type=str,action=FullPaths,
							help="""initial M1 model with x1=1 to begin training, ASCII with columns
							phase, wavelength, flux (default=%(default)s)""")
		#Choose B filter definition
		parser.add_argument_with_config_default(config,'iodata','initbfilt',  type=str,action=FullPaths,
							help="""initial B-filter to get the normalization of the initial model (default=%(default)s)""")
							
		parser.add_argument_with_config_default(config,'iodata','resume_from_outputdir',  type=str,action=FullPaths,
							help='if set, initialize using output parameters from previous run. If directory, initialize using ouptut parameters from specified directory')
		parser.add_argument_with_config_default(config,'iodata','resume_from_gnhistory',  type=str,action=FullPaths,
							help='if set, initialize using output parameters from previous run saved in gaussnewtonhistory.pickle file.')

		parser.add_argument_with_config_default(config,'iodata','fix_salt2modelpars',  type=boolean_string,
							help="""if set, fix M0/M1 for wavelength/phase range of original SALT2 model (default=%(default)s)""")
		#validation option
		parser.add_argument_with_config_default(config,'iodata','validate_modelonly',  type=boolean_string,
							help="""if set, only make model plots in the validation stage""")


		parser.add_argument_with_config_default(config,'trainparams','gaussnewton_maxiter',	 type=int,
							help='maximum iterations for Gauss-Newton (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainparams','regularize',	type=boolean_string,
							help='turn on regularization if set (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainparams','fitsalt2',  type=boolean_string,
							help='fit SALT2 as a validation check (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainparams','n_repeat',  type=int,
							help='repeat gauss newton n times (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainparams','fit_model_err',  type=boolean_string,
							help='fit for model error if set (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainparams','fit_cdisp_only',	type=boolean_string,
							help='fit for color dispersion component of model error if set (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainparams','steps_between_errorfit', type=int,
							help='fit for error model every x steps (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainparams','model_err_max_chisq', type=int,
							help='max photometric chi2/dof below which model error estimation is done (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainparams','fit_tpkoff',	type=boolean_string,
							help='fit for time of max in B-band if set (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainparams','fitting_sequence',  type=str,
							help="Order in which parameters are fit, 'default' or empty string does the standard approach, otherwise should be comma-separated list with any of the following: all, pcaparams, color, colorlaw, spectralrecalibration, sn, tpk (default=%(default)s)")


		# survey definitions
		self.surveylist = [section.replace('survey_','') for section in config.sections() if section.startswith('survey_')]
		for survey in self.surveylist:
			
			parser.add_argument_with_config_default(config,f'survey_{survey}',"kcorfile" ,type=str,clargformat=f"--{survey}" +"_{key}",action=FullPaths,
								help="kcor file for survey %s"%survey)
			parser.add_argument_with_config_default(config,f'survey_{survey}',"subsurveylist" ,type=str,clargformat=f"--{survey}" +"_{key}",
								help="comma-separated list of subsurveys for survey %s"%survey)

		return parser


	def add_training_options(self, parser=None, usage=None, config=None):
		if parser == None:
			parser = ConfigWithCommandLineOverrideParser(usage=usage, conflict_handler="resolve")

		# training params
		parser.add_argument_with_config_default(config,'trainingparams','specrecal',  type=int,
							help='number of parameters defining the spectral recalibration (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','n_processes',	type=int,
							help='number of processes to use in calculating chi2 (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','estimate_tpk',	 type=boolean_string,
							help='if set, estimate time of max with quick least squares fitting (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','fix_t0',  type=boolean_string,
							help='if set, don\'t allow time of max to float (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','regulargradientphase',	 type=float,
							help='Weighting of phase gradient chi^2 regularization during training of model parameters (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','regulargradientwave',	type=float,
							help='Weighting of wave gradient chi^2 regularization during training of model parameters (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','regulardyad',	type=float,
							help='Weighting of dyadic chi^2 regularization during training of model parameters (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','m1regularization',	 type=float,
							help='Scales regularization weighting of M1 component relative to M0 weighting (>1 increases smoothing of M1)  (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','spec_chi2_scaling',  type=float,
							help='scaling of spectral chi^2 so it doesn\'t dominate the total chi^2 (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','n_min_specrecal',	type=int,
							help='Minimum order of spectral recalibration polynomials (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','specrange_wavescale_specrecal',  type=float,
							help='Wavelength scale (in angstroms) for determining additional orders of spectral recalibration from wavelength range of spectrum (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','n_specrecal_per_lightcurve',  type=float,
							help='Number of additional spectral recalibration orders per lightcurve (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','regularizationScaleMethod',  type=str,
							help='Choose how scale for regularization is calculated (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','binspec',	type=boolean_string,
							help='bin the spectra if set (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','binspecres',  type=int,
							help='binning resolution (default=%(default)s)')
		
		#neff parameters
		parser.add_argument_with_config_default(config,'trainingparams','wavesmoothingneff',  type=float,
							help='Smooth effective # of spectral points along wave axis (in units of waveoutres) (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','phasesmoothingneff',  type=float,
							help='Smooth effective # of spectral points along phase axis (in units of phaseoutres) (default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','nefffloor',  type=float,
							help='Minimum number of effective points (has to be > 0 to prevent divide by zero errors).(default=%(default)s)')
		parser.add_argument_with_config_default(config,'trainingparams','neffmax',	type=float,
							help='Threshold for spectral coverage at which regularization will be turned off (default=%(default)s)')

		# training model parameters
		parser.add_argument_with_config_default(config,'modelparams','waverange', type=int, nargs=2,
							help='wavelength range over which the model is defined (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','colorwaverange',	type=int, nargs=2,
							help='wavelength range over which the color law is fit to data (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','interpfunc',	type=str,
							help='function to interpolate between control points in the fitting (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','errinterporder',	type=int,
							help='for model uncertainty splines/polynomial funcs, order of the function (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','interporder',	 type=int,
							help='for model splines/polynomial funcs, order of the function (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','wavesplineres',  type=float,
							help='number of angstroms between each wavelength spline knot (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','phasesplineres',	type=float,
							help='number of angstroms between each phase spline knot (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','waveinterpres',  type=float,
							help='wavelength resolution in angstroms, used for internal interpolation (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','phaseinterpres',	type=float,
							help='phase resolution in angstroms, used for internal interpolation (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','waveoutres',	type=float,
							help='wavelength resolution in angstroms of the output file (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','phaseoutres',	 type=float,
							help='phase resolution in angstroms of the output file (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','phaserange', type=int, nargs=2,
							help='phase range over which model is trained (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','n_components',  type=int,
							help='number of principal components of the SALT model to fit for (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','host_component',	type=str,
						    help="NOT IMPLEMENTED: if set, fit for a host component.  Must equal 'mass', for now (default=%(default)s)")
		parser.add_argument_with_config_default(config,'modelparams','n_colorpars',	 type=int,
							help='number of degrees of the phase-independent color law polynomial (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','n_colorscatpars',	 type=int,
							help='number of parameters in the broadband scatter model (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','error_snake_phase_binsize',  type=float,
							help='number of days over which to compute scaling of error model (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','error_snake_wave_binsize',  type=float,
							help='number of angstroms over which to compute scaling of error model (default=%(default)s)')
		parser.add_argument_with_config_default(config,'modelparams','use_snpca_knots',	 type=boolean_string,
							help='if set, define model on SNPCA knots (default=%(default)s)')		

		# priors
		for prior in __priors__:
			parser.add_argument_with_config_default(config,'priors',prior ,type=float,clargformat="--prior_{key}",
								help=f"prior on {prior}",default=SUPPRESS)

		# bounds
		for bound,val in config.items('bounds'):
			parser.add_argument_with_config_default(config,'bounds', bound, type=float,nargs=3,clargformat="--bound_{key}",
								help="bound on %s"%bound)

		return parser

	def get_saltkw(self,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc):


		saltfitkwargs = {'m1regularization':self.options.m1regularization,'bsorder':self.options.interporder,'errbsorder':self.options.errinterporder,
						 'waveSmoothingNeff':self.options.wavesmoothingneff,'phaseSmoothingNeff':self.options.phasesmoothingneff,
						 'neffFloor':self.options.nefffloor, 'neffMax':self.options.neffmax,
						 'specrecal':self.options.specrecal, 'regularizationScaleMethod':self.options.regularizationScaleMethod,
						 'phaseinterpres':self.options.phaseinterpres,'waveinterpres':self.options.waveinterpres,
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
						 'n_components':self.options.n_components,
                         'host_component':self.options.host_component,
                         'n_colorpars':self.options.n_colorpars,
						 'n_colorscatpars':self.options.n_colorscatpars,
						 'fix_t0':self.options.fix_t0,
						 'regularize':self.options.regularize,
						 'outputdir':self.options.outputdir,
						 'fit_model_err':self.options.fit_model_err,
						 'fit_cdisp_only':self.options.fit_cdisp_only,
						 'model_err_max_chisq':self.options.model_err_max_chisq,
						 'steps_between_errorfit':self.options.steps_between_errorfit,
						 'fitTpkOff':self.options.fit_tpkoff,
						 'spec_chi2_scaling':self.options.spec_chi2_scaling}
		
		for k in self.options.__dict__.keys():
			if k.startswith('prior') or k.startswith('bound'):
				saltfitkwargs[k] = self.options.__dict__[k]
		return saltfitkwargs
	
	def snshouldbecut(self,sn,cuts):
	
		passescuts= [cut.passescut(sn) for cut in cuts]
		
		if not all(passescuts): # or iPreMaxCut < 2 or medSNR < 10:
			
			log.debug('SN %s fails cuts'%sn)
			log.debug(', '.join([f'{cut.cutvalue(sn)} {cut.description} ({cut.requirement} needed)' for cut in cuts]))
		return not all(passescuts)

	def checkFilterMass(self,z,survey,flt):

		filtwave = self.kcordict[survey][flt]['filtwave']
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
	
	def getcuts(self):
		def checkfitprob(sn):
			hasvalidfitprob=sn.salt2fitprob!=-99
			if	hasvalidfitprob:
				return sn.salt2fitprob
			else:
				log.warning(f'SN {sn.snid} does not have a valid salt2 fitprob, including in sample')
				return 1
			
		cuts= [SNCut('total epochs',4,lambda sn: sum([ ((sn.photdata[flt].phase > -10) & (sn.photdata[flt].phase < 35)).sum() for flt in sn.photdata])),
		SNCut('epochs near peak',1,lambda sn: sum([ ((sn.photdata[flt].phase > -10) & (sn.photdata[flt].phase < 5)).sum() for flt in sn.photdata])),
		SNCut('epochs post peak',1,lambda sn: sum([	 ((sn.photdata[flt].phase > 5) & (sn.photdata[flt].phase < 20)).sum() for flt in sn.photdata])),
		SNCut('filters near peak',2,lambda sn: sum([ (((sn.photdata[flt].phase > -8) & (sn.photdata[flt].phase < 10)).sum())>0 for flt in sn.photdata])),
		SNCut('salt2 fitprob',1e-4,	 checkfitprob)]
		if self.options.keeponlyspec:
			cuts+=[ SNCut('spectra', 1, lambda sn: sn.num_spec)]
		return cuts
	def mkcuts(self,datadict):
		# cuts
		# 4 epochs at -10 < phase < 35
		# 1 measurement near peak
		# 1 measurement at 5 < t < 20
		# 2 measurements at -8 < t < 10
		# salt2fitprob >1e-4
		# if set by command line flag, 1 spectra
		
		#Define cuts
		cuts=self.getcuts()
		
		#Record initial demographics of the sample
		sumattr=lambda x,sndict: len(sndict) if x =='num_sn' else sum([getattr(sndict[snid],x) for snid in sndict])
		descriptionandattrs=[('photometric observations','num_photobs'),('spectroscopic observations','num_specobs'),('light-curves','num_lc'),('spectra','num_spec')]
		descriptions,attrs=zip(*descriptionandattrs)
		
		initialdemos=[sumattr(attr,datadict) for attr in attrs]
		outdict={}
		cutdict={}

		for snid in datadict:
			sn=datadict[snid]
			photdata = sn.photdata
			specdata = sn.specdata
			z = sn.zHelio
				
			for k in list(specdata.keys()):
				#Remove spectra outside phase range
				spectrum=specdata[k]
				if spectrum.phase<self.options.phaserange[0] or \
				   spectrum.phase>self.options.phaserange[1]-3:
					specdata.pop(k)
					continue

				#remove spectral data outside wavelength range
				inwaverange=(spectrum.wavelength>(self.options.waverange[0]*(1+z)))&(spectrum.wavelength<(self.options.waverange[1]*(1+z)))
				clippedspectrum=spectrum.clip(inwaverange)
				if len(clippedspectrum):
					specdata[k]=clippedspectrum
				else:
					specdata.pop(k)

			for flt in sn.filt:
				#Remove light-curves outside wavelength range
				if self.checkFilterMass(z,sn.survey,flt):
					lightcurve=sn.photdata[flt]
					#Remove photometric data outside phase range
					inphaserange=(lightcurve.phase>self.options.phaserange[0]) & (lightcurve.phase<self.options.phaserange[1])
					clippedlightcurve=lightcurve.clip(inphaserange)
					if len(clippedlightcurve):
						photdata[flt]=clippedlightcurve
					else:
						photdata.pop(flt)
				else:
					sn.photdata.pop(flt)
			#Check if SN passes all cuts
			if self.snshouldbecut(sn,cuts):
				cutdict[snid]=sn
			else:
				outdict[snid]=sn
			
		finaldemos =[sumattr(attr,outdict) for attr in attrs]
		sncutdemos =[sumattr(attr,cutdict) for attr in attrs]
		for attr,desc,initial,final,cut in zip(attrs,descriptions,initialdemos,finaldemos,sncutdemos):
			log.info(f'{initial-(final+cut)} {desc} removed as a result of cuts for phase and wavelength range')
		
		log.info(f'{len(datadict)} SNe initially, {len(cutdict)} SNe cut from the sample')
		log.info('Total number of supernovae: {}'.format(len(outdict)))
		log.info( ', '.join([f'{final} {desc}' for attr,desc,initial,final,cut in zip(attrs,descriptions,initialdemos,finaldemos,sncutdemos)])+' remaining after all cuts')

		return outdict,cutdict
