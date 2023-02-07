import os,sys
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
                        raise KeyError(f'key {key} not found in section {section} of config file')

                try:
                        includedkey,kwargs['default']=checkforflagsinconfig()
                except KeyError:
                        if 'default' in kwargs:
                                pass
                        else:
                                message=f"Key {keys[0]} not found in section {section}; valid keys include: {', '.join(keys)}"
                                if 'help' in kwargs:
                                        message+=f"\nHelp string: {kwargs['help'].format(**kwargs)}"
                                raise KeyError(message)
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
                parser.add_argument('configpositional', nargs='?',default=None, type=str,
                                    help='configuration file')
                parser.add_argument('-c','--configfile', default=None, type=str,
                                    help='configuration file')
                parser.add_argument('-s','--stage', default='all', type=str,
                                    help='stage - options are train and validate')
                parser.add_argument('--skip_validation', default=False, action="store_true",
                                    help='skip making validation plots')
                parser.add_argument('--fast', default=False, action="store_true",
                                    help='if set, run in fast mode for debugging')
                parser.add_argument('--bootstrap_single', default=False, action="store_true",
                                    help='if set, run a single bootstrap iteration and save to outputdir')
                
                def wrapaddingargument(*args,**kwargs):
                        #Wrap this method to catch exceptions, providing a true if no exception was raised, False otherwise.
                        try:
                                parser.add_argument_with_config_default(*args,**kwargs)
                                return True
                        except Exception as e:
                                log.error('\n'.join(e.args))
                                return False
                        
                # input files
                successful=True
                successful=successful&wrapaddingargument(config,'iodata','calibrationshiftfile',type=str,action=FullPaths,default='',
                                                        help='file containing a list of changes to zeropoint and central wavelength of filters by survey')
                successful=successful&wrapaddingargument(config,'iodata','calib_survey_ignore',type=boolean_string,
                                                        help='if True, ignore survey names when applying shifts to filters')
                successful=successful&wrapaddingargument(config,'iodata','loggingconfig','loggingconfigfile',  type=str,action=FullPaths,default='',
                                                        help='logging config file')
                successful=successful&wrapaddingargument(config,'iodata','trainingconfig','trainingconfigfile','modelconfigfile',  type=str,action=FullPaths,
                                                        help='Configuration file describing the construction of the model')
                successful=successful&wrapaddingargument(config,'iodata','snlists','snlistfiles',  type=str,action=FullPaths,
                                                        help="""list of SNANA-formatted SN data files, including both photometry and spectroscopy. Can be multiple comma-separated lists. (default=%(default)s)""")
                successful=successful&wrapaddingargument(config,'iodata','snparlist', 'snparlistfile', type=str,action=FullPaths,
                                                        help="""optional list of initial SN parameters.  Needs columns SNID, zHelio, x0, x1, c""")
                successful=successful&wrapaddingargument(config,'iodata','specrecallist','specrecallistfile',  type=str,action=FullPaths,
                                                        help="""optional list giving number of spectral recalibration params.  Needs columns SNID, N, phase, ncalib where N is the spectrum number for a given SN, starting at 1""")
                successful=successful&wrapaddingargument(config,'iodata','tmaxlist','tmaxlistfile',     type=str,action=FullPaths,
                                                        help="""optional space-delimited list with SN ID, tmax, tmaxerr (default=%(default)s)""")
        
                #output files
                successful=successful&wrapaddingargument(config,'iodata','outputdir',  type=str,action=FullPaths,
                                                        help="""data directory for spectroscopy, format should be ASCII 
                                                        with columns wavelength, flux, fluxerr (optional) (default=%(default)s)""")
                successful=successful&wrapaddingargument(config,'iodata','yamloutputfile', default='/dev/null',type=str,action=FullPaths,
                                                        help='File to which to output a summary of the fitting process')
                
                #options to configure cuts
                successful=successful&wrapaddingargument(config,'iodata','dospec',      type=boolean_string,
                                                        help="""if set, look for spectra in the snlist files (default=%(default)s)""")
                successful=successful&wrapaddingargument(config,'iodata','maxsn',  type=nonetype_or_int,
                                                        help="""sets maximum number of SNe to fit for debugging (default=%(default)s)""")
                successful=successful&wrapaddingargument(config,'iodata','keeponlyspec',         type=boolean_string,
                                                        help="""if set, only train on SNe with spectra (default=%(default)s)""")
                successful=successful&wrapaddingargument(config,'iodata','filter_mass_tolerance',  type=float,
                                                        help='Mass of filter transmission allowed outside of model wavelength range (default=%(default)s)')

                msg = "range of obs filter central wavelength (A)"
                successful=successful&wrapaddingargument(config, 'iodata',
                        'filtercen_obs_waverange',type=float,nargs=2,help=msg)

                #Initialize from SALT2.4                
                successful=successful&wrapaddingargument(config,'iodata','initsalt2model',      type=boolean_string,
                                                        help="""If true, initialize model parameters from prior SALT2 model""")
                successful=successful&wrapaddingargument(config,'iodata','initsalt2var',         type=boolean_string,
                                                        help="""If true, initialize model uncertainty parameters from prior SALT2 model""")
                #Initialize from user defined files                             
                successful=successful&wrapaddingargument(config,'iodata','initm0modelfile',     type=str,action=FullPaths,
                                                        help="""initial M0 model to begin training, ASCII with columns
                                                        phase, wavelength, flux (default=%(default)s)""")
                successful=successful&wrapaddingargument(config,'iodata','initm1modelfile',     type=str,action=FullPaths,
                                                        help="""initial M1 model with x1=1 to begin training, ASCII with columns
                                                        phase, wavelength, flux (default=%(default)s)""")
                #Choose B filter definition
                successful=successful&wrapaddingargument(config,'iodata','initbfilt',  type=str,action=FullPaths,
                                                        help="""initial B-filter to get the normalization of the initial model (default=%(default)s)""")
                                                        
                successful=successful&wrapaddingargument(config,'iodata','resume_from_outputdir',  type=str,action=FullPaths,
                                                        help='if set, initialize using output parameters from previous run. If directory, initialize using ouptut parameters from specified directory')
                successful=successful&wrapaddingargument(config,'iodata','resume_from_gnhistory',  type=str,action=FullPaths,
                                                        help='if set, initialize using output parameters from previous run saved in gaussnewtonhistory.pickle file.')
                successful=successful&wrapaddingargument(config,'iodata','error_dir', default='',type=str,action=FullPaths,
                                                        help='directory with previous error files, to use with --use_previous_errors option')

                
                successful=successful&wrapaddingargument(config,'iodata','fix_salt2modelpars',  type=boolean_string,
                                                        help="""if set, fix M0/M1 for wavelength/phase range of original SALT2 model (default=%(default)s)""")
                successful=successful&wrapaddingargument(config,'iodata','fix_salt2components',  type=boolean_string,
                                                        help="""if set, fix M0/M1 for *all* wavelength/phases (default=%(default)s)""")
                successful=successful&wrapaddingargument(config,'iodata','fix_salt2components_initdir',  type=str,
                                                        help="""if set, initialize component params from this directory (default=%(default)s)""")

                #validation option
                successful=successful&wrapaddingargument(config,'iodata','validate_modelonly',  type=boolean_string,
                                                        help="""if set, only make model plots in the validation stage""")
                # if resume_from_outputdir, use the errors from the previous run
                successful=successful&wrapaddingargument(config,'iodata','use_previous_errors', type=boolean_string,
                                                        help="""if set, use the errors from the previous run instead of computing new ones (can be memory intensive)""")
                successful=successful&wrapaddingargument(config,'iodata','filters_use_lastchar_only',  type=boolean_string,
                                                        help="""if set, use only the final filter character from the kcor file.  This is because of pathological SNANA records (default=%(default)s)""")


                successful=successful&wrapaddingargument(config,'trainparams','gaussnewton_maxiter',     type=int,
                                                        help='maximum iterations for Gauss-Newton (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','regularize',     type=boolean_string,
                                                        help='turn on regularization if set (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','fitsalt2',  type=boolean_string,
                                                        help='fit SALT2 as a validation check (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','n_repeat',  type=int,
                                                        help='repeat gauss newton n times (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','fit_model_err',  type=boolean_string,
                                                        help='fit for model error if set (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','fit_cdisp_only', type=boolean_string,
                                                        help='fit for color dispersion component of model error if set (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','steps_between_errorfit', type=int,
                                                        help='fit for error model every x steps (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','model_err_max_chisq', type=float,
                                                        help='max photometric chi2/dof below which model error estimation is done (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','dampingscalerate', type=float,
                                                        help=' Parameter that controls how quickly damping is adjusted in optimizer  (default=%(default)s)')
                                                        
                successful=successful&wrapaddingargument(config,'trainparams','lsmrmaxiter', type=int,
                                                        help=' Allowed number of iterations for the linear solver of the Gauss-Newton procedure to run  (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','preconditioningmaxiter', type=int,
                                                        help=' Number of operations used to evaluate preconditioning for the problem  (default=%(default)s)')


                successful=successful&wrapaddingargument(config,'trainparams','dampingscalerate', type=float,
                                                        help=' Parameter that controls how quickly damping is adjusted in optimizer  (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','dampingscalerate', type=float,
                                                        help=' Parameter that controls how quickly damping is adjusted in optimizer  (default=%(default)s)')

                successful=successful&wrapaddingargument(config,'trainparams','preconditioningchunksize', type=int,
                                                        help='Size of batches to evaluate preconditioning scales in Gauss-Newton proces. Increasing this value may increase memory performance at the cost of speed (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','photometric_zeropadding_batches', type=int,
                                                        help='Number of batches to divide the photometric data into when zero-padding. Increasing this value may improve memory performance at the cost of speed (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','spectroscopic_zeropadding_batches', type=int,
                                                        help='Number of batches to divide the spectroscopic data into when zero-padding. Increasing this value may improve memory performance at the cost of speed (default=%(default)s)')

                successful=successful&wrapaddingargument(config,'trainparams','fitting_sequence',  type=str,
                                                        help="Order in which parameters are fit, 'default' or empty string does the standard approach, otherwise should be comma-separated list with any of the following: all, pcaparams, color, colorlaw, spectralrecalibration, sn (default=%(default)s)")
                successful=successful&wrapaddingargument(config,'trainparams','fitprobmin',     type=float,
                                                        help="Minimum FITPROB for including SNe (default=%(default)s)")
                successful=successful&wrapaddingargument(config,'trainparams','errors_from_bootstrap',     type=boolean_string,
                                                        help="if set, get model surface errors from bootstrapping (default=%(default)s)")
                successful=successful&wrapaddingargument(config,'trainparams','n_bootstrap',     type=int,
                                                        help="number of bootstrap resamples (default=%(default)s)")
                successful=successful&wrapaddingargument(config,'trainparams','maxiter_bootstrap',     type=int,
                                                        help="maximum number of gauss-newton iterations for bootstrap estimation (default=%(default)s)")
                successful=successful&wrapaddingargument(config,'trainparams','bootstrap_sbatch_template',     type=str,
                                                        help="batch template for bootstrap estimation (default=%(default)s)")
                successful=successful&wrapaddingargument(config,'trainparams','bootstrap_batch_mode',     type=boolean_string,
                                                        help="batch mode for bootstrap estimation if set (default=%(default)s)")
                successful=successful&wrapaddingargument(config,'trainparams','get_bootstrap_output_only',     type=boolean_string,
                                                        help="collect the output from bootstrapping in batch mode without running new jobs (default=%(default)s)")
                successful=successful&wrapaddingargument(config,'trainparams','no_transformed_err_check',     type=boolean_string,
                                                        help="for host mass SALTShaker version, turn on this flag to ignore a current issue where x1/xhost de-correlation doesn\'t preserve errors appropriately; bootstrap errors will be needed (default=%(default)s)")
                successful=successful&wrapaddingargument(config,'trainparams','preintegrate_photometric_passband', type=boolean_string,
                                                        help='If true, integrate over the photometric passband prior to fitting the model. This approximation evaluates the color law only at the center of each spline basis function, and removes the need to integrate at every step of the fit. (default=%(default)s)')


                # survey definitions
                self.surveylist = [section.replace('survey_','') for section in config.sections() if section.startswith('survey_')]
                for survey in self.surveylist:
                        
                        successful=successful&wrapaddingargument(config,f'survey_{survey}',"kcorfile" ,type=str,clargformat=f"--{survey}" +"_{key}",action=FullPaths,
                                                                help="kcor file for survey %s"%survey)
                        successful=successful&wrapaddingargument(config,f'survey_{survey}',"subsurveylist" ,type=str,clargformat=f"--{survey}" +"_{key}",
                                                                help="comma-separated list of subsurveys for survey %s"%survey)
                        successful=successful&wrapaddingargument(config,f'survey_{survey}',"ignore_filters" ,type=str,clargformat=f"--{survey}" +"_{key}",
                                                                help="comma-separated list of filters to ignore for survey %s"%survey)

                if not successful: sys.exit(1)
                return parser


        def add_training_options(self, parser=None, usage=None, config=None):
                if parser == None:
                        parser = ConfigWithCommandLineOverrideParser(usage=usage, conflict_handler="resolve")
                def wrapaddingargument(*args,**kwargs):
                        #Wrap this method to catch exceptions, providing a true if no exception was raised, False otherwise.
                        try:
                                parser.add_argument_with_config_default(*args,**kwargs)
                                return True
                        except Exception as e:
                                log.error('\n'.join(e.args))
                                return False
                        
                # input files
                successful=True

                # training params
                successful=successful&wrapaddingargument(config,'trainingparams','specrecal',  type=int,
                                                        help='number of parameters defining the spectral recalibration (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','n_processes', type=int,
                                                        help='number of processes to use in calculating chi2 (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','estimate_tpk',         type=boolean_string,
                                                        help='if set, estimate time of max with quick least squares fitting (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','fix_t0',      type=boolean_string,
                                                        help='if set, don\'t allow time of max to float (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','regulargradientphase',         type=float,
                                                        help='Weighting of phase gradient chi^2 regularization during training of model parameters (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','regulargradientwave', type=float,
                                                        help='Weighting of wave gradient chi^2 regularization during training of model parameters (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','regulardyad', type=float,
                                                        help='Weighting of dyadic chi^2 regularization during training of model parameters (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','m1regularization',     type=float,
                                                        help='Scales regularization weighting of M1 component relative to M0 weighting (>1 increases smoothing of M1)  (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','mhostregularization',  type=float,
                                                        help='Scales regularization weighting of host component relative to M0 weighting (>1 increases smoothing of M1)  (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','spec_chi2_scaling',  type=float,
                                                        help='scaling of spectral chi^2 so it doesn\'t dominate the total chi^2 (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','n_min_specrecal',     type=int,
                                                        help='Minimum order of spectral recalibration polynomials (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','n_max_specrecal',     type=int,
                                                        help='Maximum order of spectral recalibration polynomials (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','specrange_wavescale_specrecal',  type=float,
                                                        help='Wavelength scale (in angstroms) for determining additional orders of spectral recalibration from wavelength range of spectrum (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','n_specrecal_per_lightcurve',  type=float,
                                                        help='Number of additional spectral recalibration orders per lightcurve (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','regularizationScaleMethod',  type=str,
                                                        help='Choose how scale for regularization is calculated (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','binspec',     type=boolean_string,
                                                        help='bin the spectra if set (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','binspecres',  type=int,
                                                        help='binning resolution (default=%(default)s)')
                
                #neff parameters
                successful=successful&wrapaddingargument(config,'trainingparams','wavesmoothingneff',  type=float,
                                                        help='Smooth effective # of spectral points along wave axis (in units of waveoutres) (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','phasesmoothingneff',  type=float,
                                                        help='Smooth effective # of spectral points along phase axis (in units of phaseoutres) (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','nefffloor',  type=float,
                                                        help='Minimum number of effective points (has to be > 0 to prevent divide by zero errors).(default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainingparams','neffmax',     type=float,
                                                        help='Threshold for spectral coverage at which regularization will be turned off (default=%(default)s)')

                # training model parameters
                successful=successful&wrapaddingargument(config,'modelparams','waverange', type=int, nargs=2,
                                                        help='wavelength range over which the model is defined (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','colorwaverange', type=int, nargs=2,
                                                        help='wavelength range over which the color law is fit to data (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','interpfunc',     type=str,
                                                        help='function to interpolate between control points in the fitting (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','errinterporder', type=int,
                                                        help='for model uncertainty splines/polynomial funcs, order of the function (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','interporder',     type=int,
                                                        help='for model splines/polynomial funcs, order of the function (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','wavesplineres',  type=float,
                                                        help='number of angstroms between each wavelength spline knot (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','phasesplineres', type=float,
                                                        help='number of angstroms between each phase spline knot (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','waveinterpres',  type=float,
                                                        help='wavelength resolution in angstroms, used for internal interpolation (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','phaseinterpres', type=float,
                                                        help='phase resolution in angstroms, used for internal interpolation (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','waveoutres',     type=float,
                                                        help='wavelength resolution in angstroms of the output file (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','phaseoutres',     type=float,
                                                        help='phase resolution in angstroms of the output file (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','phaserange', type=int, nargs=2,
                                                        help='phase range over which model is trained (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','n_components',  type=int,
                                                        help='number of principal components of the SALT model to fit for (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','host_component', type=str,
                                                        help="NOT IMPLEMENTED: if set, fit for a host component.  Must equal 'mass', for now (default=%(default)s)")
                successful=successful&wrapaddingargument(config,'modelparams','n_colorpars',     type=int,
                                                        help='number of degrees of the phase-independent color law polynomial (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','n_colorscatpars',         type=int,
                                                        help='number of parameters in the broadband scatter model (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','error_snake_phase_binsize',      type=float,
                                                        help='number of days over which to compute scaling of error model (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','error_snake_wave_binsize',  type=float,
                                                        help='number of angstroms over which to compute scaling of error model (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'modelparams','use_snpca_knots',         type=boolean_string,
                                                        help='if set, define model on SNPCA knots (default=%(default)s)')               
                successful=successful&wrapaddingargument(config,'modelparams','colorlaw_function',         type=str,
                                                        help='color law function, see colorlaw.py (default=%(default)s)')               

                # priors
                for prior in __priors__:
                        successful=successful&wrapaddingargument(config,'priors',prior ,type=float,clargformat="--prior_{key}",
                                                                help=f"prior on {prior}",default=SUPPRESS)

                # bounds
                for bound,val in config.items('bounds'):
                        successful=successful&wrapaddingargument(config,'bounds', bound, type=float,nargs=3,clargformat="--bound_{key}",
                                                                help="bound on %s"%bound)
                if not successful: sys.exit(1)
                return parser

        def get_saltkw(self,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc):

                saltfitkwargs = {
                'lsmrmaxiter':self.options.lsmrmaxiter, 'preconditioningmaxiter':self.options.preconditioningmaxiter,
                'dampingscalerate': self.options.dampingscalerate,
                'spectroscopic_zeropadding_batches': self.options.spectroscopic_zeropadding_batches, 'photometric_zeropadding_batches':self.options.photometric_zeropadding_batches,
                'preconditioningchunksize':self.options.preconditioningchunksize,
                'preintegrate_photometric_passband':self.options.preintegrate_photometric_passband,'m1regularization':self.options.m1regularization,'mhostregularization':self.options.mhostregularization,
                                 'bsorder':self.options.interporder,'errbsorder':self.options.errinterporder,
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
                                 'spec_chi2_scaling':self.options.spec_chi2_scaling,
                                 'debug':self.options.debug,
                                 'use_previous_errors':self.options.use_previous_errors,
                                 'fix_salt2modelpars':self.options.fix_salt2modelpars,
                                 'fix_salt2components':self.options.fix_salt2components,
                                 'no_transformed_err_check':self.options.no_transformed_err_check,
                                 'colorlaw_function':self.options.colorlaw_function}

                for k in self.options.__dict__.keys():
                    if k.startswith('prior') or k.startswith('bound'):
                        saltfitkwargs[k] = self.options.__dict__[k]
                return saltfitkwargs
        
        def snshouldbecut(self,sn,cuts):
        
            passescuts= [cut.passescut(sn) for cut in cuts]
                
            if not all(passescuts): # or iPreMaxCut < 2 or medSNR < 10:
                log.debug('SN %s fails cuts'%sn.snid)
                log.debug(', '.join([f'{cut.cutvalue(sn)} {cut.description} ({cut.requirement} needed)' for cut in cuts]))

            return not all(passescuts)

        def checkFilterMass(self,z,survey,flt):

            try: 
                filtwave = self.kcordict[survey][flt]['filtwave']
            except: 
                raise RuntimeError(f"filter {flt} not found in kcor file for survey {survey}.  Check your config file")
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
                if hasvalidfitprob:
                    return sn.salt2fitprob
                else:
                    log.warning(f'SN {sn.snid} does not have a valid salt2 fitprob, including in sample')
                    return 1
                        
            cuts= [SNCut('total epochs',4,lambda sn: sum([ ((sn.photdata[flt].phase > -10) & (sn.photdata[flt].phase < 35)).sum() for flt in sn.photdata])),
                   SNCut('epochs near peak',1,lambda sn: sum([ ((sn.photdata[flt].phase > -10) & (sn.photdata[flt].phase < 5)).sum() for flt in sn.photdata])),
                   SNCut('epochs post peak',1,lambda sn: sum([      ((sn.photdata[flt].phase > 5) & (sn.photdata[flt].phase < 20)).sum() for flt in sn.photdata])),
                   SNCut('filters near peak',2,lambda sn: sum([ (((sn.photdata[flt].phase > -8) & (sn.photdata[flt].phase < 10)).sum())>0 for flt in sn.photdata])),
                   SNCut('salt2 fitprob',self.options.fitprobmin,checkfitprob)]
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
                    #Remove light curve outside model [rest-frame] wavelength range
                    if not self.filter_select(sn.survey,flt):  # RK Nov 7 2022
                    #if flt in self.options.__dict__[f"{sn.survey.split('(')[0]}_ignore_filters"].split(','):
                        photdata.pop(flt)  # remove filter
                    elif self.checkFilterMass(z,sn.survey,flt):
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

        def filter_select(self,survey,flt):
                select = True
                if flt in self.options.__dict__[f"{survey.split('(')[0]}_ignore_filters"].split(','):
                        select = False

                lambdaeff = self.kcordict[survey][flt]['lambdaeff']
                if lambdaeff < self.options.filtercen_obs_waverange[0] or \
                   lambdaeff > self.options.filtercen_obs_waverange[1] :
                        select = False                

                return select                
                # end filter_select
