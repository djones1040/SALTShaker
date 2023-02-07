#!/usr/bin/env python
# D. Jones, R. Kessler - 8/31/18
from __future__ import print_function

import configparser
import numpy as np
import sys
import multiprocessing
import pickle
import copy

import os
from os import path
import subprocess

from scipy.linalg import lstsq
from scipy.optimize import minimize, least_squares, differential_evolution
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from sncosmo.constants import HC_ERG_AA
import astropy.table as at

from saltshaker.config.configoptions import *

from saltshaker.util import snana,readutils
from saltshaker.util.estimate_tpk_bazin import estimate_tpk_bazin
from saltshaker.util.txtobj import txtobj
from saltshaker.util.specSynPhot import getScaleForSN
from saltshaker.util.specrecal import SpecRecal

from saltshaker.training.init_hsiao import init_hsiao, init_kaepora, init_errs,init_errs_percent,init_custom,init_salt2
from saltshaker.training.base import TrainSALTBase
from saltshaker.training.saltfit import fitting
from saltshaker.training import saltfit as saltfit
from saltshaker.validation import ValidateParams,datadensity

from saltshaker.data import data_rootdir
from saltshaker.initfiles import init_rootdir

from saltshaker.config import config_rootdir,loggerconfig

import astropy.units as u
import sncosmo
import yaml

from astropy.table import Table
from saltshaker.initfiles import init_rootdir as salt2dir
_flatnu=f'{init_rootdir}/flatnu.dat'

# validation utils
import matplotlib as mpl
mpl.use('agg')
import pylab as plt
from saltshaker.validation import ValidateLightcurves
from saltshaker.validation import ValidateSpectra
from saltshaker.validation import ValidateModel
from saltshaker.validation import CheckSALTParams
from saltshaker.validation.figs import plotSALTModel
from saltshaker.util.synphot import synphot
from saltshaker.initfiles import init_rootdir as salt2dir
from saltshaker.validation import SynPhotPlot
import time
from sncosmo.salt2utils import SALT2ColorLaw
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.special import factorial
import extinction

import logging
log=logging.getLogger(__name__)


def parse_user_options( parser=None,args=None):
    if parser is None: parser= DefaultRequiredParser(usage="""SALT3 Training

usage: python TrainSALT.py -c <configfile> <options>

config file options can be overwridden at the command line""")

    group=parser.add_argument_group('Config files')
    primaryconfig=group.add_mutually_exclusive_group()
    primaryconfig.add('configpath',nargs='?',is_config_file_arg=True)
    primaryconfig.add('-c','--configpathflagged', is_config_file_arg=True,default=None)
    group.add('--surveyconfig',type=str,action=FullPaths,default=None, help='file containing paths to kcor files by survey')
    group.add('--loggingconfig','--loggingconfigfile',  type=str,action=FullPaths,default='', help='logging config file')
    group.add('--trainingconfig','--trainingconfigfile','--modelconfigfile',  type=str,action=FullPaths, help='Configuration file describing the construction of the model')

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


    group=parser.add_argument_group('Data and initialized state')
    group.add('--snlists','--snlistfiles',  type=str,action=FullPaths, help="""list of SNANA-formatted SN data files, including both photometry and spectroscopy. Can be multiple comma-separated lists. (default=%(default)s)""")
    group.add('--snparlist', '--snparlistfile', type=str,action=FullPaths, help="""optional list of initial SN parameters.  Needs columns SNID, zHelio, x0, x1, c""")
    group.add('--specrecallist','--specrecallistfile',  type=str,action=FullPaths, help="""optional list giving number of spectral recalibration params.  Needs columns SNID, N, phase, ncalib where N is the spectrum number for a given SN, starting at 1""")
    group.add('--tmaxlist','--tmaxlistfile',     type=str,action=FullPaths, help="""optional space-delimited list with SN ID, tmax, tmaxerr (default=%(default)s)""")
    group.add('--initsalt2model',      type=boolean_string, help="""If true, initialize model parameters from prior SALT2 model""")
    group.add('--initsalt2var',         type=boolean_string, help="""If true, initialize model uncertainty parameters from prior SALT2 model""")     
    group.add('--initm0modelfile',     type=str,action=FullPaths, help="""initial M0 model to begin training, ASCII with columns phase, wavelength, flux (default=%(default)s)""")
    group.add('--initm1modelfile',     type=str,action=FullPaths, help="""initial M1 model with x1=1 to begin training, ASCII with columns phase, wavelength, flux (default=%(default)s)""")
    group.add('--initbfilt',  type=str,action=FullPaths, help="""initial B-filter to get the normalization of the initial model (default=%(default)s)""")
    group.add('--resume_from_outputdir',  type=str,action=FullPaths, help='if set, initialize using output parameters from previous run. If directory, initialize using ouptut parameters from specified directory')
    group.add('--resume_from_gnhistory',  type=str,action=FullPaths, help='if set, initialize using output parameters from previous run saved in gaussnewtonhistory.pickle file.')


    group=parser.add_argument_group('Calibration options')
    group.add ('--calibrationshiftfile',type=str,action=FullPaths,default='',
    help='file containing a list of changes to zeropoint and central wavelength of filters by survey')
    group.add('--calib_survey_ignore',type=boolean_string, help='if True, ignore survey names when applying shifts to filters')

    group=parser.add_argument_group('Requirements and cuts')
    group.add('--maxsn',  type=nonetype_or_int, help="""sets maximum number of SNe to fit for debugging (default=%(default)s)""")
    group.add('--filter_mass_tolerance',  type=float, help='Mass of filter transmission allowed outside of model wavelength range (default=%(default)s)')
    group.add('--filtercen_obs_waverange',type=float,nargs=2,help="range of obs filter central wavelength (A)",
              default= [-np.inf,np.inf])
    group.add('--keeponlyspec',         type=boolean_string, help="""if set, only train on SNe with spectra (default=%(default)s)""")
    group.add('--dospec',      type=boolean_string, help="""if set, look for spectra in the snlist files (default=%(default)s)""")


    group=parser.add_argument_group('Outputs')
    group.add('--outputdir',  type=str,action=FullPaths,help="""data directory for spectroscopy, format should be ASCII  with columns wavelength, flux, fluxerr (optional) (default=%(default)s)""")
    group.add('--yamloutputfile', default='/dev/null',type=str,action=FullPaths,help='File to which to output a summary of the fitting process')

    parser.add('--estimate_tpk',         type=boolean_string,
                help='if set, estimate time of max with quick least squares fitting (default=%(default)s)')')
    parser.add('--error_dir', default='',type=str,action=FullPaths, help='directory with previous error files, to use with --use_previous_errors option')
    parser.add('--fix_salt2modelpars',  type=boolean_string, help="""if set, fix M0/M1 for wavelength/phase range of original SALT2 model (default=%(default)s)""")
    parser.add('--fix_salt2components',  type=boolean_string, help="""if set, fix M0/M1 for *all* wavelength/phases (default=%(default)s)""")
    parser.add('--fix_salt2components_initdir',  type=str, help="""if set, initialize component params from this directory (default=%(default)s)""")
    parser.add('--validate_modelonly',  type=boolean_string, help="""if set, only make model plots in the validation stage""")
    parser.add('--use_previous_errors', type=boolean_string, help="""if set, use the errors from the previous run instead of computing new ones (can be memory intensive)""")
    parser.add('--filters_use_lastchar_only',  type=boolean_string, help="""if set, use only the final filter character from the kcor file.  This is because of pathological SNANA records (default=%(default)s)""")
    parser.add('--gaussnewton_maxiter',     type=int, help='maximum iterations for Gauss-Newton (default=%(default)s)')
    parser.add('--regularize',     type=boolean_string, help='turn on regularization if set (default=%(default)s)')
    parser.add('--fitsalt2',  type=boolean_string, help='fit SALT2 as a validation check (default=%(default)s)')
    parser.add('--n_repeat',  type=int, help='repeat gauss newton n times (default=%(default)s)')
    parser.add('--fit_model_err',  type=boolean_string, help='fit for model error if set (default=%(default)s)')
    parser.add('--fit_cdisp_only', type=boolean_string, help='fit for color dispersion component of model error if set (default=%(default)s)')
    parser.add('--steps_between_errorfit', type=int, help='fit for error model every x steps (default=%(default)s)')
    parser.add('--model_err_max_chisq', type=int, help='max photometric chi2/dof below which model error estimation is done (default=%(default)s)')
    parser.add('--fitting_sequence',  type=str, help="Order in which parameters are fit, 'default' or empty string does the standard approach, otherwise should be comma-separated list with any of the following: all, pcaparams, color, colorlaw, spectralrecalibration, sn (default=%(default)s)")
    parser.add('--fitprobmin',     type=float, help="Minimum FITPROB for including SNe (default=%(default)s)")
    parser.add('--errors_from_bootstrap',     type=boolean_string, help="if set, get model surface errors from bootstrapping (default=%(default)s)")
    parser.add('--n_bootstrap',     type=int, help="number of bootstrap resamples (default=%(default)s)")
    parser.add('--maxiter_bootstrap',     type=int, help="maximum number of gauss-newton iterations for bootstrap estimation (default=%(default)s)")
    parser.add('--bootstrap_sbatch_template',     type=str, help="batch template for bootstrap estimation (default=%(default)s)")
    parser.add('--bootstrap_batch_mode',     type=boolean_string, help="batch mode for bootstrap estimation if set (default=%(default)s)")
    parser.add('--get_bootstrap_output_only',     type=boolean_string, help="collect the output from bootstrapping in batch mode without running new jobs (default=%(default)s)")
    parser.add('--no_transformed_err_check',     type=boolean_string,  help="for host mass SALTShaker version, turn on this flag to ignore a current issue where x1/xhost de-correlation doesn\'t preserve errors appropriately; bootstrap errors will be needed (default=%(default)s)")


    parsed,unparsed=parser.parse_known_args(args)
    if parsed.surveyconfig is None:
        log.warning('No survey configuration file specified, attempting to read from user options')
        parsed.surveyconfig=parsed.configpath 
    if parsed.configpathflagged is not None:
        parsed.configpath=parsed.configpathflagged

    unparsedargnames=[ arg[2:max([arg.find(char) for char in ['=',':']])] for arg in unparsed if arg.startswith('--')]
    log.warning(f'Unknown arguments in config file: '+ ', '.join(unparsedargnames))

    return parsed,unparsed


def parse_survey_options(surveyfile):
    surveyparser=configparser.ConfigParser()
    surveyparser.read(surveyfile)
    if any([ section.startswith('survey_') for section in surveyparser.sections()]):
        surveydict= {section.replace('survey_',''): surveyparser._sections[section] for section in surveyparser.sections() if section.startswith('survey_')}
    else:
        surveydict= {section : surveyparser._sections[section] for section in surveyparser.sections()  }
    assert(all( [key in surveydict[survey] for key in ['ignore_filters', 'subsurveylist','kcorfile' ] for survey in surveydict]))
    return surveydict
    
def initialconfiguration(self,args=None):
    user_options = parse_parse_user_options(args)
    if not os.path.exists(user_options.outputdir):
        os.makedirs(user_options.outputdir)

    loggerconfig.dictconfigfromYAML(user_options.loggingconfig,user_options.outputdir)

    if not os.path.exists(user_options.trainingconfig):
        log.warning('training config file %s doesn\'t exist.  Trying package directory'%user_options.trainingconfig)
        user_options.trainingconfig = '%s/%s'%(config_rootdir,user_options.trainingconfig)
    if not os.path.exists(user_options.trainingconfig):
        raise RuntimeError('can\'t find training config file!  Checked %s'%user_options.trainingconfig)


    model_options =parse_model_options(args,user_options.trainingconfig )
    surveydict=parse_survey_options(user_options.survey_config )

    if user_options.fast:
        if user_options.gaussnewton_maxiter >= 1:
            user_options.gaussnewton_maxiter = 1
        user_options.fit_model_err = False
        user_options.fit_cdisp_only = False
        user_options.validate_modelonly = True
        user_options.maxsn = 10

    if user_options.stage not in ['all','validate','train']:
        raise RuntimeError('stage must be one of all, validate, train')

    return user_options, model_options, surveydict


def RatioToSatisfyDefinitions(phase,wave,kcordict,components):
    """Ensures that the definitions of M1,M0,x0,x1 are satisfied"""

    Bmag = synphot(
        kcordict['default']['primarywave'],kcordict['default']['AB'],
        filtwave=kcordict['default']['Bwave'],filttp=kcordict['default']['Btp'],
        zpoff=0)
    
    Bflux = 10**(0.4*(Bmag+27.5))

    filttrans = kcordict['default']['Btp']
    filtwave = kcordict['default']['Bwave']
            
    pbspl = np.interp(wave,filtwave,filttrans,left=0,right=0)
    
    pbspl *= wave
    denom = np.trapz(pbspl,wave)
    pbspl /= denom*HC_ERG_AA
    kcordict['default']['Bpbspl'] = pbspl
    
    int1d = interp1d(phase,components[0],axis=0,assume_sorted=True)
    m0Bflux = np.sum(kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
        (wave[1]-wave[0])*Bflux
    
    int1d = interp1d(phase,components[1],axis=0,assume_sorted=True)
    m1Bflux = np.sum(kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
        (wave[1]-wave[0])*Bflux
    ratio=m1Bflux/m0Bflux
    return ratio


class TrainSALT(TrainSALTBase):
    def __init__(self,configfile=None):
        self.warnings = []
        self.initializationtime=time.time()
        


        
    def fitSALTModel(self,datadict,returnGN=False):

        # check for option inconsistency
        if self.options.use_previous_errors and not self.options.resume_from_outputdir and not self.options.error_dir:
            raise RuntimeError('resume_from_outputdir or error_dir must be specified to use use_previous_errors option')

        parlist,x_modelpars,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc = self.initialParameters(datadict)

        saltfitkwargs = self.get_saltkw(phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)
        n_phaseknots,n_waveknots = len(phaseknotloc)-4,len(waveknotloc)-4
        n_errphaseknots,n_errwaveknots = len(errphaseknotloc)-4,len(errwaveknotloc)-4

        if self.options.bootstrap_single:
            new_keys = np.random.choice(list(datadict.keys()),size=len(datadict.keys()))

            # make a new dictionary, ensure names are unique
            datadict_bootstrap = {}
            for nid,k in enumerate(new_keys):
                datadict_bootstrap[str(nid)] = copy.deepcopy(datadict[k])
                datadict_bootstrap[str(nid)].snid_orig = datadict[k].snid[:]
                datadict_bootstrap[str(nid)].snid = str(nid)
                
            # construct the new parlist
            x_modelpars_bs,parlist_bs,keys_done = np.array([]),np.array([]),np.array([])
            for i,xm in enumerate(x_modelpars):
                if '_' not in parlist[i] or 'modelerr' in parlist[i] or 'modelcorr' in parlist[i]:
                    x_modelpars_bs = np.append(x_modelpars_bs,xm)
                    parlist_bs = np.append(parlist_bs,parlist[i])
            for i,k in enumerate(new_keys):
                snidpars = [(x,p) for x,p in zip(x_modelpars,parlist) if '_' in p and p.split('_')[1] == k]
                for xp in snidpars:
                    x,p = xp
                    parlist_parts = p.split('_')
                    snid = p.split('_')[1]
                    if len(parlist_parts) == 2:
                        x_modelpars_bs = np.append(x_modelpars_bs,x) #x_modelpars[parlist == p])
                        parlist_bs = np.append(parlist_bs,parlist_parts[0]+'_'+str(i))
                    elif len(parlist_parts) == 3:
                        x_modelpars_bs = np.append(x_modelpars_bs,x) # x_modelpars[parlist == p])
                        parlist_bs = np.append(parlist_bs,parlist_parts[0]+'_'+str(i)+'_'+parlist_parts[2])

            datadict = copy.deepcopy(datadict_bootstrap)
            parlist = copy.deepcopy(parlist_bs)
            x_modelpars = copy.deepcopy(x_modelpars_bs)


        fitter = fitting(self.options.n_components,self.options.n_colorpars,
                         n_phaseknots,n_waveknots,
                         datadict)
        log.info('training on %i SNe!'%len(datadict.keys()))
        for i in range(self.options.n_repeat):
            if i == 0: laststepsize = None
            
            saltfitkwargs['regularize'] = self.options.regularize
            saltfitkwargs['fitting_sequence'] = self.options.fitting_sequence
            saltfitter = saltfit.GaussNewton(x_modelpars,datadict,parlist,**saltfitkwargs)
            if self.options.bootstrap_single:
                # suppress regularization
                saltfitter.neff[saltfitter.neff<saltfitter.neffMax]=10

            if returnGN:
                return fitter,saltfitter,x_modelpars
            
            # do the fitting
            trainingresult,message = fitter.gaussnewton(
                saltfitter,x_modelpars,
                self.options.gaussnewton_maxiter,
                getdatauncertainties=(not self.options.use_previous_errors and not self.options.errors_from_bootstrap))
            for k in datadict.keys():
                trainingresult.SNParams[k]['t0'] =  datadict[k].tpk_guess
        
        log.info('message: %s'%message)
        log.info('Final loglike'); log.info(saltfitter.maxlikefit(trainingresult.X_raw))
        log.info('Final photometric loglike'); log.info(saltfitter.maxlikefit(trainingresult.X_raw,dospec=False))
        
        log.info(trainingresult.X.size)



        if 'chain' in saltfitter.__dict__.keys():
            chain = saltfitter.chain
            loglikes = saltfitter.loglikes
        else: chain,loglikes = None,None

        return trainingresult,chain,loglikes,saltfitter

    def wrtoutput(self,outdir,trainingresult,chain,
                  loglikes,datadict):
        if not os.path.exists(outdir):
            raise RuntimeError('desired output directory %s doesn\'t exist'%outdir)

        #Save final model parameters
        
        with  open('{}/salt3_parameters.dat'.format(outdir),'w') as foutpars:
            foutpars.write('{: <30} {}\n'.format('Parameter Name','Value'))
            for name,par in zip(trainingresult.parlist,trainingresult.X):

                foutpars.write('{: <30} {:.15e}\n'.format(name,par))

        with  open('{}/salt3_parameters_unscaled.dat'.format(outdir),'w') as foutpars:
            foutpars.write('{: <30} {}\n'.format('Parameter Name','Value'))
            for name,par in zip(trainingresult.parlist,trainingresult.X_raw):

                foutpars.write('{: <30} {:.15e}\n'.format(name,par))

        np.save('{}/salt3_mcmcchain.npy'.format(outdir),chain)
        np.save('{}/salt3_loglikes.npy'.format(outdir),loglikes)
        # principal components and color law
        with open(f'{outdir}/salt3_template_0.dat','w') as foutm0, open('%s/salt3_template_1.dat'%outdir,'w') as foutm1,\
             open(f'{outdir}/salt3_lc_model_variance_0.dat','w') as foutm0modelerr,\
             open(f'{outdir}/salt3_lc_model_variance_1.dat','w') as foutm1modelerr,\
             open(f'{outdir}/salt3_lc_dispersion_scaling.dat','w') as fouterrmod,\
             open(f'{outdir}/salt3_lc_model_covariance_01.dat','w') as foutmodelcov,\
             open(f'{outdir}/salt3_lc_covariance_01.dat','w') as foutdatacov,\
             open(f'{outdir}/salt3_lc_variance_0.dat','w') as foutm0dataerr,\
             open(f'{outdir}/salt3_lc_variance_1.dat','w') as foutm1dataerr:
        
            for i,p in enumerate(trainingresult.phase):
                for j,w in enumerate(trainingresult.wave):
                    print(f'{p:.1f} {w:.2f} {trainingresult.M0[i,j]:8.15e}',file=foutm0)
                    print(f'{p:.1f} {w:.2f} {trainingresult.M1[i,j]:8.15e}',file=foutm1)

                    if not self.options.use_previous_errors:
                        print(f'{p:.1f} {w:.2f} {trainingresult.M0modelerr[i,j]**2.:8.15e}',file=foutm0modelerr)
                        print(f'{p:.1f} {w:.2f} {trainingresult.M1modelerr[i,j]**2.:8.15e}',file=foutm1modelerr)
                        print(f'{p:.1f} {w:.2f} {trainingresult.cov_M0_M1_model[i,j]:8.15e}',file=foutmodelcov)
                        print(f'{p:.1f} {w:.2f} {trainingresult.modelerr[i,j]:8.15e}',file=fouterrmod)
                        if self.options.errors_from_bootstrap:
                            print(f'{p:.1f} {w:.2f} {trainingresult.M0bootstraperr[i,j]**2.:8.15e}',file=foutm0dataerr)
                            print(f'{p:.1f} {w:.2f} {trainingresult.M1bootstraperr[i,j]**2.:8.15e}',file=foutm1dataerr)
                            print(f'{p:.1f} {w:.2f} {trainingresult.cov_M0_M1_bootstrap[i,j]:8.15e}',file=foutdatacov)
                        else:
                            print(f'{p:.1f} {w:.2f} {trainingresult.M0dataerr[i,j]**2.+trainingresult.M0modelerr[i,j]**2.:8.15e}',file=foutm0dataerr)
                            print(f'{p:.1f} {w:.2f} {trainingresult.M1dataerr[i,j]**2.+trainingresult.M1modelerr[i,j]**2.:8.15e}',file=foutm1dataerr)
                            print(f'{p:.1f} {w:.2f} {trainingresult.cov_M0_M1_data[i,j]+trainingresult.cov_M0_M1_model[i,j]:8.15e}',file=foutdatacov)
        if self.options.host_component:
            with open(f'{outdir}/salt3_template_host.dat','w') as foutmhost,\
             open(f'{outdir}/salt3_lc_covariance_0host.dat','w') as foutm0mhostcov,\
             open(f'{outdir}/salt3_lc_variance_host.dat','w') as foutmhostdataerr,\
             open(f'{outdir}/salt3_lc_model_variance_host.dat','w') as foutmhostmodelerr,\
             open(f'{outdir}/salt3_lc_model_covariance_0host.dat','w') as foutm0mhostmodelcov:
                for i,p in enumerate(trainingresult.phase):
                        print(f'{p:.1f} {w:.2f} {trainingresult.Mhost[i,j]:8.15e}',file=foutmhost)
                        print(f'{p:.1f} {w:.2f} {trainingresult.Mhostmodelerr[i,j]:8.15e}',file=foutmhostmodelerr)
                        print(f'{p:.1f} {w:.2f} {trainingresult.cov_M0_Mhost_model[i,j]:8.15e}',file=foutm0mhostmodelcov)

                        if self.options.errors_from_bootstrap:
                            print(f'{p:.1f} {w:.2f} {trainingresult.Mhostbootstraperr[i,j]**2.:8.15e}',file=foutmhostdataerr)
                            print(f'{p:.1f} {w:.2f} {trainingresult.cov_M0_Mhost_bootstrap[i,j]:8.15e}',file=foutm0mhostcov)
                        else:
                            print(f'{p:.1f} {w:.2f} {trainingresult.Mhostdataerr[i,j]**2.+trainingresult.Mhostmodelerr[i,j]**2.:8.15e}',file=foutmhostdataerr)
                            print(f'{p:.1f} {w:.2f} {trainingresult.cov_M0_Mhost_data[i,j]+trainingresult.cov_M0_Mhost_model[i,j]**2.:8.15e}',file=foutm0mhostcov)


        if self.options.use_previous_errors and self.options.resume_from_outputdir:
            for filename in ['salt3_lc_variance_0.dat','salt3_lc_variance_1.dat','salt3_lc_variance_host.dat',
                             'salt3_lc_covariance_01.dat','salt3_lc_covariance_0host.dat',
                             'salt3_lc_variance_0.dat','salt3_lc_variance_1.dat']:
                os.system(f"cp {self.options.resume_from_outputdir}/{filename} {outdir}/{filename}")
        elif self.options.use_previous_errors and self.options.error_dir:
            for filename in ['salt3_lc_variance_0.dat','salt3_lc_variance_1.dat',
                             'salt3_lc_covariance_01.dat','salt3_lc_variance_0.dat',
                             'salt3_lc_covariance_0host.dat',
                             'salt3_lc_variance_1.dat']:
                os.system(f"cp {self.options.error_dir}/{filename} {outdir}/{filename}")
                
        with open(f'{outdir}/salt3_color_dispersion.dat','w') as foutclscat:
            trainingresult.clscat = np.clip(trainingresult.clscat,0.,5.)
            for j,w in enumerate(trainingresult.wave):
                print(f'{w:.2f} {trainingresult.clscat[j]:8.15e}',file=foutclscat)

        foutinfotext = f"""RESTLAMBDA_RANGE: {self.options.colorwaverange[0]} {self.options.colorwaverange[1]}
COLORLAW_VERSION: 1
COLORCOR_PARAMS: {self.options.colorwaverange[0]:.0f} {self.options.colorwaverange[1]:.0f}  {len(trainingresult.clpars)}  {' '.join(['%8.10e'%cl for cl in trainingresult.clpars])}

COLOR_OFFSET:  0.0
COLOR_DISP_MAX: 1.0  # avoid crazy sim-mags at high-z

MAG_OFFSET:  0.27  # to get B-band mag from cosmology fit (Nov 23, 2011)

SEDFLUX_INTERP_OPT: 2  # 1=>linear,    2=>spline
ERRMAP_INTERP_OPT:  1  # 1  # 0=snake off;  1=>linear  2=>spline
ERRMAP_KCOR_OPT:    1  # 1/0 => on/off

MAGERR_FLOOR:   0.005            # don;t allow smaller error than this
MAGERR_LAMOBS:  0.0  2000  4000  # magerr minlam maxlam
MAGERR_LAMREST: 0.1   100   200  # magerr minlam maxlam

SIGMA_INT: 0.106  # used in simulation"""
        with open(f'{outdir}/SALT3.INFO','w') as foutinfo:
            print(foutinfotext,file=foutinfo)
        

        with open(f'{outdir}/salt3_color_correction.dat','w') as foutcl:
            print(f'{len(trainingresult.clpars):.0f}',file=foutcl)
            for c in trainingresult.clpars:
                print(f'{c:8.10e}',file=foutcl)
            print(f"""Salt2ExtinctionLaw.version 1
            Salt2ExtinctionLaw.min_lambda {self.options.colorwaverange[0]:.0f}
            Salt2ExtinctionLaw.max_lambda {self.options.colorwaverange[1]:.0f}""",file=foutcl)

        
        # best-fit and simulated SN params
        with open(f'{outdir}/salt3train_snparams.txt','w') as foutsn:
            print('# SN x0 x1 c t0 SIM_x0 SIM_x1 SIM_c SIM_t0 SALT2_x0 SALT2_x1 SALT2_c SALT2_t0',file=foutsn)
            for snlist in self.options.snlists.split(','):
                snlist = os.path.expandvars(snlist)
                if not os.path.exists(snlist):
                    log.warning(f'SN list file {snlist} does not exist. Checking {data_rootdir}/trainingdata/{snlist}')
                    snlist = f'{data_rootdir}/trainingdata/{snlist}'
                    if not os.path.exists(snlist):
                        raise RuntimeError(f'SN list file {snlist} does not exist')


                snfiles = np.genfromtxt(snlist,dtype='str')
                snfiles = np.atleast_1d(snfiles)

                for k in trainingresult.SNParams.keys():
                    foundfile = False
                    SIM_x0,SIM_x1,SIM_c,SIM_PEAKMJD,salt2x0,salt2x1,salt2c,salt2t0 = -99,-99,-99,-99,-99,-99,-99,-99
                    for l in snfiles:
                        if '.fits' in l.lower(): continue
                        if str(k) not in l: continue
                        foundfile = True
                        if '/' not in l:
                            l = f"{os.path.dirname(snlist)}/{l}"
                        sn = snana.SuperNova(l)
                        if str(k) != str(sn.SNID): continue

                        sn.SNID = str(sn.SNID)
                        if 'SIM_SALT2x0' in sn.__dict__.keys(): SIM_x0 = sn.SIM_SALT2x0
                        else: SIM_x0 = -99
                        if 'SIM_SALT2x1' in sn.__dict__.keys(): SIM_x1 = sn.SIM_SALT2x1
                        else: SIM_x1 = -99
                        if 'SIM_SALT2c' in sn.__dict__.keys(): SIM_c = sn.SIM_SALT2c
                        else: SIM_c = -99
                        if 'SIM_PEAKMJD' in sn.__dict__.keys(): SIM_PEAKMJD = float(sn.SIM_PEAKMJD.split()[0])
                        else: SIM_PEAKMJD = -99
                        break
                    if not foundfile:
                        SIM_x0,SIM_x1,SIM_c,SIM_PEAKMJD,salt2x0,salt2x1,salt2c,salt2t0 = -99,-99,-99,-99,-99,-99,-99,-99
                    elif self.options.fitsalt2:
                        salt2x0,salt2x1,salt2c,salt2t0 = self.salt2fit(sn,datadict)
                    else:
                        salt2x0,salt2x1,salt2c,salt2t0 = -99,-99,-99,-99

                    if 't0' not in trainingresult.SNParams[k].keys():
                        trainingresult.SNParams[k]['t0'] = 0.0

                    print(f"{k} {trainingresult.SNParams[k]['x0']:8.10e} {trainingresult.SNParams[k]['x1']:.10f} {trainingresult.SNParams[k]['c']:.10f} {trainingresult.SNParams[k]['t0']:.10f} {SIM_x0:8.10e} {SIM_x1:.10f} {SIM_c:.10f} {SIM_PEAKMJD:.2f} {salt2x0:8.10e} {salt2x1:.10f} {salt2c:.10f} {salt2t0:.10f}",file=foutsn)
        
        keys=['num_lightcurves','num_spectra','num_sne']
        yamloutputdict={key.upper():trainingresult.__dict__[key] for key in keys}
        yamloutputdict['CPU_MINUTES']=(time.time()-self.initializationtime)/60
        yamloutputdict['ABORT_IF_ZERO']=1
        with open(f'{self.options.yamloutputfile}','w') as file: yaml.dump(yamloutputdict,file)
        
        return

    def salt2fit(self,sn,datadict):

        if 'FLT' not in sn.__dict__.keys():
            sn.FLT = sn.BAND[:]
        for flt in np.unique(sn.FLT):
            filtwave = self.kcordict[sn.SURVEY][flt]['filtwave']
            filttrans = self.kcordict[sn.SURVEY][flt]['filttrans']

            band = sncosmo.Bandpass(
                filtwave,
                filttrans,
                wave_unit=u.angstrom,name=flt)
            sncosmo.register(band, force=True)

        data = Table(rows=None,names=['mjd','band','flux','fluxerr','zp','zpsys'],
                     dtype=('f8','S1','f8','f8','f8','U5'),
                     meta={'t0':sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]})

        sysdict = {}
        for m,flt,flx,flxe in zip(sn.MJD,sn.FLT,sn.FLUXCAL,sn.FLUXCALERR):
            if self.kcordict[sn.SURVEY][flt]['magsys'] == 'BD17': sys = 'bd17'
            elif self.kcordict[sn.SURVEY][flt]['magsys'] == 'AB': sys = 'ab'
            else: sys = 'vega'
            if self.kcordict[sn.SURVEY][flt]['lambdaeff']/(1+float(sn.REDSHIFT_HELIO.split('+-')[0])) > 2000 and \
               self.kcordict[sn.SURVEY][flt]['lambdaeff']/(1+float(sn.REDSHIFT_HELIO.split('+-')[0])) < 9200 and\
               '-u' not in self.kcordict[sn.SURVEY][flt]['fullname']:
                data.add_row((m,flt,flx,flxe,
                              27.5+self.kcordict[sn.SURVEY][flt]['zpoff'],sys))
            sysdict[flt] = sys
        
        flux = sn.FLUXCAL
        salt2model = sncosmo.Model(source='salt2')
        salt2model.set(z=float(sn.REDSHIFT_HELIO.split()[0]))
        fitparams = ['t0', 'x0', 'x1', 'c']

        result, fitted_model = sncosmo.fit_lc(
            data, salt2model, fitparams,
            bounds={'t0':(sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)][0]-10, sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)][0]+10),
                    'z':(0.0,0.7),'x1':(-3,3),'c':(-0.3,0.3)})

        return result['parameters'][2],result['parameters'][3],result['parameters'][4],result['parameters'][1]
    
    def validate(self,outputdir,datadict,modelonly=False):

        # prelims
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=0.025, hspace=0)
        x0,x1,c,t0 = np.loadtxt(f'{outputdir}/salt3train_snparams.txt',unpack=True,usecols=[1,2,3,4])
        snid = np.genfromtxt(f'{outputdir}/salt3train_snparams.txt',unpack=True,dtype='str',usecols=[0])


        if self.options.fitsalt2:
            ValidateParams.main(f'{outputdir}/salt3train_snparams.txt',f'{outputdir}/saltparcomp.png')
        
        plotSALTModel.mkModelErrPlot(outputdir,outfile=f'{outputdir}/SALTmodelerrcomp.pdf',
                                     xlimits=[self.options.waverange[0],self.options.waverange[1]])

        plotSALTModel.mkModelPlot(outputdir,outfile=f'{outputdir}/SALTmodelcomp.png',
                                  xlimits=[self.options.waverange[0],self.options.waverange[1]],
                                  n_colorpars=self.options.n_colorpars,host_component=self.options.host_component)
        SynPhotPlot.plotSynthPhotOverStretchRange(
            '{}/synthphotrange.pdf'.format(outputdir),outputdir,'SDSS')
        SynPhotPlot.overPlotSynthPhotByComponent(
            '{}/synthphotoverplot.pdf'.format(outputdir),outputdir,'SDSS')

        snfiles_tot = np.array([])
        for j,snlist in enumerate(self.options.snlists.split(',')):
            snlist = os.path.expandvars(snlist)
            snfiles = np.genfromtxt(snlist,dtype='str')
            snfiles = np.atleast_1d(snfiles)
            snfiles_tot = np.append(snfiles_tot,snfiles)
            parlist,parameters = np.genfromtxt(
                f'{outputdir}/salt3_parameters.dat',unpack=True,dtype=str,skip_header=1)
            parameters = parameters.astype(float)
            CheckSALTParams.checkSALT(parameters,parlist,snfiles,snlist,outputdir,idx=j)

        # kcor files
        kcordict = {}
        for k in self.kcordict.keys():
            if k == 'default': continue
            for k2 in self.kcordict[k].keys():
                if k2 not in ['primarywave','snflux','BD17','filtwave','AB','Vega']:
                    if self.kcordict[k][k2]['magsys'] == 'AB': primarykey = 'AB'
                    elif self.kcordict[k][k2]['magsys'] == 'Vega': primarykey = 'Vega'
                    elif self.kcordict[k][k2]['magsys'] == 'VEGA': primarykey = 'Vega'
                    elif self.kcordict[k][k2]['magsys'] == 'BD17': primarykey = 'BD17'

                    kcordict[k2] = self.kcordict[k][k2]
                    kcordict[k2]['stdmag'] = synphot(
                        self.kcordict[k]['primarywave'],
                        self.kcordict[k][primarykey],
                        filtwave=self.kcordict[k][k2]['filtwave'],
                        filttp=self.kcordict[k][k2]['filttrans'],
                        zpoff=0) - self.kcordict[k][k2]['primarymag']

        from matplotlib.backends.backend_pdf import PdfPages
        plt.close('all')

        if modelonly:
            return
        
        pdf_pages = PdfPages(f'{outputdir}/lcfits.pdf')
        import matplotlib.gridspec as gridspec
        gs1 = gridspec.GridSpec(3, 5)
        gs1.update(wspace=0.0)
        i = 0
        
        # read in and save SALT2 files
        m0file='salt3_template_0.dat'
        m1file='salt3_template_1.dat'
        salt3phase,salt3wave,salt3flux = np.genfromtxt(f'{outputdir}/{m0file}',unpack=True)
        salt3m1phase,salt3m1wave,salt3m1flux = np.genfromtxt(f'{outputdir}/{m1file}',unpack=True)
        salt2phase,salt2wave,salt2flux = np.genfromtxt(f'{salt2dir}/salt2_template_0.dat',unpack=True)
        salt2m1phase,salt2m1wave,salt2m1flux = np.genfromtxt(f'{salt2dir}/salt2_template_1.dat',unpack=True)
        salt3phase = np.unique(salt3phase)
        salt3wave = np.unique(salt3wave)
        salt3flux = salt3flux.reshape([len(salt3phase),len(salt3wave)])
        salt3m1flux = salt3m1flux.reshape([len(salt3phase),len(salt3wave)])
        salt2phase = np.unique(salt2phase)
        salt2wave = np.unique(salt2wave)
        salt2m0flux = salt2flux.reshape([len(salt2phase),len(salt2wave)])
        salt2flux = salt2flux.reshape([len(salt2phase),len(salt2wave)])
        salt2m1flux = salt2m1flux.reshape([len(salt2phase),len(salt2wave)])

        saltdict = {'salt3phase':salt3phase,'salt3wave':salt3wave,'salt3flux':salt3flux,
                    'salt3m1phase':salt3m1phase,'salt3m1wave':salt3m1wave,'salt3m1flux':salt3m1flux,
                    'salt2phase':salt2phase,'salt2wave':salt2wave,'salt2m0flux':salt2m0flux,
                    'salt2m1phase':salt2m1phase,'salt2m1wave':salt2m1wave,'salt2m1flux':salt2m1flux}

            
        for j,snlist in enumerate(self.options.snlists.split(',')):
            snlist = os.path.expandvars(snlist)
            if not os.path.exists(snlist):
                print(f'SN list file {snlist} does not exist.  Checking {data_rootdir}/trainingdata/{snlist}')
                snlist = f'{data_rootdir}/trainingdata/{snlist}'%(data_rootdir,snlist)
                if not os.path.exists(snlist):
                    raise RuntimeError(f'SN list file {snlist} does not exist')

            tspec = time.time()
            if self.options.dospec:
                if self.options.binspec:
                    binspecres = self.options.binspecres
                else:
                    binspecres = None


                ValidateSpectra.compareSpectra(
                    snlist,self.options.outputdir,specfile=f'{self.options.outputdir}/speccomp_{j:.0f}.pdf',
                    maxspec=2000,base=self,verbose=self.verbose,datadict=datadict,binspecres=binspecres)
            log.info(f'plotting spectra took {time.time()-tspec:.1f}')
                
            snfiles = np.genfromtxt(snlist,dtype='str')
            snfiles = np.atleast_1d(snfiles)
            fitx1,fitc = False,False
            if self.options.n_components == 2:
                fitx1 = True
            if self.options.n_colorpars > 0:
                fitc = True

            if self.options.binspec:
                binspecres = self.options.binspecres
            else:
                binspecres = None
            
            datadict = readutils.rdAllData(snlist,self.options.estimate_tpk,
                                           dospec=self.options.dospec,
                                           peakmjdlist=self.options.tmaxlist,
                                           binspecres=binspecres,snparlist=self.options.snparlist,
                                           maxsn=self.options.maxsn)
                
            tlc = time.time()
            count = 0
            salt2_chi2tot,salt3_chi2tot = 0,0
            plotsnlist = []
            snfilelist = []
            for l in snfiles:
                if l.lower().endswith('.fits') or l.lower().endswith('.fits.gz'):

                    if '/' not in l:
                        l = '%s/%s'%(os.path.dirname(snlist),l)
                    if l.lower().endswith('.fits') and not os.path.exists(l) and os.path.exists('{}.gz'.format(l)):
                        l = '{}.gz'.format(l)
                    # get list of SNIDs
                    hdata = fits.getdata( l, ext=1 )
                    survey = fits.getval( l, 'SURVEY')
                    Nsn = fits.getval( l, 'NAXIS2', ext=1 )
                    snidlist = np.array([ int( hdata[isn]['SNID'] ) for isn in range(Nsn) ])

                    for sniditer in snidlist:
                        sn = snana.SuperNova(
                            snid=sniditer,headfitsfile=l,photfitsfile=l.replace('_HEAD.FITS','_PHOT.FITS'),
                            specfitsfile=None,readspec=False)
                        sn.SNID = str(sn.SNID)
                        plotsnlist.append(sn)
                        snfilelist.append(l)
                
                else:
                    if '/' not in l:
                        l = f'{os.path.dirname(snlist)}/{l}'
                    sn = snana.SuperNova(l)
                    sn.SNID = str(sn.SNID)
                    if not sn.SNID in datadict:
                        continue
                    plotsnlist.append(sn)
                    snfilelist.append(l)

            for sn,l in zip(plotsnlist,snfilelist):

                if not i % 12:
                    fig = plt.figure()
                try:
                    ax1 = plt.subplot(gs1[i % 15]); ax2 = plt.subplot(gs1[(i+1) % 15]); ax3 = plt.subplot(gs1[(i+2) % 15]); ax4 = plt.subplot(gs1[(i+3) % 15]); ax5 = plt.subplot(gs1[(i+4) % 15])
                except:
                    import pdb; pdb.set_trace()


                if sn.SNID not in snid:
                    log.warning(f'sn {sn.SNID} not in output files')
                    continue
                x0sn,x1sn,csn,t0sn = \
                    x0[snid == sn.SNID][0],x1[snid == sn.SNID][0],\
                    c[snid == sn.SNID][0],t0[snid == sn.SNID][0]
                if not fitc: csn = 0
                if not fitx1: x1sn = 0

                if '.fits' in l.lower():
                    snidval = int(sn.SNID)
                else:
                    snidval = None
                salt2chi2,salt3chi2 = ValidateLightcurves.customfilt(
                    f'{outputdir}/lccomp_{sn.SNID}.png',l,outputdir,
                    t0=t0sn,x0=x0sn,x1=x1sn,c=csn,fitx1=fitx1,fitc=fitc,
                    bandpassdict=self.kcordict,n_components=self.options.n_components,
                    ax1=ax1,ax2=ax2,ax3=ax3,ax4=ax4,ax5=ax5,saltdict=saltdict,n_colorpars=self.options.n_colorpars,
                    snid=snidval)
                salt2_chi2tot += salt2chi2
                salt3_chi2tot += salt3chi2
                if i % 12 == 8:
                    pdf_pages.savefig()
                    plt.close('all')
                else:
                    for ax in [ax1,ax2,ax3,ax4]:
                        ax.xaxis.set_ticklabels([])
                        ax.set_xlabel(None)
                i += 4
                count += 1
            log.info(f'plotted light curves for {count} SNe')
            log.info(f'total chi^2 is {salt2_chi2tot:.1f} for SALT2 and {salt3_chi2tot:.1f} for SALT3')
        if not i %12 ==0:
            pdf_pages.savefig()
        pdf_pages.close()
        log.info(f'plotting light curves took {time.time()-tlc:.1f}')
        
    def main(self,returnGN=False):
        try:
            stage='initialization'
            if not len(self.surveylist):
                raise RuntimeError('surveys are not defined - see documentation')
            
            
            phasebins=np.linspace(*self.options.phaserange,int((self.options.phaserange[1]-self.options.phaserange[0])/self.options.phasesplineres)+1,True)
            wavebins=np.linspace(*self.model_options.waverange,int((self.options.waverange[1]-self.options.waverange[0])/self.options.wavesplineres)+1,True)
            datadensity.datadensityplot(path.join(self.options.outputdir,'datadensity.pdf') ,phasebins,wavebins,datadict,self.kcordict)
            # fit the model - initial pass
            if self.options.stage == "all" or self.options.stage == "train":
                # read the data
                stage='training'

                if not returnGN:
                    trainingresult,chain,loglikes,saltfitter = self.fitSALTModel(datadict,returnGN=returnGN)
                else:
                    fitter,saltfitter,modelpars = self.fitSALTModel(datadict,returnGN=returnGN)
                    return fitter,saltfitter,modelpars

                if self.options.errors_from_bootstrap:
                    if self.options.bootstrap_batch_mode:
                        fitter,saltfitter,modelpars = self.bootstrapSALTModel_batch(datadict,trainingresult,saltfitter,returnGN=returnGN)
                    else:
                        fitter,saltfitter,modelpars = self.bootstrapSALTModel(datadict,trainingresult,returnGN=returnGN)
                
                stage='output'
                # write the output model - M0, M1, c
                self.wrtoutput(self.options.outputdir,trainingresult,chain,loglikes,datadict)
            log.info('successful SALT2 training!  Output files written to %s'%self.options.outputdir)
            if not self.options.skip_validation:
                if self.options.stage == "all" or self.options.stage == "validate":
                    stage='validation'
                    if self.options.validate_modelonly:
                        self.validate(self.options.outputdir,datadict,modelonly=True)
                    else:
                        self.validate(self.options.outputdir,datadict,modelonly=False)
        except:
            log.exception(f'Exception raised during {stage}')
            if stage != 'validation':
                raise RuntimeError("Training exited unexpectedly")
        
    def createGaussNewton(self):

        fitter,saltfitter,modelpars = self.main(returnGN=True)

        # trainingresult,message = fitter.gaussnewton(
        #     saltfitter,modelpars,
        #     maxiter,getdatauncertainties=True)

        #import pdb; pdb.set_trace()

        return fitter,saltfitter,modelpars


def initialize_parameters(user_options,model_options,datadict)
    
    inithsiaofile = f'{init_rootdir}/hsiao07.dat'
    
    for file in ['initbfilt','initm0modelfile', 'initm1modelfile']:
        pathspecified=getattr(user_options,file)
        if pathspecified:
            if not os.path.exists(pathspecified):
                setattr(user_options,file, f'{init_rootdir}/{pathspecified}')
    
    if self.options.initm0modelfile and not os.path.exists(self.options.initm0modelfile):
        raise RuntimeError('model initialization file not found in local directory or %s'%init_rootdir)

    init_options = {'phaserange':model_options.phaserange,'waverange':model_options.waverange,
                    'phasesplineres':model_options.phasesplineres,'wavesplineres':model_options.wavesplineres,
                    'phaseinterpres':model_options.phaseinterpres,'waveinterpres':model_options.waveinterpres,
                    'normalize':True,'order':model_options.interporder,'use_snpca_knots':model_options.use_snpca_knots}
            
    phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_hsiao(
        inithsiaofile,self.options.initbfilt,_flatnu,**init_options)
    if self.options.host_component:
        mhostknots = m0knots*0.01 # 1% of M0?  why not
    
    if self.options.initsalt2model:
        if self.options.initm0modelfile =='':
            self.options.initm0modelfile=f'{init_rootdir}/salt2_template_0.dat'
        if self.options.initm1modelfile  =='':
            self.options.initm1modelfile=f'{init_rootdir}/salt2_template_1.dat'

    if self.options.initm0modelfile and self.options.initm1modelfile:
        if self.options.initsalt2model:
            phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_salt2(
                m0file=self.options.initm0modelfile,m1file=self.options.initm1modelfile,
                Bfilt=self.options.initbfilt,flatnu=_flatnu,**init_options)
        else:
            phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_kaepora(
                self.options.initm0modelfile,self.options.initm1modelfile,
                Bfilt=self.options.initbfilt,flatnu=_flatnu,**init_options)

        
    init_options['phasesplineres'] = model_options.error_snake_phase_binsize
    init_options['wavesplineres'] = model_options.error_snake_wave_binsize
    init_options['order']=model_options.errinterporder
    init_options['n_colorscatpars']=model_options.n_colorscatpars
        
    
    del init_options['use_snpca_knots']
    if user_options.initsalt2var:
        errphaseknotloc,errwaveknotloc,m0varknots,m1varknots,m0m1corrknots,clscatcoeffs=init_errs(
             *['%s/%s'%(init_rootdir,x) for x in ['salt2_lc_relative_variance_0.dat','salt2_lc_relative_covariance_01.dat','salt2_lc_relative_variance_1.dat','salt2_lc_dispersion_scaling.dat','salt2_color_dispersion.dat']],**init_options)
    else:
        #errphaseknotloc,errwaveknotloc,m0varknots,m1varknots,m0m1corrknots,clscatcoeffs=init_errs(**init_options)
        init_options['phase'] = phase
        init_options['wave'] = wave
        init_options['phaseknotloc'] = phaseknotloc
        init_options['waveknotloc'] = waveknotloc
        init_options['m0knots'] = m0knots
        init_options['m1knots'] = m1knots
        if self.options.host_component:
            init_options['mhostknots'] = mhostknots

        if not model_options.host_component:
            errphaseknotloc,errwaveknotloc,m0varknots,m1varknots,m0m1corrknots,clscatcoeffs=init_errs_percent(**init_options)
        else:
            errphaseknotloc,errwaveknotloc,m0varknots,m1varknots,mhostvarknots,m0m1corrknots,clscatcoeffs=init_errs_percent(**init_options)
            
    # number of parameters
    n_phaseknots,n_waveknots = len(phaseknotloc)-self.options.interporder-1,len(waveknotloc)-self.options.interporder-1
    n_errphaseknots,n_errwaveknots = len(errphaseknotloc)-model_options.errinterporder-1,len(errwaveknotloc)-model_options.errinterporder-1
    n_sn = len(datadict.keys())

    # set up the list of parameters
    parlist = np.array(['m0']*(n_phaseknots*n_waveknots))
    if model_options.n_components >= 2:
        parlist = np.append(parlist,['m1']*(n_phaseknots*n_waveknots))
    if model_options.host_component:
        parlist = np.append(parlist,['mhost']*(n_phaseknots*n_waveknots))
    if model_options.n_colorpars:
        parlist = np.append(parlist,['cl']*model_options.n_colorpars)
    if model_options.error_snake_phase_binsize and model_options.error_snake_wave_binsize:
        for i in range(model_options.n_components): parlist = np.append(parlist,['modelerr_{}'.format(i)]*n_errphaseknots*n_errwaveknots)
        if model_options.host_component:
            parlist = np.append(parlist,['modelerr_host']*len(mhostvarknots))
            parlist = np.append(parlist,['modelcorr_0host']*len(mhostvarknots))
            parlist = np.append(parlist,['modelcorr_1host']*len(mhostvarknots))
        if model_options.n_components == 2:
            parlist = np.append(parlist,['modelcorr_01']*n_errphaseknots*n_errwaveknots)
    
    if model_options.n_colorscatpars:
        parlist = np.append(parlist,['clscat']*(model_options.n_colorscatpars))

    # SN parameters
    if not model_options.host_component:
        for k in datadict.keys():
            parlist = np.append(parlist,[f'x0_{k}',f'x1_{k}',f'c_{k}'])
    else:
        for k in datadict.keys():
            parlist = np.append(parlist,[f'x0_{k}',f'x1_{k}',f'xhost_{k}',f'c_{k}'])

    if user_options.specrecallist:
        spcrcldata = at.Table.read(user_options.specrecallist,format='ascii')
        
    # spectral params
    for sn in datadict.keys():
        specdata=datadict[sn].specdata
        photdata=datadict[sn].photdata
        for k in specdata.keys():
            if not self.options.specrecallist:
                order=self.options.n_min_specrecal+int(np.log((specdata[k].wavelength.max() - \
                    specdata[k].wavelength.min())/self.options.specrange_wavescale_specrecal) + \
                    len(datadict[sn].filt)* self.options.n_specrecal_per_lightcurve)
            else:
                spcrclcopy = spcrcldata[spcrcldata['SNID'] == sn]
                order = int(spcrclcopy['ncalib'][spcrclcopy['N'] == k+1])
            recalParams=[f'specx0_{sn}_{k}']+[f'specrecal_{sn}_{k}']*(order-1)
            parlist=np.append(parlist,recalParams)
    # initial guesses
    n_params=parlist.size
    guess = np.zeros(parlist.size)
    if self.options.resume_from_outputdir:
        log.info(f"resuming from output directory {self.options.resume_from_outputdir}")
        
        names=None
        for possibleDir in [self.options.resume_from_outputdir,self.options.outputdir]:
            for possibleFile in ['salt3_parameters_unscaled.dat','salt3_parameters.dat']:   
                if names is None:
                    try:
                        names,pars = np.loadtxt(path.join(possibleDir,possibleFile),unpack=True,skiprows=1,dtype="U30,f8")
                        break
                    except:
                        continue
        if self.options.resume_from_gnhistory:
            with open(f"{self.options.resume_from_gnhistory}/gaussnewtonhistory.pickle",'rb') as fin:
                data = pickle.load(fin)
                pars = data[-1][0]
        for key in np.unique(parlist):
            try:
                guess[parlist == key] = pars[names == key]
            except:
                print(key)
                log.critical(f'Problem while initializing parameter {key} from previous training')
                sys.exit(1)
    else:
        m0knots[m0knots == 0] = 1e-4
        guess[parlist == 'm0'] = m0knots
        for i in range(3): guess[parlist == 'modelerr_{}'.format(i)] = 1e-6 
        if self.options.n_components >= 2:
            guess[parlist == 'm1'] = m1knots
        if self.options.host_component:
            guess[parlist == 'mhost'] = mhostknots
        if self.options.n_colorpars:
            if self.options.initsalt2model:
                if self.options.n_colorpars == 4:
                    guess[parlist == 'cl'] = [-0.504294,0.787691,-0.461715,0.0815619]
                else:
                    clwave = np.linspace(self.options.waverange[0],self.options.waverange[1],1000)
                    salt2cl = SALT2ColorLaw([2800.,7000.], [-0.504294,0.787691,-0.461715,0.0815619])(clwave)
                    def bestfit(p):
                        cl_init = SALT2ColorLaw(self.options.colorwaverange, p)(clwave)
                        return cl_init-salt2cl

                    md = least_squares(bestfit,[0,0,0,0,0])
                    if 'termination conditions are satisfied' not in md.message and \
                       'termination condition is satisfied' not in md.message:
                        
                        raise RuntimeError('problem initializing color law!')
                    guess[parlist == 'cl'] = md.x
            else:
                guess[parlist == 'cl'] =[0.]*self.options.n_colorpars 
        if self.options.n_colorscatpars:

            guess[parlist == 'clscat'] = clscatcoeffs

        guess[(parlist == 'm0') & (guess < 0)] = 1e-4
        
        guess[parlist=='modelerr_0']=m0varknots
        guess[parlist=='modelerr_1']=m1varknots
        if self.options.host_component: guess[parlist=='modelerr_host']=1e-9 # something small...  #mhostvarknots
        guess[parlist=='modelcorr_01']=m0m1corrknots

        # if SN param list is provided, initialize with these params
        if self.options.snparlist:
            snpar = Table.read(self.options.snparlist,format='ascii')
            snpar['SNID'] = snpar['SNID'].astype(str)

        for sn in datadict.keys():
            if self.options.snparlist:
                # hacky matching, but SN names are a mess as usual
                iSN = ((sn == snpar['SNID']) | ('sn'+sn == snpar['SNID']) |
                       ('sn'+sn.lower() == snpar['SNID']) | (sn+'.0' == snpar['SNID']))
                if len(snpar['SNID'][iSN]) > 1:
                    raise RuntimeError(f"found duplicate in parameter list for SN {snpar['SNID'][iSN][0]}")
                if len(snpar[iSN]):
                    guess[parlist == 'x0_%s'%sn] = snpar['x0'][iSN]
                    guess[parlist == 'x1_%s'%sn] = snpar['x1'][iSN]
                    if self.options.host_component:
                        guess[parlist == f'xhost_{sn}'] = snpar['xhost'][iSN]

                    guess[parlist == 'c_%s'%sn] = snpar['c'][iSN]
                else:
                    log.warning(f'SN {sn} not found in SN par list {self.options.snparlist}')
                    guess[parlist == 'x0_%s'%sn] = 10**(-0.4*(cosmo.distmod(datadict[sn].zHelio).value-19.36-10.635))

            elif 'SIM_SALT2x1' in datadict[sn].__dict__.keys():
                # simulated samples need an initialization list also
                # initializing to sim. values is not the best but running SNANA fits adds a lot of overhead
                log.info(f'initializing parameters using simulated values for SN {sn}')
                guess[parlist == 'x0_%s'%sn] = datadict[sn].SIM_SALT2x0
                guess[parlist == 'x1_%s'%sn] = datadict[sn].SIM_SALT2x1
                guess[parlist == 'c_%s'%sn] = datadict[sn].SIM_SALT2c
            else:
                guess[parlist == 'x0_%s'%sn] = 10**(-0.4*(cosmo.distmod(datadict[sn].zHelio).value-19.36-10.635))

                
            for k in datadict[sn].specdata : 
                guess[parlist==f'specx0_{sn}_{k}']= guess[parlist == 'x0_%s'%sn]

        # let's redefine x1 before we start
        ratio = RatioToSatisfyDefinitions(phase,wave,self.kcordict,[m0,m1])
        ix1 = np.array([i for i, si in enumerate(parlist) if si.startswith('x1')],dtype=int)
        guess[ix1]/=1+ratio*guess[ix1]
        guess[ix1]-=np.mean(guess[ix1])
        x1std = np.std(guess[ix1])
        if x1std == x1std and x1std != 0.0:
            guess[ix1]/= x1std
            
        def specflux(obsphase,obswave,m0phase,m0wave,m0flux,m1flux,colorlaw,z,x0,x1,c,mwebv):

            modelflux = x0*(m0flux + x1*m1flux)*1e-12/(1+z)

            m0interp = interp1d(np.unique(m0phase)*(1+z),m0flux*1e-12/(1+z),axis=0,
                                kind='nearest',bounds_error=False,fill_value="extrapolate")
            m0phaseinterp = m0interp(obsphase)
            m0interp = np.interp(obswave,np.unique(m0wave)*(1+z),m0phaseinterp)

            m1interp = interp1d(np.unique(m0phase)*(1+z),m1flux*1e-12/(1+z),axis=0,
                                kind='nearest',bounds_error=False,fill_value="extrapolate")
            m1phaseinterp = m1interp(obsphase)
            m1interp = np.interp(obswave,np.unique(m0wave)*(1+z),m1phaseinterp)


            intphase = interp1d(np.unique(m0phase)*(1+z),modelflux,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
            modelflux_phase = intphase(obsphase)
            intwave = interp1d(np.unique(m0wave)*(1+z),modelflux_phase,kind='nearest',bounds_error=False,fill_value="extrapolate")
            modelflux_wave = intwave(obswave)
            modelflux_wave = x0*(m0interp + x1*m1interp)
            mwextcurve = 10**(-0.4*extinction.fitzpatrick99(obswave.astype(float),mwebv*3.1))
            modelflux_wave *= mwextcurve

            return modelflux_wave


        # spectral params
        for sn in datadict.keys():
            specdata=datadict[sn].specdata
            photdata=datadict[sn].photdata
            for k in specdata.keys():
                order=(parlist == 'specrecal_{}_{}'.format(sn,k)).sum()
                
                pow=(order)-np.arange(order)
                recalCoord=(specdata[k].wavelength-np.mean(specdata[k].wavelength))/2500
                drecaltermdrecal=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]

                zHel,x0,x1,c = datadict[sn].zHelio,guess[parlist == f'x0_{sn}'],guess[parlist == f'x1_{sn}'],guess[parlist == f'c_{sn}']
                mwebv = datadict[sn].MWEBV
                colorlaw = SALT2ColorLaw(self.options.colorwaverange,guess[parlist == 'cl'])


                uncalledModel = specflux(specdata[k].tobs,specdata[k].wavelength,phase,wave,
                                         m0,m1,colorlaw,zHel,x0,x1,c,mwebv=mwebv)
    
                def recalpars(x):
                    recalexp=np.exp((drecaltermdrecal*x[1:][np.newaxis,:]).sum(axis=1))
                    return (x[0]*uncalledModel*recalexp - specdata[k].flux)/specdata[k].fluxerr

                md = least_squares(recalpars,[np.median(specdata[k].flux)/np.median(uncalledModel)]+list(guess[parlist == 'specrecal_{}_{}'.format(sn,k)]))

                guess[parlist == f'specx0_{sn}_{k}' ]= md.x[0]*x0
                guess[parlist == f'specrecal_{sn}_{k}'] = md.x[1:]


    if self.options.fix_salt2components_initdir:
        log.info(f"resuming from output directory {self.options.fix_salt2components_initdir}")
        
        names=None
        for possibleDir in [self.options.fix_salt2components_initdir]:
            for possibleFile in ['salt3_parameters_unscaled.dat','salt3_parameters.dat']:   
                if names is None:
                    try:
                        names,pars = np.loadtxt(path.join(possibleDir,possibleFile),unpack=True,skiprows=1,dtype="U30,f8")
                        break
                    except:
                        continue
        if self.options.resume_from_gnhistory:
            with open(f"{self.options.resume_from_gnhistory}/gaussnewtonhistory.pickle",'rb') as fin:
                data = pickle.load(fin)
                pars = data[-1][0]
        for key in np.unique(parlist):
            try:
                guess[parlist == key] = pars[names == key]
            except:
                log.info(f'Could not initializing parameter {key} from previous training')
                pass

                
    return parlist,guess,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc


def initialize_all_data(user_options,model_options,surveydict):

        tkstart = time.time()
        kcordict=readutils.rdkcor(surveydict, filters_use_lastchar_only = user_options.filters_use_lastchar_only  , calibrationshiftfile=user_options.calibrationshiftfile, calib_survey_ignore= user_options.calib_survey_ignore)
        log.info(f'took {time.time()-tkstart:.3f} to read in kcor files')
        # TODO: ASCII filter files
            
        if model_options.binspec:
            binspecres = model_options.binspecres
        else:
            binspecres = None

        tdstart = time.time()
        datadict = readutils.rdAllData(user_options.snlists,user_options.estimate_tpk,
                                       dospec=user_options.dospec,
                                       peakmjdlist=user_options.tmaxlist,
                                       binspecres=binspecres,snparlist=user_options.snparlist,maxsn=user_options.maxsn)
        log.info(f'took {time.time()-tdstart:.3f} to read in data files')
        tcstart = time.time()

        datadict = self.mkcuts(datadict)[0]
        log.info(f'took {time.time()-tcstart:.3f} to apply cuts')
        return kcordict,datadict
        
def __main__(self):

    user_options,model_options,surveydict=configoptions.initialconfiguration()
    kcordict,datadict=initialize_all_data(user_options,model_options,surveydict )

