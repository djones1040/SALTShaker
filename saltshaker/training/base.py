import os,sys
import argparse
import configparser
import numpy as np
from saltshaker.config import config_rootdir
from saltshaker.config.configparsing import *

from saltshaker.util.specSynPhot import getColorsForSN
from argparse import SUPPRESS
from saltshaker.training.priors import __priors__


import logging

log=logging.getLogger(__name__)




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
                
                wrapaddingargument=generateerrortolerantaddmethod(parser)
                        
                # input files
                successful=True
                successful=successful&wrapaddingargument(config,'iodata','calibrationshiftfile',type=str,action=FullPaths,default='',
                                                        help='file containing a list of changes to zeropoint and central wavelength of filters by survey')
                successful=successful&wrapaddingargument(config,'iodata','calib_survey_ignore',type=boolean_string,
                                                        help='if True, ignore survey names when applying shifts to filters')
                successful=successful&wrapaddingargument(config,'iodata','loggingconfig','loggingconfigfile',  type=str,action=FullPaths,default='',
                                                        help='logging config file')
                successful=successful&wrapaddingargument(config,'iodata','modelconfig','modelconfigfile',  type=str,action=FullPaths,
                                                        help='Configuration file describing the construction of the model')
                successful=successful&wrapaddingargument(config,'iodata','trainingconfig','trainingconfigfile',  type=str,action=FullPaths,
                                                        help='Configuration file providing hyperparameters to the optimizer',default=None)
                                                        
                                                        
                successful=successful&wrapaddingargument(config,'iodata','snlists','snlistfiles',  type=str,action=FullPaths,
                                                        help="""list of SNANA-formatted SN data files, including both photometry and spectroscopy. Can be multiple comma-separated lists. (default=%(default)s)""")
                successful=successful&wrapaddingargument(config,'iodata','snparlist', 'snparlistfile', type=str,action=FullPaths,
                                                        help="""optional list of initial SN parameters.  Needs columns SNID, zHelio, x0, x1, c""")
                successful=successful&wrapaddingargument(config,'iodata','specrecallist','specrecallistfile',  type=str,action=FullPaths,
                                                        help="""optional list giving number of spectral recalibration params.  Needs columns SNID, N, phase, ncalib where N is the spectrum number for a given SN, starting at 1""")
                successful=successful&wrapaddingargument(config,'iodata','tmaxlist','tmaxlistfile',     type=str,action=FullPaths,
                                                        help="""optional space-delimited list with SN ID, tmax, tmaxerr (default=%(default)s)""")
                successful=successful&wrapaddingargument(config,'iodata','trainingcachefile','trainingcache', default='',    type=str,action=FullPaths,
                                                        help="""Optional file path used to store a cached version of the training data pre-computed to speed up subsequent runs. If this file exists already, it will be loaded in, otherwise it will be written to""")
        
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
                successful=successful&wrapaddingargument(config,'iodata', 'spectra_cut',  type=float , help='Makes cut so that only spectra of certain S/N are used ')


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

                successful=successful&wrapaddingargument(config,'trainparams','fitsalt2',  type=boolean_string,
                                                        help='fit SALT2 as a validation check (default=%(default)s)')

                successful=successful&wrapaddingargument(config,'trainparams','regularize',     type=boolean_string,
                                                        help='turn on regularization if set (default=%(default)s)')

                successful=successful&wrapaddingargument(config,'trainparams','optimizer',     type=str,
                                                        help="Choice of optimizer to use (default=%(default)s)")
                successful=successful&wrapaddingargument(config,'trainparams','n_repeat',  type=int,
                                                help='repeat optimization n times (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','fitprobmin',     type=float,
                                                        help="Minimum FITPROB for including SNe (default=%(default)s)")
                successful=successful&wrapaddingargument(config,'trainparams','errors_from_hessianapprox',     type=boolean_string,
                                                        help="if set, get model surface errors from an approximated Hessian matrix (default=%(default)s)")
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
                successful=successful&wrapaddingargument(config,'trainparams','fixedparams','fixedparameters',     type=str, default='',
                                                        help="Comma-separated list of parameters to be kept fixed (default=%(default)s)")

                successful=successful&wrapaddingargument(config,'trainparams','photometric_zeropadding_batches', type=int,
                                                        help='Number of batches to divide the photometric data into when zero-padding. Increasing this value may improve memory performance at the cost of speed (default=%(default)s)')
                successful=successful&wrapaddingargument(config,'trainparams','spectroscopic_zeropadding_batches', type=int,
                                                help='Number of batches to divide the spectroscopic data into when zero-padding. Increasing this value may improve memory performance at the cost of speed (default=%(default)s)')

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

            filters = []
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
                     #import pdb; pdb.set_trace()
                    if np.median(spectrum.flux/spectrum.fluxerr) <self.options.spectra_cut:
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
                    if flt not in filters:
                        filters += [flt]
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
            log.info(f'Unique filters: {np.unique(filters)}')
            return outdict,cutdict

        def filter_select(self,survey,flt):
                if flt in self.options.__dict__[f"{survey.split('(')[0]}_ignore_filters"].replace(' ','').split(','):
                    return  False

                else:
                    lambdaeff = self.kcordict[survey][flt]['lambdaeff']
                    if lambdaeff < self.options.filtercen_obs_waverange[0] or \
                                 lambdaeff > self.options.filtercen_obs_waverange[1] :
                         return False

                return True 
                # end filter_select
