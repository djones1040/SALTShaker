#!/usr/bin/env python
# D. Jones, R. Kessler - 8/31/18
from __future__ import print_function


import argparse
import configparser
import numpy as np
import sys
import multiprocessing
import pickle
import copy
import yaml
import time
initializationtime=time.time()
import matplotlib as mpl
mpl.use('agg')
import pylab as plt

import os
from os import path
import subprocess

from scipy.interpolate import interp1d
from scipy.special import factorial
from scipy.linalg import lstsq
from scipy.optimize import minimize, least_squares, differential_evolution

from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from astropy.table import Table

from saltshaker.util import snana,readutils
from saltshaker.util.estimate_tpk_bazin import estimate_tpk_bazin
from saltshaker.util.txtobj import txtobj
from saltshaker.util.specSynPhot import getScaleForSN
from saltshaker.util.specrecal import SpecRecal
from saltshaker.util.synphot import synphot
from saltshaker.util.example_data_utils import download_dir

from saltshaker.initfiles import init_rootdir
from saltshaker.initfiles import init_rootdir as salt2dir

from saltshaker.training.init_hsiao import init_hsiao, init_kaepora, init_errs,init_errs_percent,init_custom,init_salt2
from saltshaker.training.base import TrainSALTBase
from saltshaker.training import saltresids
from saltshaker.training import optimizers
from saltshaker.training import colorlaw

from saltshaker.validation import ValidateParams,datadensity
from saltshaker.validation import ValidateLightcurves
from saltshaker.validation import ValidateSpectra
from saltshaker.validation import ValidateModel
from saltshaker.validation import CheckSALTParams
from saltshaker.validation.figs import plotSALTModel
from saltshaker.validation import SynPhotPlot

from saltshaker.data import data_rootdir

from saltshaker.config import config_rootdir,loggerconfig

import extinction
import sncosmo
from sncosmo.salt2utils import SALT2ColorLaw
from sncosmo.constants import HC_ERG_AA

_example_data_url = "https://github.com/djones1040/SALTShaker/raw/main/examples/saltshaker-latest-training.tgz"
_flatnu=f'{init_rootdir}/flatnu.dat'

# validation utils

import logging
log=logging.getLogger(__name__)

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


class TrainSALT(TrainSALTBase):

        
    def initialParameters(self,datadict):
        from saltshaker.initfiles import init_rootdir
        self.options.inithsiaofile = f'{init_rootdir}/hsiao07.dat'
        if not os.path.exists(self.options.initbfilt):
             self.options.initbfilt = f'{init_rootdir}/{self.options.initbfilt}'
        if self.options.initm0modelfile and not os.path.exists(self.options.initm0modelfile):
            if self.options.initm0modelfile:
                self.options.initm0modelfile = f'{init_rootdir}/{self.options.initm0modelfile}'
            if self.options.initm1modelfile:
                self.options.initm1modelfile = f'{init_rootdir}/{self.options.initm1modelfile}'
        
        if self.options.initm0modelfile and not os.path.exists(self.options.initm0modelfile):
            raise RuntimeError('model initialization file not found in local directory or %s'%init_rootdir)

        # initial guesses
        init_options = {'phaserange':self.options.phaserange,'waverange':self.options.waverange,
                        'phasesplineres':self.options.phasesplineres,'wavesplineres':self.options.wavesplineres,
                        'phaseinterpres':self.options.phaseinterpres,'waveinterpres':self.options.waveinterpres,
                        'normalize':True,'order':self.options.bsorder,'use_snpca_knots':self.options.use_snpca_knots}

        phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_hsiao(
            self.options.inithsiaofile,self.options.initbfilt,_flatnu,**init_options)
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
        #zero out the flux and the 1st derivative at the start of the phase range
        m0knots[:(waveknotloc.size-self.options.bsorder-1) * 2]=0
        m1knots[:(waveknotloc.size-self.options.bsorder-1) * 2]=0
        
        init_options['phasesplineres'] = self.options.error_snake_phase_binsize
        init_options['wavesplineres'] = self.options.error_snake_wave_binsize
        init_options['order']=self.options.errbsorder
        init_options['n_colorscatpars']=self.options.n_colorscatpars
            
        
        del init_options['use_snpca_knots']
        if self.options.initsalt2var:
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

            if not self.options.host_component:
                errphaseknotloc,errwaveknotloc,m0varknots,m1varknots,m0m1corrknots,clscatcoeffs=init_errs_percent(**init_options)
            else:
                errphaseknotloc,errwaveknotloc,m0varknots,m1varknots,mhostvarknots,m0m1corrknots,clscatcoeffs=init_errs_percent(**init_options)

        # number of parameters
        n_phaseknots,n_waveknots = len(phaseknotloc)-self.options.bsorder-1,len(waveknotloc)-self.options.bsorder-1
        n_errphaseknots,n_errwaveknots = len(errphaseknotloc)-self.options.errbsorder-1,len(errwaveknotloc)-self.options.errbsorder-1
        n_sn = len(datadict.keys())
        parlist=[]
        # set up the list of parameters
        for i in range(self.options.n_components):
            parlist = np.append(parlist,['m'+str(i)]*(n_phaseknots*n_waveknots))
        if self.options.host_component:
            parlist = np.append(parlist,['mhost']*(n_phaseknots*n_waveknots))
        if self.options.n_colorpars:
            parlist = np.append(parlist,[[f'cl{i}']*num for i,num in enumerate(self.options.n_colorpars)])
        if self.options.error_snake_phase_binsize and self.options.error_snake_wave_binsize:
            for i in range(self.options.n_errorsurfaces): 
                parlist = np.append(parlist,['modelerr_{}'.format(i)]*n_errphaseknots*n_errwaveknots)
                for j in range(i):
                    parlist = np.append(parlist,[f'modelcorr_{j}{i}']*n_errphaseknots*n_errwaveknots)
            if self.options.host_component:
                parlist = np.append(parlist,['modelerr_host']*len(mhostvarknots))
                parlist = np.append(parlist,['modelcorr_0host']*len(mhostvarknots))
                parlist = np.append(parlist,['modelcorr_1host']*len(mhostvarknots))
        
        if self.options.n_colorscatpars:
            parlist = np.append(parlist,['clscat']*(self.options.n_colorscatpars))

        # SN parameters
        parlist = np.append(parlist,sum( [[f'x{i}_{k}' for i in range(self.options.n_components)]
            +[f'c{i}_{k}' for i in range(len(self.options.n_colorpars)) ] for k in datadict.keys()],[]))

        if self.options.host_component:
            parlist = np.append(parlist,[f'xhost_{k}' for k in datadict.keys()])

        if self.options.specrecallist:
            spcrcldata = Table.read(self.options.specrecallist,format='ascii')
            
        # spectral params
        for sn in datadict.keys():
            specdata=datadict[sn].specdata
            photdata=datadict[sn].photdata
            for k in specdata.keys():
                order=self.options.n_min_specrecal+int(np.log((specdata[k].wavelength.max() - \
                    specdata[k].wavelength.min())/self.options.specrange_wavescale_specrecal) + \
                    len(datadict[sn].filt)* self.options.n_specrecal_per_lightcurve)
                order=min(max(order,self.options.n_min_specrecal ), self.options.n_max_specrecal)

                # save the order as part of the specrecal list
                if not self.options.specrecallist or sn not in spcrcldata['SNID'] or k+1 not in spcrcldata['N'][spcrcldata['SNID'] == sn]:
                    datadict[sn].specdata[k].n_specrecal = order
                #if datadict[sn].specdata[k].n_specrecal is None:
                #    import pdb; pdb.set_trace()
                recalParams=[f'specx0_{sn}_{k}']+[f'specrecal_{sn}_{k}']*(order-1)
                parlist=np.append(parlist,recalParams)

        modelconfiguration=saltresids.saltconfiguration(parlist=parlist,phaseknotloc =phaseknotloc ,waveknotloc=waveknotloc,
            errphaseknotloc=errphaseknotloc,errwaveknotloc=errwaveknotloc)
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
            if self.options.n_components >= 3:
            
                guess[parlist=='m2'] =( (np.arange(m0knots.size)< (n_waveknots*  (n_phaseknots//6)))
                                       &  (np.arange(m0knots.size)> (n_waveknots* 1 ))  )*np.std(m0knots)*.2
            if self.options.host_component:
                guess[parlist == 'mhost'] = mhostknots
            if self.options.n_colorpars:
                if self.options.initsalt2model:
                    #if len(self.options.n_colorpars)>1: raise ValueError('Multiple color laws specified with initsalt2model option')
                    if self.options.n_colorpars == [4]:
                        guess[parlist == 'cl'] = [-0.504294,0.787691,-0.461715,0.0815619]
                    else:
                        clwave = np.linspace(self.options.waverange[0],self.options.waverange[1],1000)
                        salt2cl = SALT2ColorLaw([2800.,7000.], [-0.504294,0.787691,-0.461715,0.0815619])(clwave)
                        initcolorlaw=colorlaw.getcolorlaw(self.options.colorlaw_function[0])(self.options.n_colorpars[0],self.options.colorwaverange)
                        def bestfit(p):
                            cl_init = initcolorlaw(1, p,clwave)
                            return cl_init-salt2cl

                        md = least_squares(bestfit,np.zeros(self.options.n_colorpars[0]))
                        if 'termination conditions are satisfied' not in md.message and \
                           'termination condition is satisfied' not in md.message:
                            
                            raise RuntimeError('problem initializing color law!')
                        guess[parlist == 'cl0'] = md.x
                else:
                    guess[parlist == 'cl0'] =[0.]*self.options.n_colorpars 
            if self.options.n_colorscatpars:

                guess[parlist == 'clscat'] = clscatcoeffs

            guess[(parlist == 'm0') & (guess < 0)] = 1e-4
            
            guess[parlist=='modelerr_0']=m0varknots
            if self.options.n_errorsurfaces > 1:
                guess[parlist=='modelerr_1']=m1varknots
                guess[parlist=='modelcorr_01']=m0m1corrknots
            if self.options.host_component: guess[parlist=='modelerr_host']=1e-9 # something small...  #mhostvarknots
            

            # if SN param list is provided, initialize with these params
            if self.options.snparlist:
                snpar = Table.read(self.options.snparlist,format='ascii')
                snpar['SNID'] = snpar['SNID'].astype(str)

            from numpy.random import default_rng
            rng = default_rng(134912348)
            for sn in datadict.keys():
                if self.options.snparlist:
                    # hacky matching, but SN names are a mess as usual
                    iSN = ((sn == snpar['SNID']) | ('sn'+sn == snpar['SNID']) |
                           ('sn'+sn.lower() == snpar['SNID']) | (sn+'.0' == snpar['SNID']))
                    if len(snpar['SNID'][iSN]) > 1:
                        raise RuntimeError(f"found duplicate in parameter list for SN {snpar['SNID'][iSN][0]}")
                    if len(snpar[iSN]):
                        if self.options.host_component:
                            guess[parlist == f'xhost_{sn}'] = snpar['xhost'][iSN]
                        
                        for i in range((self.options.n_components)):
                            if ('x'+str(i)) in snpar.keys():
                                guess[parlist==f'x{i}_{sn}'] = snpar[ 'x'+str(i)][iSN]
                            else:
                                guess[parlist==f'x{i}_{sn}'] = rng.standard_normal()
                        if snpar['x0'][iSN]<= 0:
                            log.warning(f'Bad input value for {sn}: x0={ float(snpar["x0"][iSN])}')
                            guess[parlist==f'x0_{sn}'] = 10**(-0.4*(cosmo.distmod(datadict[sn].zHelio).value-19.36-10.635))

                        guess[parlist == 'c0_%s'%sn] = snpar['c'][iSN]
                        guess[parlist == 'c1_%s'%sn] = np.random.exponential(0.2)
                    else:
                        log.warning(f'SN {sn} not found in SN par list {self.options.snparlist}')
                        guess[parlist == 'x0_%s'%sn] = 10**(-0.4*(cosmo.distmod(datadict[sn].zHelio).value-19.36-10.635))

                elif 'SIM_SALT2x1' in datadict[sn].__dict__.keys():
                    # simulated samples need an initialization list also
                    # initializing to sim. values is not the best but running SNANA fits adds a lot of overhead
                    log.info(f'initializing parameters using simulated values for SN {sn}')
                    guess[parlist == 'x0_%s'%sn] = datadict[sn].SIM_SALT2x0
                    guess[parlist == 'x1_%s'%sn] = datadict[sn].SIM_SALT2x1
                    guess[parlist == 'c0_%s'%sn] = datadict[sn].SIM_SALT2c
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
                    
                    uncalledModel = specflux(specdata[k].tobs,specdata[k].wavelength,phase,wave,
                                             m0,m1,lambda wave : initcolorlaw( 1, guess[parlist=='cl0'],wave),zHel,x0,x1,c,mwebv=mwebv)
        
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

                    
        return guess,modelconfiguration

    def bootstrapSALTModel_batch(self,datadict,trainingresult,saltfitter):
        # runs bootstrapping in batch mode via calls to trainsalt
        USERNAME = os.environ['USER']

        sbatch_file = os.path.expandvars(self.options.bootstrap_sbatch_template)
        if not os.path.exists(sbatch_file):
            raise RuntimeError(f'sbatch file {sbatch_file} not found')

        # get initial job ids
        cmd = (f"squeue -u {USERNAME} -h -o '%i %j' ")
        ret = subprocess.run( [cmd], shell=True,
                              capture_output=True, text=True )
        pid_init = ret.stdout.split()

        # worried that jobs will fail, we need a workaround
        if not self.options.get_bootstrap_output_only:
            for i in range(self.options.n_bootstrap):
                if self.options.fast: faststr = '--fast'
                else: faststr = ''
                cmd = f"trainsalt -c {self.options.configfile} --outputdir {self.options.outputdir}/bootstrap_{i} --use_previous_errors True --error_dir {self.options.outputdir} --gaussnewton_maxiter {self.options.maxiter_bootstrap} --errors_from_bootstrap False {faststr} --validate_modelonly True --bootstrap_single"
                with open(sbatch_file) as fin, open(f"saltshaker_batch_{i}","w") as fout:
                    for line in fin:
                        line = line.replace('\n','')
                        print(line.replace('REPLACE_JOB',cmd).\
                              replace('REPLACE_MEM','30000').\
                              replace('REPLACE_NAME',f'saltshaker_bootstrap_{i}').\
                              replace('REPLACE_LOGFILE',f'saltshaker_bootstrap_log_{i}').\
                              replace('REPLACE_WALLTIME','04:00:00').\
                              replace('REPLACE_CPUS_PER_TASK','1'),file=fout)
                print(f"submitting batch job saltshaker_batch_{i}")
                os.system(f"sbatch saltshaker_batch_{i}")

        # now let's get the current job ids
        cmd = (f"squeue -u {USERNAME} -h -o '%i %j' ")
        ret = subprocess.run( [cmd], shell=True,
                              capture_output=True, text=True )
        pid_new = ret.stdout.split()
        
        # while the jobs are running, sit there patiently....
        tstart = time.time()
        print('waiting for jobs to finish')
        time.sleep(120)
        unfinished_jobs = True
        while unfinished_jobs:
            time.sleep(5)
            ret = subprocess.run( [cmd], shell=True,
                                  capture_output=True, text=True )
            pid_all = ret.stdout.split()
            unfinished_jobs = False
            for p in pid_all:
                if p not in pid_init and p in pid_new: unfinished_jobs = True
            # sometimes things just hang and there's nothing to be done
            # give it 12 hours and then give up (since Midway has long lags)
            if (time.time()-tstart)/60/60 > 12:
                print('warning : there were unfinished jobs!')
                unfinished_jobs = False

        # use all the output files to get the errors
        M0_bs,M1_bs,Mhost_bs = np.zeros([np.shape(trainingresult.M0)[0],np.shape(trainingresult.M0)[1],self.options.n_bootstrap]),\
            np.zeros([np.shape(trainingresult.M0)[0],np.shape(trainingresult.M0)[1],self.options.n_bootstrap]),\
            np.zeros([np.shape(trainingresult.M0)[0],np.shape(trainingresult.M0)[1],self.options.n_bootstrap])

        iGood = np.array([],dtype=int)
        for i in range(self.options.n_bootstrap):
            if not os.path.exists(f"{self.options.outputdir}/bootstrap_{i}/salt3_parameters.dat"):
                
                continue
            iGood = np.append(iGood,i)
            try:
                params,parvals = np.loadtxt(f"{self.options.outputdir}/bootstrap_{i}/salt3_parameters.dat",unpack=True,dtype=str,skiprows=1)
            except:
                import pdb; pdb.set_trace()
            parvals = parvals.astype(float)
            
            # save each result
            if not self.options.host_component:
                M0,M1 = saltfitter.SALTModel(parvals,evaluatePhase=saltfitter.phaseout,evaluateWave=saltfitter.waveout)
            else:
                M0,M1,Mhost = saltfitter.SALTModel(parvals,evaluatePhase=saltfitter.phaseout,evaluateWave=saltfitter.waveout)
            M0_bs[:,:,i] = M0 #parvals[params == 'm0'].reshape(np.shape(trainingresult.M0))
            M1_bs[:,:,i] = M1 #parvals[params == 'm1'].reshape(np.shape(trainingresult.M0))
            if self.options.host_component and len(parvals[params == 'mhost']):
                Mhost_bs[:,:,i] = Mhost #parvals[params == 'mhost'].reshape(np.shape(trainingresult.M0))


        trainingresult.M0bootstraperr = np.std(M0_bs[:,:,iGood],axis=2)
        trainingresult.M1bootstraperr = np.std(M1_bs[:,:,iGood],axis=2)
        trainingresult.Mhostbootstraperr = np.std(Mhost_bs[:,:,iGood],axis=2)
        trainingresult.cov_M0_M1_bootstrap = np.zeros(np.shape(trainingresult.M0))
        trainingresult.cov_M0_Mhost_bootstrap = np.zeros(np.shape(trainingresult.M0))
        for j in range(np.shape(M0_bs)[0]):
            for i in range(np.shape(M0_bs)[1]):
                trainingresult.cov_M0_M1_bootstrap[j,i] = np.sum((M0_bs[j,i]-np.mean(M0_bs[j,i,iGood]))*(M1_bs[j,i]-np.mean(M1_bs[j,i,iGood])))/(self.options.n_bootstrap-1)
                trainingresult.cov_M0_Mhost_bootstrap[j,i] = np.sum((M0_bs[j,i]-np.mean(M0_bs[j,i,iGood]))*(Mhost_bs[j,i]-np.mean(Mhost_bs[j,i,iGood])))/(self.options.n_bootstrap-1)

        return trainingresult

        
    def bootstrapSALTModel(self,datadict,trainingresult):

        # check for option inconsistency
        if self.options.use_previous_errors and not self.options.resume_from_outputdir and not self.options.error_dir:
            raise RuntimeError('resume_from_outputdir or error_dir must be specified to use use_previous_errors option')

        x_modelpars,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc = self.initialParameters(datadict)

        saltfitkwargs = self.get_saltkw(phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)
        n_phaseknots,n_waveknots = len(phaseknotloc)-4,len(waveknotloc)-4
        n_errphaseknots,n_errwaveknots = len(errphaseknotloc)-4,len(errwaveknotloc)-4

        # bootstrap modifications of datadict and parlist
        M0_bs,M1_bs,Mhost_bs = np.zeros([np.shape(trainingresult.M0)[0],np.shape(trainingresult.M0)[1],self.options.n_bootstrap]),\
            np.zeros([np.shape(trainingresult.M0)[0],np.shape(trainingresult.M0)[1],self.options.n_bootstrap]),\
            np.zeros([np.shape(trainingresult.M0)[0],np.shape(trainingresult.M0)[1],self.options.n_bootstrap])
        for j in range(self.options.n_bootstrap):
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
            
            fitter = fitting(self.options.n_components,self.options.n_colorpars,
                             n_phaseknots,n_waveknots,
                             datadict_bootstrap)
            log.info(f'bootstrap iteration {j+1}!')

            saltfitkwargs['regularize'] = self.options.regularize
            saltfitkwargs['fitting_sequence'] = self.options.fitting_sequence
            saltfitkwargs['fit_model_err'] = False
            saltfitter = saltfit.GaussNewton(x_modelpars_bs,datadict_bootstrap,parlist_bs,**saltfitkwargs)

            # suppress regularization
            saltfitter.neff[saltfitter.neff<saltfitter.neffMax]=10
            
            # do the fitting
            trainingresult_bs,message = fitter.gaussnewton(
                saltfitter,x_modelpars_bs,
                self.options.maxiter_bootstrap,getdatauncertainties=False)
            for k in datadict_bootstrap.keys():
                trainingresult_bs.SNParams[k]['t0'] =  datadict_bootstrap[k].tpk_guess

            log.info('message: %s'%message)
            log.info('Final loglike'); saltfitter.maxlikefit(trainingresult_bs.params_raw)
            #log.info('Final photometric loglike'); saltfitter.maxlikefit(trainingresult_bs.params_raw,dospec=False)
            
            log.info(trainingresult_bs.params.size)

            # save each result
            M0_bs[:,:,j] = trainingresult_bs.M0[:]
            M1_bs[:,:,j] = trainingresult_bs.M1[:]
            Mhost_bs[:,:,j] = trainingresult_bs.Mhost[:]


        trainingresult.M0bootstraperr = np.std(M0_bs,axis=2)
        trainingresult.M1bootstraperr = np.std(M1_bs,axis=2)
        trainingresult.Mhostbootstraperr = np.std(Mhost_bs,axis=2)
        trainingresult.cov_M0_M1_bootstrap = np.zeros(np.shape(trainingresult.M0))
        trainingresult.cov_M0_Mhost_bootstrap = np.zeros(np.shape(trainingresult.M0))
        for j in range(np.shape(M0_bs)[0]):
            for i in range(np.shape(M0_bs)[1]):
                trainingresult.cov_M0_M1_bootstrap[j,i] = np.sum((M0_bs[j,i]-np.mean(M0_bs[j,i,:]))*(M1_bs[j,i]-np.mean(M1_bs[j,i,:])))/(self.options.n_bootstrap-1)
                trainingresult.cov_M0_Mhost_bootstrap[j,i] = np.sum((M0_bs[j,i]-np.mean(M0_bs[j,i,:]))*(Mhost_bs[j,i]-np.mean(Mhost_bs[j,i,:])))/(self.options.n_bootstrap-1)

        return trainingresult
    
    
    def initializesaltmodelobject(self,datadict):
        x_modelpars,modelconfiguration = self.initialParameters(datadict)
        return x_modelpars,saltresids.SALTResids(datadict,self.kcordict,modelconfiguration,self.options)

    def fitSALTModel(self,datadict,x_modelpars,saltresids,returnGN=True):
        # check for option inconsistency
        if self.options.use_previous_errors and not self.options.resume_from_outputdir and not self.options.error_dir:
            raise RuntimeError('resume_from_outputdir or error_dir must be specified to use use_previous_errors option')
        
        
        if self.options.bootstrap_single:
            new_keys = np.random.choice(list(datadict.keys()),size=len(datadict.keys()),replace=True)

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

        optimizer=optimizers.getoptimizer(self.options.optimizer)
        
        log.info('training on %i SNe!'%len(datadict.keys()))
        for i in range(self.options.n_repeat):
            
            saltfitter = optimizer(x_modelpars,saltresids,self.options.outputdir,self.options)
            if returnGN:
            #This is an awful hack and should be removed
                return saltfitter,x_modelpars
            
            # do the fitting
            x_modelpars = saltfitter.optimize( x_modelpars)
            
        Xfinal= saltresids.constraints.enforcefinaldefinitions(x_modelpars,saltresids.SALTModel(x_modelpars))
        # hack!
        self.options.errors_from_hessianapprox = False
        if self.options.errors_from_hessianapprox: 
            sigma=saltresids.estimateparametererrorsfromhessian(Xfinal)
            np.save(path.join(self.options.outputdir,'parametercovariance.npy'), sigma)
        else: sigma=None
        trainingresult=saltresids.processoptimizedparametersforoutput(Xfinal,x_modelpars,sigma)
        for k in datadict.keys():
            trainingresult.snparams[k]['t0'] =  datadict[k].tpk_guess
        
        log.info('Final loglike'); log.info(saltresids.maxlikefit(trainingresult.params_raw))
        #log.info('Final photometric loglike'); log.info(saltresids.maxlikefit(trainingresult.params_raw,dospec=False))
        
        log.info(trainingresult.params.size)



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
            for name,par in zip(trainingresult.parlist,trainingresult.params):

                foutpars.write('{: <30} {:.15e}\n'.format(name,par))

        with  open('{}/salt3_parameters_unscaled.dat'.format(outdir),'w') as foutpars:
            foutpars.write('{: <30} {}\n'.format('Parameter Name','Value'))
            for name,par in zip(trainingresult.parlist,trainingresult.params_raw):

                foutpars.write('{: <30} {:.15e}\n'.format(name,par))

        np.save('{}/salt3_mcmcchain.npy'.format(outdir),chain)
        np.save('{}/salt3_loglikes.npy'.format(outdir),loglikes)
    
        #Handle nones when dataerrsurfaces is None by writing it with 0's
        if trainingresult.dataerrsurfaces is None:
            dataerrsurfaces=[ np.zeros((trainingresult.phase.size,trainingresult.wave.size) ) for j in range(len(trainingresult.componentnames))]
        else: 
            dataerrsurfaces= trainingresult.dataerrsurfaces
        
        if trainingresult.datacovsurfaces is None:
            datacovsurfaces=sum([ [(i,j, np.zeros((trainingresult.phase.size,trainingresult.wave.size) )) for j in range(i+1,len(trainingresult.componentnames))] for i in range(len(trainingresult.componentnames))],[])
        else:
            datacovsurfaces=trainingresult.datacovsurfaces
    
        #Loop through the components and write their output files
        for fluxmodel,errmodel,errdata,name in zip( trainingresult.componentsurfaces, trainingresult.modelerrsurfaces,dataerrsurfaces, trainingresult.componentnames):
            with open(f'{outdir}/salt3_template_{name[1:]}.dat','w') as foutmodel,\
             open(f'{outdir}/salt3_lc_model_variance_{name[1:]}.dat','w') as foutmodelerr,\
             open(f'{outdir}/salt3_lc_variance_{name[1:]}.dat','w') as foutdataerr:
                for i,p in enumerate(trainingresult.phase):
                    for j,w in enumerate(trainingresult.wave):
                        foutmodel.write(f'{p:.1f} {w:.2f} {fluxmodel[i,j]:8.15e}\n')
                        if not self.options.use_previous_errors:
                            foutmodelerr.write(f'{p:.1f} {w:.2f} {errmodel[i,j]**2.:8.15e}\n')
                            foutdataerr.write(f'{p:.1f} {w:.2f} {errmodel[i,j]**2.+errdata[i,j]**2.:8.15e}\n')
        
        #Copy previous variance files from relevant output
        if  self.options.use_previous_errors:
            if  self.options.resume_from_outputdir:
                errordir=self.options.resume_from_outputdir
            elif self.options.error_dir:
                errordir=self.options.error_dir
            else:
                log.critical('"use_previous_errors" enabled, but no directory given to pull errors from')
            for filename in os.listdir(errordir):
                if 'variance' in filename and 'salt3' in filename:
                    try: os.system(f"cp {errordir}/{filename} {outdir}/{filename}")
                    except: pass
        else:
            #Otherwise, loop through the model covariance surfaces and write them, storing them by index in a dictionary
            modelerrdict={}
            for firstind,secondind, covsurface in trainingresult.modelcovsurfaces:
                if firstind>secondind:
                    secondind,firstind=firstind,secondind
                modelerrdict[firstind,secondind]=covsurface
                with open(f'{outdir}/salt3_lc_model_covariance_{trainingresult.componentnames[firstind][1:]}{trainingresult.componentnames[secondind][1:]}.dat','w') as foutcov:
                    for i,p in enumerate(trainingresult.phase):
                        for j,w in enumerate(trainingresult.wave):
                            foutcov.write(f'{p:.1f} {w:.2f} {covsurface[i,j]:8.15e}\n')
        
            #Loop through and write the data covariance surfaces, including model error
            for firstind,secondind, datasurface in datacovsurfaces:
                if firstind>secondind:
                    secondind,firstind=firstind,secondind
                try: modelsurface=modelerrdict[firstind,secondind] 
                except KeyError:
                    modelsurface=np.zeros(datasurface.shape)
                with open(f'{outdir}/salt3_lc_covariance_{trainingresult.componentnames[firstind][1:]}{trainingresult.componentnames[secondind][1:]}.dat','w') as foutcov:
                    for i,p in enumerate(trainingresult.phase):
                        for j,w in enumerate(trainingresult.wave):
                            foutcov.write(f'{p:.1f} {w:.2f} {modelsurface[i,j]+datasurface[i,j]:8.15e}\n')
        
        #Write dispersion file, with everything set to 1
        with open(f'{outdir}/salt3_lc_dispersion_scaling.dat','w') as lcdispfile:
                for i,p in enumerate(trainingresult.phase):
                    for j,w in enumerate(trainingresult.wave):
                        lcdispfile.write(f'{p:.1f} {w:.2f} 1.00e+00\n')
        
        #Write the color dispersion, clipping at one
        cldispersionmax=1.
        with open(f'{outdir}/salt3_color_dispersion.dat','w') as foutclscat:
            for j,w in enumerate(trainingresult.wave):
                print(f'{w:.2f} {np.clip(trainingresult.clscat[j],0.,cldispersionmax):8.15e}',file=foutclscat)

        foutinfotext = f"""RESTLAMBDA_RANGE: {self.options.colorwaverange[0]} {self.options.colorwaverange[1]}
COLORLAW_VERSION: {self.options.colorlaw_function[0]}
COLORCOR_PARAMS: {self.options.colorwaverange[0]:.0f} {self.options.colorwaverange[1]:.0f}  {len(trainingresult.clpars[0])}  {' '.join(['%8.10e'%cl for cl in trainingresult.clpars[0]])}

COLOR_OFFSET:  0.0
COLOR_DISP_MAX: {cldispersionmax:.1f}  # avoid crazy sim-mags at high-z

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
        if len(trainingresult.clpars)==1: 
            colorlaw=trainingresult.clpars[0]
            with open(f'{outdir}/salt3_color_correction.dat','w') as foutcl:
                print(f'{len(colorlaw):.0f}',file=foutcl)
                
                for c in colorlaw:
                    print(f'{c:8.10e}',file=foutcl)
                print(f"""Salt2ExtinctionLaw.version 1
Salt2ExtinctionLaw.min_lambda {self.options.colorwaverange[0]:.0f}
Salt2ExtinctionLaw.max_lambda {self.options.colorwaverange[1]:.0f}""",file=foutcl)
        else:
            for i,name,colorlaw in (zip(range(trainingresult.clpars.shape[0]),self.options.colorlaw_function,trainingresult.clpars)):
                with open(f'{outdir}/salt3_color_correction_{i}.dat','w') as foutcl:
                    print(f'{len(colorlaw):.0f}',file=foutcl)
                    for c in colorlaw:
                        print(f'{c:8.10e}',file=foutcl)
                    print(f"""Salt2ExtinctionLaw.version 1
Salt2ExtinctionLaw.min_lambda {self.options.colorwaverange[0]:.0f}
Salt2ExtinctionLaw.max_lambda {self.options.colorwaverange[1]:.0f}""",file=foutcl)


        # best-fit and simulated SN params
        with open(f'{outdir}/salt3train_snparams.txt','w') as foutsn:
            print('# SN x0 x1 c t0 SIM_x0 SIM_x1 SIM_c SIM_t0 SALT2_x0 SALT2_x1 SALT2_c SALT2_t0',file=foutsn)
            #print('# SN x0 x1 c_i c_g t0 SIM_x0 SIM_x1 SIM_c SIM_t0 SALT2_x0 SALT2_x1 SALT2_c SALT2_t0',file=foutsn)
            for snlist in self.options.snlists.split(','):
                snlist = os.path.expandvars(snlist)
                if not os.path.exists(snlist):
                    log.warning(f'SN list file {snlist} does not exist. Checking {data_rootdir}/trainingdata/{snlist}')
                    snlist = f'{data_rootdir}/trainingdata/{snlist}'
                    if not os.path.exists(snlist):
                        raise RuntimeError(f'SN list file {snlist} does not exist')


                snfiles = np.genfromtxt(snlist,dtype='str')
                snfiles = np.atleast_1d(snfiles)

                for k in trainingresult.snparams.keys():
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

                    if 't0' not in trainingresult.snparams[k].keys():
                        trainingresult.snparams[k]['t0'] = 0.0

                    print(f"{k} {trainingresult.snparams[k]['x0']:8.10e} {trainingresult.snparams[k]['x1']:.10f} {trainingresult.snparams[k]['c'] if 'c' in trainingresult.snparams[k] else trainingresult.snparams[k]['c0']:.10f} {trainingresult.snparams[k]['t0']:.10f} {SIM_x0:8.10e} {SIM_x1:.10f} {SIM_c:.10f} {SIM_PEAKMJD:.2f} {salt2x0:8.10e} {salt2x1:.10f} {salt2c:.10f} {salt2t0:.10f}",file=foutsn)
                    #print(f"{k} {trainingresult.snparams[k]['x0']:8.10e} {trainingresult.snparams[k]['x1']:.10f} {trainingresult.snparams[k]['c'] if 'c' in trainingresult.snparams[k] else trainingresult.snparams[k]['c0']:.10f} {trainingresult.snparams[k]['c1']:.10f} {trainingresult.snparams[k]['t0']:.10f} {SIM_x0:8.10e} {SIM_x1:.10f} {SIM_c:.10f} {SIM_PEAKMJD:.2f} {salt2x0:8.10e} {salt2x1:.10f} {salt2c:.10f} {salt2t0:.10f}",file=foutsn)
                    
        keys=['num_lightcurves','num_spectra','num_sne']
        yamloutputdict={key.upper():getattr(trainingresult,key) for key in keys}
        yamloutputdict['CPU_MINUTES']=(time.time()-initializationtime)/60
        yamloutputdict['ABORT_IF_ZERO']=1
        with open(f'{self.options.yamloutputfile}','w') as file: yaml.dump(yamloutputdict,file)



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
                                  n_colorpars=self.options.n_colorpars,host_component=self.options.host_component,
                                  colorlaw_function=self.options.colorlaw_function)
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
            if self.options.n_colorpars[0] > 0:
                fitc = True

            if self.options.binspec:
                binspecres = self.options.binspecres
            else:
                binspecres = None

            datadict = readutils.rdAllData(snlist,self.options.estimate_tpk,
                                           dospec=self.options.dospec,
                                           peakmjdlist=self.options.tmaxlist,
                                           binspecres=binspecres,snparlist=self.options.snparlist,
                                           maxsn=self.options.maxsn,
                                           specrecallist=self.options.specrecallist)
                
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

                if not i % 15:
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
                if i % 15 == 10:
                    pdf_pages.savefig()
                    plt.close('all')
                else:
                    for ax in [ax1,ax2,ax3,ax4,ax5]:
                        ax.xaxis.set_ticklabels([])
                        ax.set_xlabel(None)
                i += 5
                count += 1
            log.info(f'plotted light curves for {count} SNe')
            log.info(f'total chi^2 is {salt2_chi2tot:.1f} for SALT2 and {salt3_chi2tot:.1f} for SALT3')
        if not i %15 ==0:
            pdf_pages.savefig()
        pdf_pages.close()
        log.info(f'plotting light curves took {time.time()-tlc:.1f}')
        
    def main(self,returnGN=False):
        
        try:
            stage='initialization'
            if not len(self.surveylist):
                raise RuntimeError('surveys are not defined - see documentation')
            tkstart = time.time()
            self.kcordict=readutils.rdkcor(self.surveylist,self.options)
            log.info(f'took {time.time()-tkstart:.3f} to read in kcor files')
            # TODO: ASCII filter files
                
            if not os.path.exists(self.options.outputdir):
                os.makedirs(self.options.outputdir)
            if self.options.binspec:
                binspecres = self.options.binspecres
            else:
                binspecres = None

            tdstart = time.time()
            datadict = readutils.rdAllData(self.options.snlists,self.options.estimate_tpk,
                                           dospec=self.options.dospec,
                                           peakmjdlist=self.options.tmaxlist,
                                           binspecres=binspecres,
                                           snparlist=self.options.snparlist,
                                           maxsn=self.options.maxsn,
                                           specrecallist=self.options.specrecallist)
            log.info(f'took {time.time()-tdstart:.3f} to read in data files')
            tcstart = time.time()
            for snid,sn in datadict.items():
                for filt in sn.filt:
                    if filt not in self.kcordict[sn.survey]:
                        if filt not in self.options.__dict__[f"{sn.survey.split('(')[0]}_ignore_filters"].replace(' ','').split(','): 
                            raise ValueError(f'Kcor file missing key {filt} from survey {sn.survey} for sn {snid}; valid keys are {", ".join([x for x in self.kcordict[sn.survey] if "lambdaeff" in self.kcordict[sn.survey][x]])}')
            datadict = self.mkcuts(datadict)[0]
            log.info(f'took {time.time()-tcstart:.3f} to apply cuts')
            
            phasebins=np.linspace(*self.options.phaserange,int((self.options.phaserange[1]-self.options.phaserange[0])/self.options.phasesplineres)+1,True)
            wavebins=np.linspace(*self.options.waverange,int((self.options.waverange[1]-self.options.waverange[0])/self.options.wavesplineres)+1,True)
            datadensity.datadensityplot(path.join(self.options.outputdir,'datadensity.pdf') ,phasebins,wavebins,datadict,self.kcordict,1500,200)
            # fit the model - initial pass
            if self.options.stage == "all" or self.options.stage == "train":
                # read the data
                stage='training'
                x_modelpars,saltresids=self.initializesaltmodelobject(datadict)
                if not returnGN:
                    trainingresult,chain,loglikes,saltfitter = self.fitSALTModel(datadict,x_modelpars,saltresids,returnGN=returnGN)
                else:
                    fitter,saltfitter,modelpars = self.fitSALTModel(datadict,x_modelpars,saltresids,returnGN=returnGN)
                    return fitter,saltfitter,modelpars

                if self.options.errors_from_bootstrap:
                    if self.options.bootstrap_batch_mode:
                        trainingresult = self.bootstrapSALTModel_batch(datadict,trainingresult,x_modelpars,saltresids)
                    else:
                        trainingresult = self.bootstrapSALTModel(datadict,trainingresult,x_modelpars,saltresids)
                
                stage='output'
                # write the output model - M0, M1, c
                self.wrtoutput(self.options.outputdir,trainingresult,chain,loglikes,datadict)
            log.info('successful SALT3 training!  Output files written to %s'%self.options.outputdir)
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

class RunTraining:

    def __init__(self):

        self.usagestring = """SALT3 Training

usage: python TrainSALT.py -c <configfile> <options>

config file options can be overwridden at the command line"""

        
    def get_config_options(self,salt,configfile,args=None):
        
        if configfile:
            pass
        else:
            raise RuntimeError('Configuration file must be specified')

        config = configparser.ConfigParser(inline_comment_prefixes='#')
        if not os.path.exists(configfile):
            raise RuntimeError(f'Configfile {configfile} doesn\'t exist!')

        config.read(configfile)

        user_parser = salt.add_user_options(usage=self.usagestring,config=config)
        user_options = user_parser.parse_known_args(args)[0]

        loggerconfig.dictconfigfromYAML(user_options.loggingconfig,user_options.outputdir)

        if not os.path.exists(user_options.modelconfig):
            print('warning : model config file %s doesn\'t exist.  Trying package directory'%user_options.modelconfig)
            user_options.modelconfig = '%s/%s'%(config_rootdir,user_options.modelconfig)
        if not os.path.exists(user_options.modelconfig):
            raise RuntimeError('can\'t find model config file!  Checked %s'%user_options.modelconfig)
                    
        if not os.path.exists(user_options.trainingconfig):
            print('warning : trainingconfig config file %s doesn\'t exist.  Trying package directory'%user_options.trainingconfig)
            user_options.trainingconfig = '%s/%s'%(config_rootdir,user_options.trainingconfig)
        if not os.path.exists(user_options.trainingconfig):
            raise RuntimeError('can\'t find trainingconfig config file!  Checked %s'%user_options.trainingconfig)

        optimizer=optimizers.getoptimizer(user_options.optimizer)
        modelconfig = configparser.ConfigParser(inline_comment_prefixes='#')
        modelconfig.read(user_options.modelconfig)
        model_parser = saltresids.SALTResids.add_model_options(
            parser=user_parser,config=modelconfig)
        model_parser.addhelp()
        model_options = model_parser.parse_known_args(args)

        trainingconfig = configparser.ConfigParser(inline_comment_prefixes='#')
        trainingconfig.read(user_options.trainingconfig)
        training_parser = optimizer.add_training_options(
            parser=user_parser,config=trainingconfig)
        training_parser.addhelp()
        training_options = training_parser.parse_args(args)
        
        salt.options = training_options
        salt.options.host_component= True if salt.options.host_component else False
        if training_options.fast:
            if user_options.optimizer=='gaussnewton':
                if salt.options.gaussnewton_maxiter >= 1:
                    salt.options.gaussnewton_maxiter = 1
                salt.options.fit_model_err = False
                salt.options.fit_cdisp_only = False
                salt.options.validate_modelonly = True
            salt.options.maxsn = 10

        if salt.options.stage not in ['all','validate','train']:
            raise RuntimeError('stage must be one of all, validate, train')
        with open(path.join(salt.options.outputdir,'options.json'),'w') as optfile:
            import json
            json.dump(vars(salt.options),optfile)

        return

    def get_example_data(self):
        
        if os.path.exists('saltshaker-latest-training'):
            raise RuntimeError("saltshaker-latest-training exists in local directory.  Please remove it before continuing")
        download_dir(_example_data_url,os.getcwd())
    
    def main(self,configfile=None,args=None):
        
        salt = TrainSALT()
        
        if configfile is None:
            assert(args is None)
            parser = argparse.ArgumentParser(usage=self.usagestring, conflict_handler="resolve",add_help=False)
            parser.add_argument('configpositional',nargs='?',default=None,type=str,help='configuration file')
            parser.add_argument('-c','--configfile', default=None, type=str,
                                help='configuration file')
            parser.add_argument('-g','--get-example-data', default=False, action="store_true",
                                help='download the example data')

            options, _ = parser.parse_known_args()
            configfile,configpos=options.configfile,options.configpositional


        if options.get_example_data:
            self.get_example_data()
            print('example data has been downloaded to the saltshaker-latest-training directory')
        else:
            if configfile is None and configpos is not None: 
                configfile=configpos
            elif configfile is None and configpos is None:
                raise RuntimeError('Configuration file must be specified at command line')

            
            self.get_config_options(salt,configfile,args)
        
            salt.main()

