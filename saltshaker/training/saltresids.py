from saltshaker.util.inpynb import in_ipynb

from saltshaker.util.synphot import synphot
from saltshaker.util import batching
from saltshaker.util.jaxoptions import jaxoptions,sparsejac,wrapjvpmultipleargs
from saltshaker.training import init_hsiao
from saltshaker.training.datamodels import SALTfitcacheSN,modeledtraininglightcurve,modeledtrainingspectrum
from saltshaker.training import colorlaw

from saltshaker.training.priors import SALTPriors,__priors__,__nongaussianpriors__
from saltshaker.training.constraints import SALTconstraints

from saltshaker.config.configparsing import *

from sncosmo.models import StretchSource
from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
from sncosmo.utils import integration_grid
from sncosmo.salt2utils import SALT2ColorLaw

import scipy.stats as ss
from scipy.optimize import minimize, least_squares
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from scipy.special import factorial
from scipy.interpolate import splprep,splev,bisplev,bisplrep,interp1d,interp2d,RegularGridInterpolator,RectBivariateSpline
from scipy.integrate import trapezoid as trapz
from scipy import linalg
from scipy import sparse as scisparse

import glob
import numpy as np
from numpy.random import standard_normal
from numpy.linalg import slogdet

from astropy.cosmology import Planck15 as cosmo
from multiprocessing import Pool, get_context
from inspect import signature
from functools import partial,reduce
from itertools import starmap
if in_ipynb:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import matplotlib as mpl
mpl.use('agg')
import pylab as plt

import jax
from jax import numpy as jnp

from jax.scipy import linalg as jaxlinalg
from jax.experimental import sparse
from jax import lax

from collections import namedtuple
from argparse import SUPPRESS
from itertools import combinations_with_replacement

from os import path
import pickle

import time
import sys
import extinction
import copy
import warnings
import logging
import random

log=logging.getLogger(__name__)

from scipy import sparse as scisparse

_B_LAMBDA_EFF = np.array([4302.57])  # B-band-ish wavelength
_V_LAMBDA_EFF = np.array([5428.55])  # V-band-ish wavelength
warnings.simplefilter('ignore',category=FutureWarning)
interactive= sys.stdout.isatty()

saltconfiguration=namedtuple('saltconfiguration',

    ['parlist','phaseknotloc','waveknotloc','errphaseknotloc','errwaveknotloc'] )

salttrainingresult= namedtuple('salttrainingresult',
            ['num_lightcurves' , 'num_spectra' ,'num_sne' ,
            'parlist' , 'params' , 'params_raw' ,'phase' ,'wave' , 'componentnames', 
            
            'componentsurfaces' ,  'modelerrsurfaces', 'dataerrsurfaces' ,
            'modelcovsurfaces','datacovsurfaces','clpars','clscat','snparams'])        
            
def ensurepositivedefinite(matrix,maxiter=5):

    for i in range(maxiter):
        try: mineigenval=np.linalg.eigvalsh(matrix)[0]
        except np.linalg.LinAlgError:
            mineigenval= -np.abs(np.diag(matrix).min() * 1e-4)
        if mineigenval>0:
            return matrix
        else:
            if maxiter==0: 
                raise ValueError('Unable to make matrix positive semidefinite')
        matrix+=np.diag(-mineigenval*4* np.ones(matrix.shape[0]))
    return matrix

def getgaussianfilterdesignmatrix(shape,smoothing):
    windowsize=10+shape%2
    window=gaussian_filter1d(1.*(np.arange(windowsize)==windowsize//2),smoothing)
    while ~(np.any(window==0)):
        windowsize=2*windowsize+shape%2
        window=gaussian_filter1d(1.*(np.arange(windowsize)==windowsize//2),smoothing)
    window=window[window>0]

    diagonals=[]
    offsets=list(range(-(window.size//2),window.size//2+1))
    for i,offset in enumerate(offsets):
        diagonals+=[np.tile(window[i],shape-np.abs(offset))]
    design=scisparse.diags(diagonals,offsets).tocsr()
    for i in range(window.size//2+1):
        design[i,:window.size//2+1]=gaussian_filter1d(1.*(np.arange(design.shape[0])== i ),5)[:window.size//2+1]
        design[-i-1,-(window.size//2+1) : ]=gaussian_filter1d(1.*(np.arange(design.shape[0])== i ),5)[:window.size//2+1][::-1]    
    return design

    
class SALTResids:

    parameters = ['x0','x1','xhost','c','m0','m1','mhost','spcrcl','spcrcl_norm','spcrcl_poly',
                           'modelerr','modelcorr','clscat','clscat_0','clscat_poly']
    configoptionnames = set()
    
    def __init__(self,datadict,kcordict,saltconfiguration,options):
        inittime=time.time()
        self.nstep = 0
        self.datadict = datadict
        self.kcordict= kcordict
        
        self.nsn = len(self.datadict.keys())
        
        for key,value in saltconfiguration._asdict().items():
            self.__dict__[key] =value
            
        for key,value  in  options.__dict__.items(): 
            self.__dict__[key] = options.__dict__[key]
            
        self.npar = len(self.parlist)
        assert(type(self.parlist) == np.ndarray)

        self.usePriors = []
        self.priorWidths = []
        self.boundedParams = []
        self.bounds = []
        
        for opt in self.__dict__.keys():
            if opt.startswith('prior_'):
                self.usePriors += [opt[len('prior_'):]]
                self.priorWidths += [self.__dict__[opt]]
            elif opt.startswith('bound_'):
                self.boundedParams += [opt[len('bound_'):]]
                self.bounds += [tuple([float(x) for x in self.__dict__[opt]])]

        specrecalparams =self.parlist[(np.array([x.startswith('specrecal') for x in self.parlist]))]
        numrecalparams=[(specrecalparams==x ).sum() for x in np.unique(specrecalparams)]
        try:
            assert(all([numrecalparams[0]==x for x in numrecalparams]))
        except:
            raise NotImplementedError('Varying number of spectral recalibration parameters unimplemented in jax-compiled code')
        # pre-set some indices
        self.set_param_indices()
        
        # set some phase/wavelength arrays
        self.phase = np.linspace(self.phaserange[0],self.phaserange[1],
                                 int((self.phaserange[1]-self.phaserange[0])/self.phaseinterpres)+1,True)
        self.phaseout = np.linspace(self.phaserange[0],self.phaserange[1],
                                 int((self.phaserange[1]-self.phaserange[0])/self.phaseoutres)+1,True)
        self.interpMethod='nearest'
        
        self.wave = np.linspace(self.waverange[0],self.waverange[1],
                                int((self.waverange[1]-self.waverange[0])/self.waveinterpres)+1,True)
        self.waveout = np.linspace(self.waverange[0],self.waverange[1],
                                int((self.waverange[1]-self.waverange[0])/self.waveoutres)+1,True)
                
        self.neff=0


        self.m0guess = -19.49 #10**(-0.4*(-19.49-27.5))
        self.extrapolateDecline=0.015
        # set up the filters
        self.stdmag = {}
        self.fluxfactor = {}
        
        #Check that we've got suitable filters!
        for snid,sn in self.datadict.items():
            if sn.survey not in self.kcordict:
                raise ValueError(f'Could not find corresponding kcor for survey {sn.survey} for SN {snid} ')

        for survey in self.kcordict:
            if survey == 'default': 
                self.stdmag[survey] = {}
                self.bbandoverlap = (
                    self.wave>=self.kcordict['default']['Bwave'].min())&\
                    (self.wave<=self.kcordict['default']['Bwave'].max())
                self.bbandpbspl = np.interp(
                    self.wave[self.bbandoverlap],self.kcordict['default']['Bwave'],
                    self.kcordict['default']['Btp'])
                self.bbandpbspl *= self.wave[self.bbandoverlap]
                self.bbandpbspl /= np.trapz(self.bbandpbspl,self.wave[self.bbandoverlap])*HC_ERG_AA
                self.stdmag[survey]['B']=synphot(
                    self.kcordict[survey]['primarywave'],self.kcordict[survey]['AB'],
                    filtwave=self.kcordict['default']['Bwave'],filttp=self.kcordict[survey]['Btp'],
                    zpoff=0)
                self.stdmag[survey]['V']=synphot(
                    self.kcordict[survey]['primarywave'],self.kcordict[survey]['AB'],
                    filtwave=self.kcordict['default']['Vwave'],filttp=self.kcordict[survey]['Vtp'],
                    zpoff=0)
                self.fluxfactor['default']={}
                self.fluxfactor[survey]['B'] = 10**(0.4*(self.stdmag[survey]['B']+27.5))
                self.fluxfactor[survey]['V'] = 10**(0.4*(self.stdmag[survey]['V']+27.5))
                continue

            self.stdmag[survey] = {}
            self.fluxfactor[survey] = {}
            primarywave = self.kcordict[survey]['primarywave']
            for flt in self.kcordict[survey].keys():
                if flt == 'filtwave' or flt == 'primarywave' or \
                   flt == 'snflux' or flt == 'AB' or \
                   flt == 'BD17' or flt == 'Vega': continue
                if self.kcordict[survey][flt]['magsys'] == 'AB': primarykey = 'AB'
                elif self.kcordict[survey][flt]['magsys'].upper() == 'VEGA': primarykey = 'Vega'
                elif self.kcordict[survey][flt]['magsys'] == 'BD17': primarykey = 'BD17'
                self.stdmag[survey][flt] = synphot(
                    primarywave,self.kcordict[survey][primarykey],
                    filtwave=self.kcordict[survey][flt]['filtwave'],
                    filttp=self.kcordict[survey][flt]['filttrans'],
                    zpoff=0) - self.kcordict[survey][flt]['primarymag']
                self.fluxfactor[survey][flt] = 10**(0.4*(self.stdmag[survey][flt]+27.5))
                self.kcordict[survey][flt]['minlam'] = \
                    np.min(self.kcordict[survey][flt]['filtwave'][self.kcordict[survey][flt]['filttrans'] > 0.01])
                self.kcordict[survey][flt]['maxlam'] = \
                    np.max(self.kcordict[survey][flt]['filtwave'][self.kcordict[survey][flt]['filttrans'] > 0.01])
        # rest-frame B
        filttrans = self.kcordict['default']['Btp']
        filtwave = self.kcordict['default']['Bwave']
            
        pbspl = np.interp(self.wave,filtwave,filttrans,left=0,right=0)
        
        pbspl *= self.wave
        denom = np.trapz(pbspl,self.wave)
        pbspl /= denom*HC_ERG_AA
        self.kcordict['default']['Bpbspl'] = pbspl
        self.kcordict['default']['dwave'] = self.wave[1] - self.wave[0]
        
        #rest-frame V
        filttrans = self.kcordict['default']['Vtp']
        filtwave = self.kcordict['default']['Vwave']
            
        pbspl = np.interp(self.wave,filtwave,filttrans,left=0,right=0)
        
        pbspl *= self.wave
        denom = np.trapz(pbspl,self.wave)
        pbspl /= denom*HC_ERG_AA
        self.kcordict['default']['Vpbspl'] = pbspl
        
                
        #Count number of photometric and spectroscopic points
        self.num_spec=sum([datadict[sn].num_specobs for sn in datadict])
        self.num_spectra=sum([datadict[sn].num_spec for sn in datadict])
        self.num_lc=sum([datadict[sn].num_lc for sn in datadict])
        self.num_phot=sum([datadict[sn].num_photobs for sn in datadict])

        starttime=time.time()
    
        #Store the lower and upper edges of the phase/wavelength basis functions
        self.phaseBins=self.phaseknotloc[:-(self.bsorder+1)],self.phaseknotloc[(self.bsorder+1):]
        self.waveBins=self.waveknotloc[:-(self.bsorder+1)],self.waveknotloc[(self.bsorder+1):]
    
        #Find the iqr of the phase/wavelength basis functions
        self.phaseRegularizationBins=np.linspace(self.phase[0],self.phase[-1],self.phaseBins[0].size*2+1,True)
        self.waveRegularizationBins=np.linspace(self.wave[0],self.wave[-1],self.waveBins[0].size*2+1,True)

    
        self.phaseRegularizationPoints=(self.phaseRegularizationBins[1:]+self.phaseRegularizationBins[:-1])/2
        self.waveRegularizationPoints=(self.waveRegularizationBins[1:]+self.waveRegularizationBins[:-1])/2


        basisfunctions=[bisplev(
                self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i*(self.waveBins[0].size),self.bsorder,self.bsorder)) for i in range(self.phaseBins[0].size) ]
        self.phaseBinCenters=np.array(

            [(self.phase[:,np.newaxis]* x).sum()/x.sum() for x in basisfunctions ])
        basisfunctions=[bisplev(
                self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder)) for i in range(self.waveBins[0].size) ]

        self.waveBinCenters=np.array(
            [(self.wave[np.newaxis,:]*  x ).sum()/x.sum() for x in basisfunctions])

        
        
        #Find the basis functions evaluated at the centers of the basis functions for use in the regularization derivatives
        regularizationDerivs=[np.zeros((self.phaseRegularizationPoints.size*self.waveRegularizationPoints.size,self.im0.size)) for i in range(4)]
        for i in range(len(self.im0)):
            for j,derivs in enumerate([(0,0),(1,0),(0,1),(1,1)]):
                if self.bsorder == 0: continue
                regularizationDerivs[j][:,i]=bisplev(
                    self.phaseRegularizationPoints,self.waveRegularizationPoints,
                    (self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder),
                    dx=derivs[0],dy=derivs[1]).flatten()
        regularizationDerivs=map(sparse.BCOO.fromdense,regularizationDerivs)
        self.componentderiv,self.dcompdphasederiv,self.dcompdwavederiv,self.ddcompdwavedphase =regularizationDerivs

        #Color law initialization
        if self.preintegrate_photometric_passband:
            self.wavebasis=self.waveBinCenters
        else:
            self.wavebasis=self.wave 
        
        try:
            assert(len(self.n_colorpars)==len(self.colorlaw_function))
            self.ncl=len(self.colorlaw_function)
            
        except:
            raise ValueError(f'Inconsistent number of color law specifications given {self.colorlaw_function} {self.n_colorpars}' )
        self.colorlawfunction=[colorlaw.getcolorlaw(function)( npars,
        self.colorwaverange) for function,npars in zip(self.colorlaw_function,self.n_colorpars)]
        

        self.guessScale=np.ones(self.n_components)
        
        self.relativeregularizationweights=jnp.array([1]+[self.variantregularization]*(self.n_components-1)+( [self.mhostregularization] if self.host_component else []))
        
        if self.regularize:
            self.updateEffectivePoints()


        
        def getphotdatacounts(datadict):
            for snid,sn in datadict.items():
                for flt,lc in sn.photdata.items():
                    yield len(lc)
                    
        def getspecdatacounts(datadict):
            for snid,sn in datadict.items():
                for flt,lc in sn.specdata.items():
                    yield len(lc)

        photdatasizes=list(getphotdatacounts(datadict))
        photpadding,efficiency=batching.optimizepaddingsizes(self.photometric_zeropadding_batches,photdatasizes)
        
        log.info(f'Separating photometric data into {len(photpadding)} batches, at a space efficiency of {efficiency:.0%}')
        efficiencywarningthreshold= .3
        if efficiency<efficiencywarningthreshold:
            log.warning(f'Efficiency less than {efficiencywarningthreshold:.0%}, consider increasing batching of data')
        specdatasizes=list(getspecdatacounts(datadict))
        if len(specdatasizes):
            specpadding,efficiency=batching.optimizepaddingsizes(self.spectroscopic_zeropadding_batches,specdatasizes)
            log.info(f'Separating spectroscopic data into {len(specpadding)} batches, at a space efficiency of {efficiency:.0%}')
            if efficiency<efficiencywarningthreshold:
                log.warning(f'Efficiency less than {efficiencywarningthreshold:.0%}, consider increasing batching of data')
        else:
            specpadding = 0; efficiency = 1
            log.info(f'no spectroscopic data, no batches needed')

        log.info('Calculating cached quantities')
        start=time.time()
        
        
        if bool(self.trainingcachefile) and path.exists(self.trainingcachefile):
            log.info(f'Loading precomputed quantities from file {self.trainingcachefile}')
            with open(self.trainingcachefile,'rb') as file: cachedfile=  pickle.load(file)
            
            self.datadict = cachedfile['datadict']
            self.batchedphotdata= cachedfile['batchedphotdata']
            self.batchedspecdata= cachedfile['batchedspecdata']
            self.allphotdata = sum([[x.photdata[lc] for lc in x.photdata ]for x in self.datadict.values() ],[])
            self.allspecdata = sum([[x.specdata[key] for key in x.specdata ]for x in self.datadict.values() ],[])

        else:
            
            log.info('Precomputing filters')
            iterable=list(self.datadict.items())
            #Shuffle it so that tqdm's estimates are hopefully more accurate
            random.shuffle(iterable)
            if sys.stdout.isatty() or in_ipynb:
                iterable=tqdm(iterable,smoothing=.1)
            self.datadict={snid: sn if isinstance(sn,SALTfitcacheSN) else SALTfitcacheSN(sn,self,self.kcordict,photpadding,specpadding)  for snid,sn in iterable}
            log.info('Batching data')
            self.allphotdata = sum([[x.photdata[lc] for lc in x.photdata ]for x in self.datadict.values() ],[])
            self.allspecdata = sum([[x.specdata[key] for key in x.specdata ]for x in self.datadict.values() ],[])

            #self.batchedphotdata= batching.batchdatabysize(self.allphotdata)
            #self.batchedspecdata= batching.batchdatabysize(self.allspecdata)
            batching.batchdatabysize(self.allphotdata,self.outputdir,prefix='phot')
            self.batchedphotdata = []
            for cachefile in glob.glob(f'{self.outputdir}/caching_phot_*.pkl'):
                with open(cachefile,'rb') as fin: 
                    self.batchedphotdata += [pickle.load(fin)['data']]
            batching.batchdatabysize(self.allspecdata,self.outputdir,prefix='spec')
            self.batchedspecdata = []
            for cachefile in glob.glob(f'{self.outputdir}/caching_spec_*.pkl'):
                with open(cachefile,'rb') as fin: 
                    self.batchedspecdata += [pickle.load(fin)['data']]
            
            if self.trainingcachefile:
                log.info(f'Writing precomputed quantities to file {self.trainingcachefile}')
                with open( self.trainingcachefile,'wb') as file:
                    pickle.dump({'datadict':self.datadict,
                                'batchedphotdata':self.batchedphotdata,
                                'batchedspecdata':self.batchedspecdata},file)
            
        log.info('Constructing batched methods')

        self.batchedphotresiduals=batching.batchedmodelfunctions(lambda *args,**kwargs: modeledtraininglightcurve.modelresidual(*args,**kwargs)['residuals'],
                                  self.batchedphotdata, modeledtraininglightcurve,
                                  flatten=True)
        
        self.batchedphotlikelihood=batching.batchedmodelfunctions(lambda *args,**kwargs: modeledtraininglightcurve.modelloglikelihood(*args,**kwargs),
                                  self.batchedphotdata, modeledtraininglightcurve,
                                  sum=True)

        self.batchedphotvariances=jax.jit(batching.batchedmodelfunctions(  modeledtraininglightcurve.modelfluxvariance,
                                  self.batchedphotdata, modeledtraininglightcurve,
                                  ))
                                  
        self.batchedphotfluxes=jax.jit(batching.batchedmodelfunctions(  modeledtraininglightcurve.modelflux,
                                  self.batchedphotdata, modeledtraininglightcurve,
                                  ))

        self.batchedspecresiduals=batching.batchedmodelfunctions(lambda *args,**kwargs: modeledtrainingspectrum.modelresidual(*args,**kwargs)['residuals'],
                                  self.batchedspecdata, modeledtrainingspectrum,
                                  flatten=True)

        self.batchedspeclikelihood=batching.batchedmodelfunctions(lambda *args,**kwargs: modeledtrainingspectrum.modelloglikelihood(*args,**kwargs),
                          self.batchedspecdata, modeledtrainingspectrum,
                          sum=True)
        
        self.batchedspecvariances=jax.jit(batching.batchedmodelfunctions(  modeledtrainingspectrum.modelfluxvariance,
                                  self.batchedspecdata, modeledtrainingspectrum,
                                  ))
                                  
        self.batchedspecfluxes=jax.jit(batching.batchedmodelfunctions(  modeledtrainingspectrum.modelflux,
                                  self.batchedspecdata, modeledtrainingspectrum,
                                  ))
                                  
        self.priors = SALTPriors(self)
        self.constraints=SALTconstraints(self)
        
        log.info('Time required to calculate cached quantities {:.1f}s'.format(time.time()-start))
                
    @classmethod
    def add_model_options(cls,parser,config,addargsonly=False):
        if parser == None:
                parser = ConfigWithCommandLineOverrideParser(usage='', conflict_handler="resolve")
        temp=generateerrortolerantaddmethod(parser)
        def wrapaddingargument(*args,**kwargs):
            if 'clargformat' in kwargs:
                name=kwargs['clargformat'].format(key=args[2])
            else:
                name=args[2]
            cls.configoptionnames.add(name)
            if addargsonly:
                return True
            else: return temp(*args,**kwargs)
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
        successful=successful&wrapaddingargument(config,'trainingparams','variantregularization','m1regularization',     type=float,
                                                help='Scales regularization weighting of varying components (all other than M0) relative to M0 weighting (>1 increases smoothing of M1)  (default=%(default)s)')
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
        successful=successful&wrapaddingargument(config,'modelparams','errbsorder','errinterporder', type=int,
                                                help='for model uncertainty splines/polynomial funcs, order of the function (default=%(default)s)')
        successful=successful&wrapaddingargument(config,'modelparams','bsorder','interporder',     type=int,
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
        successful=successful&wrapaddingargument(config,'modelparams','n_colorpars',    nargs='+',      type=int,
                                                help='number of degrees of the phase-independent color law polynomial (default=%(default)s)')

        successful=successful&wrapaddingargument(config,'modelparams','n_errorsurfaces',          type=int, default=2,
                                                help='number of surfaces used for the error model (default=%(default)s)')

        successful=successful&wrapaddingargument(config,'modelparams','colorlaw_function',   nargs='+',      type=str,
                                                help='color law function, see colorlaw.py (default=%(default)s)')               

        successful=successful&wrapaddingargument(config,'modelparams','n_colorscatpars',         type=int,
                                                help='number of parameters in the broadband scatter model (default=%(default)s)')
        successful=successful&wrapaddingargument(config,'modelparams','error_snake_phase_binsize',      type=float,
                                                help='number of days over which to compute scaling of error model (default=%(default)s)')
        successful=successful&wrapaddingargument(config,'modelparams','error_snake_wave_binsize',  type=float,
                                                help='number of angstroms over which to compute scaling of error model (default=%(default)s)')
        successful=successful&wrapaddingargument(config,'modelparams','use_snpca_knots',         type=boolean_string,
                                                help='if set, define model on SNPCA knots (default=%(default)s)')               

        successful=successful&wrapaddingargument(config,'modelparams','constraint_names','constraints',  default='', nargs='*',      type=str,
                                                help='constraints enforced on the model, see constraints.py (default=%(default)s)')               

        successful=successful&wrapaddingargument(config,'modelparams','secondary_constraint_names','secondary_constraints',  default='', nargs='*',      type=str,
                                                help='constraints enforced on the model after an initial burn-in, see constraints.py (default=%(default)s)')


        for prior in __priors__:
                successful=successful&wrapaddingargument(config,'priors',prior ,type=float,clargformat="--prior_{key}",
                                                        help=f"prior on {prior}",default=SUPPRESS)
        for prior in __nongaussianpriors__:
                successful=successful&wrapaddingargument(config,'priors',prior ,type=float,clargformat="--prior_{key}",
                                                        help=f"prior on {prior}",default=SUPPRESS)

        # bounds
        for param in cls.parameters:
                successful=successful&wrapaddingargument(config,'bounds', param, type=float,nargs=3,clargformat="--bound_{key}",
                                                        help="bound on %s"%param,default=SUPPRESS)

        if not successful: sys.exit(1)
        return parser

    
    def set_param_indices(self):
    

        self.m0min = np.min(np.where(self.parlist == 'm0')[0])
        self.m0max = np.max(np.where(self.parlist == 'm0')[0])

        self.errmin = [np.min(np.where(self.parlist == f'modelerr_{i}')[0]) for i in range(self.n_errorsurfaces)]
        self.errmax = [np.max(np.where(self.parlist == f'modelerr_{i}')[0]) for i in range(self.n_errorsurfaces)]
        if self.host_component:
            self.errmin += [np.min(np.where(self.parlist == 'modelerr_host')[0])]
            self.errmax += [np.max(np.where(self.parlist == 'modelerr_host')[0])]
        self.errmin = tuple(self.errmin)
        self.errmax = tuple(self.errmax)

        self.corrcombinations=[x for x in  combinations_with_replacement(np.arange(self.n_errorsurfaces),2) if x[0]!=x[1]]
        if self.host_component:
            self.corrcombinations+=[(0,'host')]#=sum([[(i,j) for j in range(i+1,self.n_components)]for i in range(self.n_components)] ,[])

        self.corrmin = tuple([np.min(np.where(self.parlist == 'modelcorr_{}{}'.format(i,j))[0]) for i,j in self.corrcombinations]) 
        self.corrmax = tuple([np.max(np.where(self.parlist == 'modelcorr_{}{}'.format(i,j))[0]) for i,j in self.corrcombinations]) 
        
        self.im0 = np.where(self.parlist == 'm0')[0]
        self.im1 = np.where(self.parlist == 'm1')[0]
        self.imhost = np.where(self.parlist == 'mhost')[0]
        
        wavemin = []
        for i in range(self.im0.size):
            wavemin += [self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),
                                          i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]][0]]
        self.im0new = np.where(self.parlist == 'm0')[0][np.array(wavemin) > 8500]
        self.im1new = np.where(self.parlist == 'm1')[0][np.array(wavemin) > 8500]
        
        self.ix0 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x0') or si.startswith('specx0')],dtype=int)
        self.ix1 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x1')],dtype=int)
        self.ixhost = np.array([i for i, si in enumerate(self.parlist) if si.startswith('xhost')],dtype=int)
        
        self.ic  = np.array([np.where(np.char.startswith(self.parlist,f'c{i}_'  ))[0] for i in range(len(self.n_colorpars))])
        self.ic0  = np.where(np.char.startswith(self.parlist,'c0_'))[0]
        self.iCL = np.array([np.where(self.parlist == f'cl{i}')[0] for i in range(len(self.n_colorpars))])

        # Galactic color law has "fake" free parameters so we need to ignore it
        # during error estimation
        iCLNG = np.where(np.array(self.colorlaw_function) != 'colorlaw_galactic')[0]
        self.iCLNoGal = np.where(self.parlist == f'cl{iCLNG}')[0]

        self.ispcrcl_norm = np.array([i for i, si in enumerate(self.parlist) if si.startswith('specx0')],dtype=int)
        if self.ispcrcl_norm.size==0: self.ispcrcl_norm=np.zeros(self.npar,dtype=bool)
        self.ispcrcl = np.array([i for i, si in enumerate(self.parlist) if si.startswith('spec')],dtype=int) # used to be specrecal
        self.ispcrcl_coeffs = np.array([i for i, si in enumerate(self.parlist) if si.startswith('specrecal')],dtype=int) # used to be specrecal
        if self.ispcrcl.size==0: self.ispcrcl=np.zeros(self.npar,dtype=bool)
        self.imodelerr = np.array([i for i, si in enumerate(self.parlist) if si.startswith('modelerr')],dtype=int)
        self.imodelerr0 = np.array([i for i, si in enumerate(self.parlist) if si ==('modelerr_0')],dtype=int)
        self.imodelerr1 = np.array([i for i, si in enumerate(self.parlist) if si==('modelerr_1')],dtype=int)
        self.imodelerrhost = np.array([i for i, si in enumerate(self.parlist) if si==('modelerr_host')],dtype=int)
        self.imodelcorr = np.array([i for i, si in enumerate(self.parlist) if si.startswith('modelcorr')],dtype=int)
        self.imodelcorr01 = np.array([i for i, si in enumerate(self.parlist) if si==('modelcorr_01')],dtype=int)
        self.imodelcorr0host = np.array([i for i, si in enumerate(self.parlist) if si==('modelcorr_0host')],dtype=int)
        self.iclscat = np.where(self.parlist=='clscat')[0]

        self.iModelParam=np.ones(self.npar,dtype=bool)
        self.iModelParam[self.imodelerr]=False
        self.iModelParam[self.imodelcorr]=False
        self.iModelParam[self.iclscat]=False
        self.icoordinates = np.array([np.where(np.char.startswith(self.parlist,f'x{i}') )[0]
                    for i in range(1,self.n_components)]+list(self.ixhost)
        )
        self.icomponents =[np.where(self.parlist == f'm{i}')[0] for i in range(self.n_components)]
        if self.host_component:
            self.icomponents+=[self.imhost]
        self.icomponents = np.array(self.icomponents)
        self.iclscat_0,self.iclscat_poly = np.array([],dtype='int'),np.array([],dtype='int')
        if len(self.ispcrcl):
            for i,parname in enumerate(np.unique(self.parlist[self.iclscat])):
                self.iclscat_0 = np.append(self.iclscat_0,np.where(self.parlist == parname)[0][-1])
                self.iclscat_poly = np.append(self.iclscat_poly,np.where(self.parlist == parname)[0][:-1])        


    @partial(jaxoptions, static_argnums=[0,3,4,5,6],static_argnames= ['dopriors','dospecresids','usesns','suppressregularization'],diff_argnum=1,jitdefault=True) 
    def lsqwrap(self,guess,uncertainties,dopriors=True,dospecresids=True,usesns=None,suppressregularization=False):
        """
        """
        residuals = []
        if not (usesns is  None): raise NotImplementedError('Have not implemented a restricted set of sne')
        lcuncertainties,specuncertainties=uncertainties
        residuals+=[ self.batchedphotresiduals(guess,lcuncertainties,fixuncertainties=True) ]

        
        if dospecresids:
            residuals+=[ self.batchedspecresiduals(guess,specuncertainties,fixuncertainties=True) ]

        if dopriors:
            residuals+=[self.priors.priorresids(guess)]
            if self.regularize:
                if suppressregularization:
                    residuals+=[
                        func(guess,self.suppressedneff) \
                                for func in [self.dyadicRegularization,self.phaseGradientRegularization,self.waveGradientRegularization]]
                else:
                    residuals+=[func(guess,self.neff) \
                                     for func in [self.dyadicRegularization,self.phaseGradientRegularization,self.waveGradientRegularization]]

        return  jnp.concatenate(residuals)
    
    def lsqwrap_sources(self,guess,uncertainties,dopriors=True,dospecresids=True,usesns=None)   :
        numresids= lambda func: jax.eval_shape(func,guess).shape[0]
        numresids_reg= lambda func: jax.eval_shape(func,guess,self.neff).shape[0]
        
        sources=[]
        sources+=['phot']* numresids(lambda x: self.batchedphotresiduals(x,uncertainties[0],fixuncertainties=True) )
        if dospecresids: sources+=[f'spec']* numresids(lambda x: self.batchedspecresiduals(x,uncertainties[1],fixuncertainties=True) )
       
        if dopriors: 
            sources+=[f'priors']*numresids(self.priors.priorresids)
            for label,func in [('reg_dyad',self.dyadicRegularization), ('reg_phase',self.phaseGradientRegularization), ('reg_wave',self.waveGradientRegularization)] :
                sources+=[label]*numresids_reg(func)
        return sources

      

      
      
    def getChi2Contributions(self,X,**kwargs):
        uncertainties= self.calculatecachedvals(X,target='variances')
        residuals= self.lsqwrap(X,uncertainties,**kwargs)
        sourceskwargs={key:val for key,val in kwargs.items() if not (key=='jit')}
        sources=self.lsqwrap_sources(X,uncertainties,**sourceskwargs)
        sources=np.array([x.split('_')[0] for x in  sources])
        assert(np.isin(sources,['reg','phot','spec','priors']).all())
        def loop():
            for name,abbrev in [('Photometric', 'phot'),('Spectroscopic','spec'),('Prior','priors'),('Regularization','reg')]:
                x=residuals[sources==abbrev]
                #Number of data points shouldn't include the padding
                if abbrev=='phot': ndof=self.num_phot
                elif abbrev=='spec': ndof=self.num_spec
                else:
                    ndof=x.size
                yield (name,(x**2).sum(),ndof)

        return list(loop())
            

    @partial(jaxoptions, static_argnums=[0,3,4,5,6 ,7],static_argnames= ['fixfluxes','fixuncertainties','dopriors','dospec','usesns','usesecondary'],diff_argnum=1,jitdefault=True) 
    def constrainedmaxlikefit(self,params,*args,usesecondary=True,**kwargs):
        return self.maxlikefit(self.constraints.transformtoconstrainedparams(params,usesecondary),*args,**kwargs)


    @partial(jaxoptions, static_argnums=[0,3,4,5,6 ,7],static_argnames= ['fixfluxes','fixuncertainties','dopriors','dospec','usesns'],diff_argnum=1,jitdefault=True) 
    def maxlikefit(
            self,guess,cachedresults=None,fixuncertainties=False,fixfluxes=False,dopriors=True,dospec=True,usesns=None):
        """
        Calculates the likelihood of given SALT model to photometric and spectroscopic data given during initialization
        
        Parameters
        ----------
        x : array
            SALT model parameters
                    
        Returns
        -------
        
        chi2: float
            Goodness of fit of model to training data   
        """
                
        if cachedresults is None: cachedresults=None,None
        if not (usesns is  None): raise NotImplementedError('Have not implemented a restricted set of sne')

        loglike=(self.batchedphotlikelihood (guess,cachedresults[0],fixuncertainties,fixfluxes))

        
        if dospec:
            
            loglike+= (
             self.batchedspeclikelihood(guess,cachedresults[1],fixuncertainties,fixfluxes)
             )

        if dopriors:
            loglike+=self.priors.priorloglike(guess)
                
            if self.regularize:
                loglike+=sum([-(func(guess,self.neff)**2).sum()/2. for func in [self.dyadicRegularization,self.phaseGradientRegularization,self.waveGradientRegularization]] )
        return loglike


    def calculatecachedvals(self,x,target=None):
        if target == 'fluxes' : return [self.batchedphotfluxes(x),self.batchedspecfluxes(x)]
        if target=='variances': return [self.batchedphotvariances(x),self.batchedspecvariances(x)]
        
    def bestfitsinglebandnormalizationsforSN(self,x,sn):
        sndata=self.datadict[sn]
        for flt in sndata.photdata:
            lcdata=sndata.photdata[flt]
            designmatrix=lcdata.modelflux(x)
            vals=lcdata.fluxcal
            variance=lcdata.fluxcalerr**2 +lcdata.modelfluxvariance(x)
            
            normvariance=1/(designmatrix**2/variance).sum()
            normalization=(designmatrix*vals/variance).sum()*normvariance
            results[flt]=normalization,normvariance
        return results


    def estimateparametererrorsfromhessian(self,X,hessian=None):
        """Approximate Hessian by jacobian times own transpose to determine uncertainties in flux surfaces"""
        log.info("determining M0/M1 errors by approximated Hessian")

        if hessian is None:
            varyingParams=reduce(lambda x,y:  x | np.isin(np.arange(self.npar), y),[
                self.icomponents,self.iCLNoGal,self.imhost],False)

            logging.debug('Allowing parameters {np.unique(self.parlist[varyingParams])} in calculation of inverse Hessian')
            X=jnp.array(X)
            lsqwrap_partial=lambda x: self.lsqwrap(X.at[varyingParams].set(x),self.calculatecachedvals(X,target='variances'),suppressregularization=True,dospecresids=self.dospec)
            jvp= wrapjvpmultipleargs(lsqwrap_partial ,[0])
            jac = sparsejac(lsqwrap_partial,jvp,[0], True )(X[varyingParams])
    #         np.save(path.join(self.outputdir,'jac.npy'), jac)

            hessian=(jac.T @ jac ).toarray()

            maxval=np.max( np.abs(jnp.nan_to_num(hessian,nan=0,posinf=0,neginf=0)) )
            hessian=jnp.nan_to_num(hessian,nan=0,posinf=maxval,neginf=-maxval)

            hessian = ensurepositivedefinite(hessian)
            #Simple preconditioning of the jacobian before attempting to invert
        
        scales=np.sqrt(np.diag(hessian) )
        scales[scales==0]=1
        preconditioningelementwise=np.outer( scales,scales)

        sigma= linalg.cho_solve( linalg.cho_factor(hessian/preconditioningelementwise), np.identity(scales.size))/preconditioningelementwise
        
        sigmafull=np.zeros((self.npar,self.npar))
        sigmafull[np.outer(varyingParams,varyingParams)]=sigma.flatten()
        return sigmafull

    def computeuncertaintiesfromparametererrors(self,X,sigma,smoothingfactor=150):

        varyingParams=reduce(lambda x,y:  x | np.isin(np.arange(self.npar), y),[
            self.icomponents,self.iCLNoGal,self.imhost],False)

        # DJ: check for bad errors
        iBad = np.where(np.diag(sigma)[varyingParams] == 0)[0]
        if len(iBad):
            raise RuntimeError('model parameters had zero uncertainty!')
        iOK = np.where(np.diag(sigma)[~varyingParams] == 0)[0]
        if len(iOK):
            ### this shouldn't matter, just setting to random number
            ### so cholesky doesn't complain
            sigma[np.where(~varyingParams)[0][iOK],np.where(~varyingParams)[0][iOK]] = 100
        
        #Turning spline_derivs into a sparse matrix for speed
        logging.info('Computing flux surface uncertainties')
        chunkindex,chunksize=0,10
        assert(isinstance(self.host_component,bool))
        corrcombinations=[x for x in  combinations_with_replacement(np.arange(self.n_components+self.host_component),2) ]
        surfaces={x:np.empty((self.phaseout.size,self.waveout.size)) for x in corrcombinations }

        iterable=np.arange(self.waveout.size)[::chunksize]
        if sys.stdout.isatty() or in_ipynb:
            iterable=tqdm(iterable,smoothing=.1)

        for chunkindex in iterable:

            spline_derivs = np.empty([self.phaseout.size, min(self.waveout.size-chunkindex, chunksize),self.im0.size])
            for i in range(self.im0.size):
                if self.bsorder == 0: continue
                spline_derivs[:,:,i]=bisplev(self.phaseout,self.waveout[chunkindex:chunkindex+chunksize],(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder))
            spline2d=scisparse.csr_matrix(spline_derivs.reshape(-1,self.im0.size))

            #Smooth things a bit, since this is supposed to be for broadband photometry
            if smoothingfactor>0:
                smoothingmatrix=getgaussianfilterdesignmatrix(spline2d.shape[0],smoothingfactor/self.waveoutres)
                spline2d=smoothingmatrix*spline2d
            mask=np.zeros((self.phaseout.size,self.waveout.size),dtype=bool)
            mask[:,chunkindex:chunkindex+chunksize]=True        

            for i,j in corrcombinations:
                #.A1 method just converts the output "matrix" type to an ndarray
                surfaces[(i,j)][mask]=spline2d.multiply( 
                        spline2d @ sigma[self.icomponents[i],:][:,self.icomponents[j]] 
                    ).sum(axis= 1).A1


        components=self.SALTModel(X)

        stderrs=[np.clip(np.sqrt(surfaces[(i,i)]),0, np.abs(comp).max()*2) for i,comp in zip(range(self.n_components+self.host_component),(components))]
        correlations=[ ( i,j,
            np.clip(np.nan_to_num(surfaces[(i,j)]/(stderrs[i]*stderrs[j])),-1,1)*(stderrs[i]*stderrs[j]))
                    for i,j in corrcombinations if i!=j]

        return stderrs,correlations

    
    def processoptimizedparametersforoutput(self,rescaledparams,rawparams, parametercovariance=None):
        X=np.array(rawparams)
        Xfinal= np.array(rescaledparams)
        errmodel=self.ErrModel(Xfinal,evaluatePhase=self.phaseout,evaluateWave=self.waveout)
        componentnames=  ['M'+str(i) for i in range(self.n_components)]+(['Mhost'] if self.host_component else [])
        
        if parametercovariance is None:
            dataerrs,datacovs=None,None
        else : 
            dataerrs,datacovs=self.computeuncertaintiesfromparametererrors( X,parametercovariance)
        
        covmodel=[]
        corrmodel=self.CorrelationModel(Xfinal,evaluatePhase=self.phaseout,evaluateWave=self.waveout)
        for i,combination in enumerate(self.corrcombinations):
            
            covmodel+= [(*combination,corrmodel[i] * errmodel[combination[0]] * errmodel[-1 if combination[1]=='host' else combination[1]])]
    
        return salttrainingresult( num_lightcurves=self.num_lc,num_spectra=self.num_spectra,num_sne=len(self.datadict), parlist=self.parlist,
            params=Xfinal,params_raw=rawparams,phase= self.phaseout ,wave=self.waveout, 
            componentnames=componentnames, 
            componentsurfaces= self.SALTModel(Xfinal,evaluatePhase=self.phaseout,evaluateWave=self.waveout),
            modelerrsurfaces=errmodel,
            modelcovsurfaces = covmodel,
            dataerrsurfaces =dataerrs,
            datacovsurfaces =datacovs,
             clpars= Xfinal[self.iCL] , clscat=self.colorscatter(Xfinal,self.waveout),
            snparams= {sn: {
                              (name if not ((name=='c0') and (len(self.n_colorpars)==1)) else 'c' 
                              
                              ):X[self.parlist == '_'.join([name,sn])][0] for name in (
                                [f'x{i}' for i in range(self.n_components)]+
                                ([f'c{i}' for i in range(len(self.n_colorpars))]
                                )
                              )}                              
                              for sn in self.datadict})

    def SALTModel(self,x,evaluatePhase=None,evaluateWave=None):
        """Returns flux surfaces of SALT model"""
        components=[]
        for i in range( self.n_components  + self.host_component):
            comppars=x[self.icomponents[i]]
            
            if self.bsorder != 0:
                surface = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
                             self.wave if evaluateWave is None else evaluateWave,
                             (self.phaseknotloc,self.waveknotloc,comppars,self.bsorder,self.bsorder))
            else:
                phase = self.phase if evaluatePhase is None else evaluatePhase
                wave = self.wave if evaluateWave is None else evaluateWave
                n_repeat_phase = int(phase.size/(self.phaseknotloc.size-1))+1
                n_repeat_phase_extra = -1*(n_repeat_phase*(self.phaseknotloc.size-1) % phase.size)
                if n_repeat_phase_extra == 0: n_repeat_phase_extra = None
                n_repeat_wave = int(wave.size/(self.waveknotloc.size-1))+1
                n_repeat_wave_extra = -1*(n_repeat_wave*(self.waveknotloc.size-1) % wave.size)
                if n_repeat_wave_extra == 0: n_repeat_wave_extra = None
                surface = np.repeat(np.repeat(comppars.reshape([self.phaseknotloc.size-1,self.waveknotloc.size-1]),n_repeat_phase,axis=0),
                               n_repeat_wave,axis=1)[:n_repeat_phase_extra,:n_repeat_wave_extra]
            components+=[surface]
        
        return np.array(components)

    def SALTModelDeriv(self,x,dx,dy,evaluatePhase=None,evaluateWave=None):
        """Returns derivatives of flux surfaces of SALT model"""
        m0pars = x[self.m0min:self.m0max+1]

        m0 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
                 self.wave if evaluateWave is None else evaluateWave,
                 (self.phaseknotloc,self.waveknotloc,m0pars,self.bsorder,self.bsorder),
                 dx=dx,dy=dy)
            
        if self.n_components >= 2 and self.n_components <= 3:
            m1pars = x[self.im1]
            m1 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
                         self.wave if evaluateWave is None else evaluateWave,
                         (self.phaseknotloc,self.waveknotloc,m1pars,self.bsorder,self.bsorder),
                         dx=dx,dy=dy)
            if self.n_components == 2 and not self.host_component:
                components = (m0,m1)
            elif self.host_component:
                mhostpars = x[self.imhost]
                mhost = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
                             self.wave if evaluateWave is None else evaluateWave,
                             (self.phaseknotloc,self.waveknotloc,mhostpars,self.bsorder,self.bsorder),
                             dx=dx,dy=dy)

                components = (m0,m1,mhost)
        elif self.n_components == 1:
            components = (m0,)
        else:
            raise RuntimeError('A maximum of three principal components is allowed')
            
        return components

    def CorrelationModel(self,x,evaluatePhase=None,evaluateWave=None):
        """Returns correlation between SALT model components as a function of phase and wavelength"""
        components=[]
        phase=self.phase if evaluatePhase is None else evaluatePhase
        wave=self.wave if evaluateWave is None else evaluateWave
        for min,max in zip(self.corrmin,self.corrmax):
            errpars = x[min:max+1]
            if self.errbsorder == 0:
                binphasecenter=((self.errphaseknotloc)[1:]+(self.errphaseknotloc)[:-1])/2
                binwavecenter =((self.errwaveknotloc)[1:]+(self.errwaveknotloc)[:-1])/2
                interp=RegularGridInterpolator((binphasecenter,binwavecenter),errpars.reshape(binphasecenter.size,binwavecenter.size),'nearest',False,0)
                gridwave,gridphase=np.meshgrid(wave,phase)
                clipinterp=lambda x,y: interp((np.clip(x,binphasecenter.min(),binphasecenter.max()),np.clip(y,binwavecenter.min(),binwavecenter.max())))
                result=clipinterp(gridphase.flatten(),gridwave.flatten()).reshape((phase.size,wave.size))
                components+=[result]
            else:
                components+=[  bisplev(phase,
                                   wave,
                                   (self.errphaseknotloc,self.errwaveknotloc,errpars,self.errbsorder,self.errbsorder))]
        return components
    
    def colorscatter(self,x,wave):
        clscatpars = x[self.parlist == 'clscat']
        pow=clscatpars.size-1-np.arange(clscatpars.size)
        coeffs=clscatpars/factorial(pow)
        clscat=jnp.exp(jnp.polyval(coeffs,(wave-5500)/1000))
        return clscat
        
    
    def ErrModel(self,x,evaluatePhase=None,evaluateWave=None):
        """Returns modeled variance of SALT model components as a function of phase and wavelength"""
        x=np.array(x)
        phase=self.phase if evaluatePhase is None else evaluatePhase
        wave=self.wave if evaluateWave is None else evaluateWave
        components=[]
        for min,max in zip(self.errmin,self.errmax):
            errpars = x[min:max+1]
            if self.errbsorder == 0:
                binphasecenter=((self.errphaseknotloc)[1:]+(self.errphaseknotloc)[:-1])/2
                binwavecenter =((self.errwaveknotloc)[1:]+(self.errwaveknotloc)[:-1])/2
                
                interp=RegularGridInterpolator((binphasecenter,binwavecenter),errpars.reshape(binphasecenter.size,binwavecenter.size),'nearest',False,0)
                clipinterp=lambda x,y: interp((np.clip(x,binphasecenter.min(),binphasecenter.max()),np.clip(y,binwavecenter.min(),binwavecenter.max())))
                gridwave,gridphase=np.meshgrid(wave,phase)
                result=clipinterp(gridphase.flatten(),gridwave.flatten()).reshape((phase.size,wave.size))
                components+=[result]
            else:
            
                components+=[  bisplev(phase,
                               wave,
                               (self.errphaseknotloc,self.errwaveknotloc,errpars,self.errbsorder,self.errbsorder))]
        if self.n_components > self.n_errorsurfaces :
            components+=(self.n_components-self.n_errorsurfaces)*[np.zeros((phase.size,wave.size))]
        return components


            
    def updateEffectivePoints(self):
        """
        Updates the "effective number of points" constraining a given bin in 
        phase/wavelength space.
        
        Parameters
        ----------
            
        x : array
            SALT model parameters
                    
        """
        #Clean out array
        self.neffRaw=np.zeros((self.phaseRegularizationPoints.size,self.waveRegularizationPoints.size))

        for snid,sn in self.datadict.items():
            for k,spectrum in sn.specdata.items():
                # weight by ~mag err?
                err=spectrum.fluxerr/spectrum.flux
                snr=spectrum.flux/spectrum.fluxerr
                restWave=spectrum.wavelength/(1+sn.zHelio)
                phase=spectrum.phase
                
                if phase<self.phaseRegularizationBins[0]:
                    phaseIndex=0
                elif phase>self.phaseRegularizationBins[-1]:
                    phaseIndex=-1
                else:
                    phaseIndex= np.where( (phase>=self.phaseRegularizationBins[:-1]) & (phase<self.phaseRegularizationBins[1:]))[0][0]
                #Weight each spectroscopic point's contribution relative to total flux

                self.neffRaw[phaseIndex,:]+=ss.binned_statistic(
                    restWave,spectrum.flux/spectrum.flux.max()/len(spectrum),
                    bins=self.waveRegularizationBins,statistic='sum').statistic
                

        self.neffRaw=gaussian_filter1d(self.neffRaw,self.phasesmoothingneff,0)
        self.neffRaw=gaussian_filter1d(self.neffRaw,self.wavesmoothingneff,1)

        self.neff=self.neffRaw.copy()
        self.neff[self.neff>self.neffmax]=np.inf
        
        if not np.any(np.isinf(self.neff)): log.warning('Regularization is being applied to the entire phase/wavelength space: consider lowering neffmax (currently {:.2e})'.format(self.neffmax))
        
        self.neff=np.clip(self.neff,self.nefffloor,None).flatten()
        self.suppressedneff = np.select([self.neff>=self.neffmax,self.neff<self.neffmax],[np.tile(10,self.neff.shape),np.tile(10,self.neff.shape)]) #[self.neff,np.tile(10,self.neff.shape)])
        
    def plotEffectivePoints(self,phases=None,output=None):
        import matplotlib.pyplot as plt
        if phases is None:
            plt.imshow(self.neffRaw,cmap='Greys',aspect='auto')
            xticks=np.linspace(0,self.waveRegularizationPoints.size,8,False)
            plt.xticks(xticks,['{:.0f}'.format(self.waveRegularizationPoints[int(x)]) for x in xticks])
            plt.xlabel('$\lambda$ / Angstrom')
            yticks=np.linspace(0,self.phaseRegularizationPoints.size,8,False)
            plt.yticks(yticks,['{:.0f}'.format(self.phaseRegularizationPoints[int(x)]) for x in yticks])
            plt.ylabel('Phase / days')
        else:
            inds=np.searchsorted(self.phaseRegularizationPoints,phases)

            for i in inds:
                plt.plot(self.waveRegularizationPoints[:],self.neffRaw[i,:],label='{:.1f} days'.format(self.phaseRegularizationPoints[i]))
            plt.ylabel('$N_eff$')
            plt.xlabel('$\lambda (\AA)$')
            plt.xlim(self.waveRegularizationPoints.min(),self.waveRegularizationPoints.max())
            plt.legend()

        if output is None:
            plt.show()
        else:
            plt.savefig(output,dpi=288)
        plt.clf()
        
    
    def dyadicRegularization(self,x, neff):
        coeffs=x[self.icomponents]

        fluxes= self.componentderiv @ coeffs.T
        dfluxdwave=self.dcompdwavederiv @ coeffs.T
        dfluxdphase=self.dcompdphasederiv @ coeffs.T
        d2fluxdphasedwave=self.ddcompdwavedphase @ coeffs.T

        #Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
        normalization=jnp.sqrt( self.regulardyad*self.relativeregularizationweights/( (self.waveBins[0].size-1) *(self.phaseBins[0].size-1))**2.)
        #0 if model is locally separable in phase and wavelength i.e. flux=g(phase)* h(wavelength) for arbitrary functions g and h
        numerator=(dfluxdphase *dfluxdwave -d2fluxdphasedwave *fluxes )
        return (normalization[np.newaxis,:]* (numerator / (  neff[:,np.newaxis] ))).flatten()  
  
    def phaseGradientRegularization(self, x, neff):
        coeffs=x[self.icomponents]

        dfluxdphase=self.dcompdphasederiv @ coeffs.T
        normalization=jnp.sqrt(self.regulargradientphase *self.relativeregularizationweights /( (self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))
        return (normalization[np.newaxis,:]* ( dfluxdphase / neff[:,np.newaxis]  )).flatten()


    def waveGradientRegularization(self, x, neff):
        coeffs=x[self.icomponents]

        dfluxdwave=self.dcompdwavederiv @ coeffs.T
        normalization=jnp.sqrt(self.regulargradientwave *self.relativeregularizationweights/( (self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))

        return  (normalization[np.newaxis,:]* ( dfluxdwave / neff[:,np.newaxis] )).flatten()

