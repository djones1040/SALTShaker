from saltshaker.util.synphot import synphot
from saltshaker.training import init_hsiao
from saltshaker.training.datamodels import SALTfitcachelightcurve,  SALTfitcachespectrum,SALTfitcacheSN

from saltshaker.training.priors import SALTPriors

from sncosmo.models import StretchSource
from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
from sncosmo.utils import integration_grid

import scipy.stats as ss
from scipy.optimize import minimize, least_squares
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from scipy.special import factorial
from scipy.interpolate import splprep,splev,bisplev,bisplrep,interp1d,interp2d,RegularGridInterpolator,RectBivariateSpline
from scipy.integrate import trapz
from scipy import linalg

import numpy as np
from numpy.random import standard_normal
from numpy.linalg import slogdet

from astropy.cosmology import Planck15 as cosmo
from multiprocessing import Pool, get_context
from inspect import signature
from functools import partial
from itertools import starmap
from tqdm import tqdm

import matplotlib as mpl
mpl.use('agg')
import pylab as plt

import jax
from jax import numpy as jnp
from jaxlib.xla_extension import DeviceArray
from jax.scipy import linalg as jaxlinalg
from jax.experimental import sparse
from jax import lax

from collections import namedtuple

import time
import sys
import extinction
import copy
import warnings
import logging
log=logging.getLogger(__name__)

_B_LAMBDA_EFF = np.array([4302.57])  # B-band-ish wavelength
_V_LAMBDA_EFF = np.array([5428.55])  # V-band-ish wavelength
warnings.simplefilter('ignore',category=FutureWarning)

 
def rankOneCholesky(variance,beta,v):
    """Given variances, a scalar, and a vector, returns the cholesky matrix describing the covariance formed by the sum of the diagonal variance and the self outer product of the vector multiplied by the scalar"""
    b=1
    Lprime=np.zeros((variance.size,variance.size))
    if beta==0: return np.diag(np.sqrt(variance))
    for j in range(v.size):
        Lprime[j,j]=np.sqrt(variance[j]+beta/b*v[j]**2)
        gamma=(b*variance[j]+beta*v[j]**2)
        Lprime[j+1:,j]=Lprime[j,j]*beta*v[j+1:]*v[j]/gamma
        b+=beta*v[j]**2/variance[j]
    return Lprime

def jaxrankOneCholesky(variance,beta,v):
    Lprime=jnp.zeros((variance.size,variance.size))
    b=1
    for j in range(v.size):
        Lprime=Lprime.at[j,j].set(jnp.sqrt(variance[j]+beta/b*v[j]**2))
        gamma=(b*variance[j]+beta*v[j]**2)
        Lprime=Lprime.at[j+1:,j].set(Lprime[j,j]*beta*v[j+1:]*v[j]/gamma)
        b=b+beta*v[j]**2/variance[j]
    return Lprime

def toidentifier(input):
    return "x"+str(abs(hash(input)))

def evaljacobianforsomeindices(function,x,varyingparameters):
    if varyingparameters.all():
        return jax.jacfwd(function)(jnp.array(x))
    elif (~varyingparameters).all():
        return np.zeros(( function(x).size,x.size))
    else:
        #Map indices from the original vector to the contracted vector composed of only the varying parameters
        varyingparamsintindices=np.where(varyingparameters)[0]
        varyingparamsintindices={idx:i for i,idx in enumerate(varyingparamsintindices)}

        def varysomeparams(xcontracted):
            #Take only the varying parameters from the contracted vector, rest from the original
            xnew=jnp.array([(xcontracted[varyingparamsintindices[i]] if i in varyingparamsintindices else x[i]) for i in range(x.size) ])
            return function(xnew)
        #Read out the jacobian
        jacresult=jax.jacfwd(varysomeparams)(jnp.array(x[varyingparameters]))
        jacobianmatrix=np.zeros(( jacresult.shape[0],x.size))
        jacobianmatrix[:,varyingparameters]=jacresult

        return jacobianmatrix

    
class SALTResids:
    def __init__(self,guess,datadict,parlist,**kwargs):

        self.options=kwargs
        assert type(parlist) == np.ndarray
        self.nstep = 0
        self.parlist = parlist
        self.npar = len(parlist)
        self.datadict = datadict
        
        self.bsorder=3
        self.initparams = guess
        self.nsn = len(self.datadict.keys())
        
        for key, value in kwargs.items(): 
            self.__dict__[key] = value

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

        # initialize the model
        self.components = self.SALTModel(guess)
        self.salterr = self.ErrModel(guess)

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

        self.fixedUncertainties={}
        starttime=time.time()
        #Store derivatives of a spline with fixed knot locations with respect to each knot value
    
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
        phase=self.phaseRegularizationPoints
        wave=self.waveRegularizationPoints
        fluxes=self.SALTModel(guess,evaluatePhase=self.phaseRegularizationPoints,evaluateWave=self.waveRegularizationPoints)

        self.guessScale=[1.0 for f in fluxes]
        
        
        cachekeys=[]
        for sn in self.datadict:
            sndata=self.datadict[sn]
            for flt in sndata.photdata:
                cachekeys+=[f'phot_fluxes_{sn}_{flt}', f'phot_variances_{sn}_{flt}']
            for k in sndata.specdata:
                cachekeys+=  [f'spec_fluxes_{sn}_{k}',f'spec_variances_{sn}_{k}']
        saltcachedresultsunhash=namedtuple('saltcachedresults',[toidentifier(x) for x in cachekeys])

        class saltcachedresults(saltcachedresultsunhash):
    
            def __hash__(self):
                #Hack, figure out a better way to do this (probably the memoize python module?)
                hashval=0
                for field in self._fields:
                    entry=getattr(self,field)
                    if isinstance(entry,tuple): hashval=hashval+sum([hash(float((x**2 ).sum())) for x in entry])
                    else: hashval=hashval+hash(float((entry**2 ).sum()))
                    
                return hash(hashval)
        self.saltcachedresults=saltcachedresults

        if self.regularize:
            self.updateEffectivePoints(guess)

        
    
        
        self.datadictnocache=datadict
        log.info('Calculating cached quantities for speed in fitting loop')
        start=time.time()
        iterable=self.datadict.items()
        if sys.stdout.isatty():
            iterable=tqdm(iterable)
        self.datadict={snid: SALTfitcacheSN(sn,self,self.kcordict) for snid,sn in iterable}
        self.priors = SALTPriors(self)
        log.info('Time required to calculate cached quantities {:.1f}s'.format(time.time()-start))
            
    def set_param_indices(self):

        self.parameters = ['x0','x1','xhost','c','m0','m1','mhost','spcrcl','spcrcl_norm','spcrcl_poly',
                           'modelerr','modelcorr','clscat','clscat_0','clscat_poly']
        if not self.host_component:
            self.corrcombinations=sum([[(i,j) for j in range(i+1,self.n_components)]for i in range(self.n_components)] ,[])
        else:
            self.corrcombinations=[(0,1),(0,'host')]
        self.m0min = np.min(np.where(self.parlist == 'm0')[0])
        self.m0max = np.max(np.where(self.parlist == 'm0')[0])

        self.errmin = [np.min(np.where(self.parlist == f'modelerr_{i}')[0]) for i in range(self.n_components)]
        self.errmax = [np.max(np.where(self.parlist == f'modelerr_{i}')[0]) for i in range(self.n_components)]
        if self.host_component:
            self.errmin += [np.min(np.where(self.parlist == 'modelerr_host')[0])]
            self.errmax += [np.max(np.where(self.parlist == 'modelerr_host')[0])]
        self.errmin = tuple(self.errmin)
        self.errmax = tuple(self.errmax)

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
        
        self.iCL = np.where(self.parlist == 'cl')[0]
        self.ix0 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x0') or si.startswith('specx0')],dtype=int)
        self.ix1 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x1')],dtype=int)
        self.ixhost = np.array([i for i, si in enumerate(self.parlist) if si.startswith('xhost')],dtype=int)
        self.ic  = np.array([i for i, si in enumerate(self.parlist) if si.startswith('c_')],dtype=int)
        self.ispcrcl_norm = np.array([i for i, si in enumerate(self.parlist) if si.startswith('specx0')],dtype=int)
        if self.ispcrcl_norm.size==0: self.ispcrcl_norm=np.zeros(self.npar,dtype=bool)
        self.ispcrcl = np.array([i for i, si in enumerate(self.parlist) if si.startswith('spec')],dtype=int) # used to be specrecal
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

        self.iclscat_0,self.iclscat_poly = np.array([],dtype='int'),np.array([],dtype='int')
        if len(self.ispcrcl):
            for i,parname in enumerate(np.unique(self.parlist[self.iclscat])):
                self.iclscat_0 = np.append(self.iclscat_0,np.where(self.parlist == parname)[0][-1])
                self.iclscat_poly = np.append(self.iclscat_poly,np.where(self.parlist == parname)[0][:-1])
 
    
    def lsqwrap(self,guess,storedResults,varyParams=None,doPriors=True,doSpecResids=True,usesns=None):
        if varyParams is None:
            varyParams=np.zeros(self.npar,dtype=bool)
        if varyParams[self.imodelerr].any() or varyParams[self.imodelcorr].any() or varyParams[self.iclscat].any():
            raise ValueError('lsqwrap not allowed to handle varying model uncertainties')

        residuals = []
        jacobian = [] # Jacobian matrix from r
        for sn in self.datadict.keys() if usesns is None else usesns:
            photresidsdict,specresidsdict=self.ResidsForSN(guess,sn,storedResults,varyParams,fixuncertainties=True)
            for residsdict in ([photresidsdict,specresidsdict] if doSpecResids else [photresidsdict]):
                residuals+=[residsdict[k]['resid'] for k in residsdict]
                jacobian+=[residsdict[k]['resid_jacobian'] for k in residsdict]

        if doPriors:
            priorResids,priorVals,priorJac=self.priors.priorResids(guess)
            residuals+=[priorResids]
            jacobian+=[priorJac]
            BoundedPriorResids,BoundedPriorVals,BoundedPriorJac = \
                self.priors.BoundedPriorResids(guess)
            residuals+=[BoundedPriorResids]
            jacobian+=[sparse.csr_matrix(BoundedPriorJac)]

        if self.regularize:
            for regularization, weight,regKey in [(self.phaseGradientRegularization, self.regulargradientphase,'regresult_phase'),
                                           (self.waveGradientRegularization,self.regulargradientwave,'regresult_wave' ),
                                           (self.dyadicRegularization,self.regulardyad,'regresult_dyad')]:
                if weight ==0:
                    continue
                if regKey in storedResults and not (varyParams[self.im0].any() or varyParams[self.im1].any()):
                    residuals += storedResults[regKey]
                    jacobian +=  [sparse.csr_matrix((r.size,self.parlist.size)) for r in storedResults[regKey]]
                else:
                    for regResids,regJac,relativeweight in zip(
                            *regularization(guess,storedResults,varyParams),[1,self.m1regularization,self.mhostregularization]):
                        residuals += [regResids*np.sqrt(weight*relativeweight)]
                        if np.isnan(residuals[-1]).any(): import pdb;pdb.set_trace()
                        jacobian+=[sparse.csr_matrix(regJac)*np.sqrt(weight*relativeweight)]
                    storedResults[regKey]=residuals[-self.n_components:]

        if varyParams.any():
            return np.concatenate(residuals),sparse.vstack(jacobian)
        else:
            return  np.concatenate(residuals)

    def maxlikefit(
            self,x,storedResults=None,varyParams=None,pool=None,debug=False,
            fixfluxes=False,dospec=True,usesns=None):
        """
        Calculates the likelihood of given SALT model to photometric and spectroscopic data given during initialization
        
        Parameters
        ----------
        x : array
            SALT model parameters
            
        pool :  multiprocessing.pool.Pool, optional
            Optional worker pool to be used for calculating chi2 values for each SN. If not provided, all work is done in root process
        
        debug : boolean, optional
            Debug flag
        
        Returns
        -------
        
        chi2: float
            Goodness of fit of model to training data   
        """
        
        if storedResults is None: storedResults={}
        if varyParams is None:
            varyParams=np.zeros(self.npar,dtype=bool)
        computeDerivatives=np.any(varyParams)
        
        chi2 = 0
        #Construct arguments for maxlikeforSN method
        #If worker pool available, use it to calculate chi2 for each SN; otherwise, do it in this process
        args=[(x,sn,storedResults,varyParams,debug,fixfluxes,dospec) \
              for sn in (self.datadict.keys() if usesns is None else usesns)]

        mapFun=pool.map if pool else starmap


        result=list(mapFun(self.loglikeforSN,args))


        loglike=sum(result)
        logp = loglike

        if computeDerivatives:
            loglike=sum([r[0] for r in result])
            grad=sum([r[1] for r in result])
        else:
            loglike=sum(result)
        logp = loglike
        if len(self.usePriors):
            priorResids,priorVals,priorJac=self.priors.priorResids(x)   
            logp -=(priorResids**2).sum()/2
            if computeDerivatives:
                grad-= (priorResids [:,np.newaxis] * priorJac).sum(axis=0)
            BoundedPriorResids,BoundedPriorVals,BoundedPriorJac = \
            self.priors.BoundedPriorResids(x)
            logp -=(BoundedPriorResids**2).sum()/2
            if computeDerivatives:
                grad-= (BoundedPriorResids [:,np.newaxis] * BoundedPriorJac).sum(axis=0)


        if self.regularize:
            for regularization, weight in [
                    (self.phaseGradientRegularization, self.regulargradientphase),
                    (self.waveGradientRegularization,self.regulargradientwave ),
                    (self.dyadicRegularization,self.regulardyad)]:
                if weight ==0:
                    continue
                regResids,regJac=regularization(x,storedResults,varyParams)
                logp-= sum([(res**2).sum()*weight*componentweight/2 \
                            for res,componentweight in zip(regResids,[1,self.m1regularization,self.mhostregularization])])
                if computeDerivatives:
                    for res,jac,componentweight in zip(regResids,regJac,[1,self.m1regularization,self.mhostregularization]):
                        grad -= (res[:,np.newaxis]*jac *weight*componentweight).sum(axis=0)
        self.nstep += 1

        if computeDerivatives:
            return logp,grad
        else:
            return logp


    def calculatecachedvals(self,x):
        results={}
        for sn in self.datadict:
            sndata=self.datadict[sn]
            for flt in sndata.photdata:
                fluxkey=toidentifier(f'phot_fluxes_{sn}_{flt}')
                varkey = toidentifier(f'phot_variances_{sn}_{flt}')
                lcdata=sndata.photdata[flt]
                results[fluxkey]=lcdata.modelflux(x)
                results[varkey]=lcdata.modelfluxvariance(x),lcdata.colorscatter(x)
            for k in sndata.specdata:
                spectrum=sndata.specdata[k]
                fluxkey=toidentifier(f'spec_fluxes_{sn}_{k}')
                varkey= toidentifier(f'spec_variances_{sn}_{k}')
                results[fluxkey]=spectrum.modelflux(x)
                results[varkey]=spectrum.modelfluxvariance(x)
        return self.saltcachedresults(**results)

    def ResidsForSN(self,x,sn,cachedresults=None,fixuncertainties=False,fixfluxes=False):
        """ This method should be the only one required for any fitter to process the supernova data. 
        Find the residuals of a set of parameters to the photometric and spectroscopic data of a given supernova. 
        Photometric residuals are first decorrelated to diagonalize color scatter"""

        sndata=self.datadict[sn]
        photresids={}

        for flt in sndata.photdata:
            fluxkey=toidentifier(f'phot_fluxes_{sn}_{flt}')
            varkey = toidentifier(f'phot_variances_{sn}_{flt}')
            lcdata=sndata.photdata[flt]
            if fixfluxes:
                modelflux=getattr(cachedresults,fluxkey)
            else:
                modelflux=lcdata.modelflux(x)

            if fixuncertainties:
                 modelvariance,clscat=getattr(cachedresults,varkey)
            else:
                modelvariance=lcdata.modelfluxvariance(x)
                clscat=lcdata.colorscatter(x)

            variance=lcdata.fluxcalerr**2 + modelvariance            
            
            #if clscat>0, then need to use a cholesky matrix to find pulls
            def choleskyresidsandnorm( variance,clscat,modelflux):
                cholesky=jaxrankOneCholesky(variance,clscat**2,modelflux)
                return {'residuals':jaxlinalg.solve_triangular(cholesky, modelflux-lcdata.fluxcal,lower=True), 
                'lognorm': -jnp.log(jnp.diag(cholesky)).sum()}
            
            def diagonalresidsandnorm(variance,clscat,modelflux):
                sigma=jnp.sqrt(variance)
                return {'residuals':(modelflux-lcdata.fluxcal)/sigma,'lognorm': -jnp.log(sigma).sum()}

            photresids[flt]=lax.cond(clscat==0, diagonalresidsandnorm, choleskyresidsandnorm, 
            variance,clscat,modelflux )

        #Handle spectra
        specresids={}
        for k in sndata.specdata:
            fluxkey=toidentifier(f'spec_fluxes_{sn}_{k}')
            varkey=toidentifier( f'spec_variances_{sn}_{k}')
            spectrum=sndata.specdata[k]

            if fixfluxes:
                modelflux=getattr(cachedresults,fluxkey)
            else:
                modelflux=spectrum.modelflux(x)
            
            if fixuncertainties:
                 modelvariance=getattr(cachedresults,varkey)
            else:
                modelvariance=spectrum.modelfluxvariance(x)
            
            variance=spectrum.fluxerr**2 + modelvariance
              
            uncertainty=jnp.sqrt(variance)

            spectralSuppression=np.sqrt(self.num_phot/self.num_spec)*self.spec_chi2_scaling

            specresids[k]={'residuals': spectralSuppression * (modelflux-spectrum.flux)/uncertainty,
                        'lognorm': -jnp.log(uncertainty).sum()}

        return photresids,specresids
    
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
        
    
    def loglikeforSN(self,x,sn,cachedresults,fixuncertainties=False,fixfluxes=False,dospec=True):
        
        """
        Calculates the likelihood of given SALT model to photometric and spectroscopic observations of a single SN 

        Parameters
        ----------

        sn : str
            Name of supernova to compare to model
            
        x : array
            SALT model parameters
                    
        Returns
        -------
        chi2: float
            Model chi2 relative to training data    
        """

        photResidsDict,specResidsDict = self.ResidsForSN(
            x,sn,storedResults,fixuncertainties=fixuncertainties,fixfluxes=fixfluxes)

        loglike=0
        for residsdict in ([photResidsDict,specResidsDict] if dospec else [photResidsDict]):
            for key in residsdict:
                resids=residsdict[key]
                loglike+= resids['lognorm']- (resids['residuals']**2).sum() / 2.  
        return loglike
                
    
    def SALTModel(self,x,evaluatePhase=None,evaluateWave=None):
        """Returns flux surfaces of SALT model"""
        m0pars = x[self.m0min:self.m0max+1]

        if self.bsorder != 0:
            m0 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
                         self.wave if evaluateWave is None else evaluateWave,
                         (self.phaseknotloc,self.waveknotloc,m0pars,self.bsorder,self.bsorder))
        else:
            phase = self.phase if evaluatePhase is None else evaluatePhase
            wave = self.wave if evaluateWave is None else evaluateWave
            n_repeat_phase = int(phase.size/(self.phaseknotloc.size-1))+1
            n_repeat_phase_extra = -1*(n_repeat_phase*(self.phaseknotloc.size-1) % phase.size)
            if n_repeat_phase_extra == 0: n_repeat_phase_extra = None
            n_repeat_wave = int(wave.size/(self.waveknotloc.size-1))+1
            n_repeat_wave_extra = -1*(n_repeat_wave*(self.waveknotloc.size-1) % wave.size)
            if n_repeat_wave_extra == 0: n_repeat_wave_extra = None
            m0 = np.repeat(np.repeat(m0pars.reshape([self.phaseknotloc.size-1,self.waveknotloc.size-1]),n_repeat_phase,axis=0),
                           n_repeat_wave,axis=1)[:n_repeat_phase_extra,:n_repeat_wave_extra]

        

        if self.n_components >= 2 and self.n_components <= 3:
            m1pars = x[self.im1]
            if self.bsorder != 0:
                m1 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
                             self.wave if evaluateWave is None else evaluateWave,
                             (self.phaseknotloc,self.waveknotloc,m1pars,self.bsorder,self.bsorder))
            else:
                m1 = np.repeat(
                    np.repeat(m1pars.reshape([self.phaseknotloc.size-1,self.waveknotloc.size-1]),
                              n_repeat_phase,axis=0),n_repeat_wave,axis=1)[:n_repeat_phase_extra,:n_repeat_wave_extra]

            if self.n_components == 2 and not self.host_component:
                components = (m0,m1)
            elif self.host_component:
                mhostpars = x[self.imhost]
                if self.bsorder != 0:
                    mhost = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
                                 self.wave if evaluateWave is None else evaluateWave,
                                 (self.phaseknotloc,self.waveknotloc,mhostpars,self.bsorder,self.bsorder))
                else:
                    mhost = np.repeat(
                        np.repeat(mhostpars.reshape([self.phaseknotloc.size-1,self.waveknotloc.size-1]),
                                  n_repeat_phase,axis=0),n_repeat_wave,axis=1)[:n_repeat_phase_extra,:n_repeat_wave_extra]
                components = (m0,m1,mhost)
        elif self.n_components == 1:
            components = (m0,)
        else:
            raise RuntimeError('A maximum of three principal components is allowed')

        return components

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
    
    def colorscatter(self,x,wave,varyParams=None):
        clscatpars = x[self.parlist == 'clscat']
        pow=clscatpars.size-1-np.arange(clscatpars.size)
        coeffs=clscatpars/factorial(pow)
        clscat=np.exp(np.poly1d(coeffs)((wave-5500)/1000))
        if varyParams is None:
            return clscat
        else:
            pow=pow[varyParams[self.parlist=='clscat']]
            dcolorscatdx= clscat*(((wave-5500)/1000) ** (pow) )/ factorial(pow)
            return clscat,dcolorscatdx
        
    
    def ErrModel(self,x,evaluatePhase=None,evaluateWave=None):
        """Returns modeled variance of SALT model components as a function of phase and wavelength"""
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
            
        return components

    def CustomErrModel(self,m0var,m1var,m01cov,evaluatePhase=None,evaluateWave=None):
        """Returns modeled variance of SALT model components as a function of phase and wavelength"""

                
        return components

    
    def getParsGN(self,x):

        m0pars = x[self.parlist == 'm0']
        m1pars = x[self.parlist == 'm1']
        mhostpars = x[self.parlist == 'mhost']
        
        clpars = x[self.parlist == 'cl']
        clerr = np.zeros(len(x[self.parlist == 'cl']))
        

        clscat=self.colorscatter(x,self.wave)
        resultsdict = {}
        n_sn = len(self.datadict.keys())
        for k in self.datadict.keys():
            resultsdict[k] = {'x0':x[self.parlist == f'x0_{k}'][0],
                              'x1':x[self.parlist == f'x1_{k}'][0],
                              'c':x[self.parlist == f'c_{k}'][0],
                              'x0err':x[self.parlist == f'x0_{k}'][0],
                              'x1err':x[self.parlist == f'x1_{k}'][0],
                              'cerr':x[self.parlist == f'c_{k}'][0]
                              }             

        if self.host_component:
            m0,m1,mhost=self.SALTModel(x,evaluatePhase=self.phaseout,evaluateWave=self.waveout)
            m0err,m1err,mhosterr = self.ErrModel(x,evaluatePhase=self.phaseout,evaluateWave=self.waveout)
        else:
            m0,m1=self.SALTModel(x,evaluatePhase=self.phaseout,evaluateWave=self.waveout)
            mhost = np.zeros(np.shape(m0))
            m0err,m1err = self.ErrModel(x,evaluatePhase=self.phaseout,evaluateWave=self.waveout)
            mhosterr = np.zeros(np.shape(m0err))
        if not len(clpars): clpars = []

        # model errors
        if not self.host_component:
            cov_m0_m1 = self.CorrelationModel(x,evaluatePhase=self.phaseout,evaluateWave=self.waveout)[0]*m0err*m1err
            cov_m0_mhost = np.zeros(np.shape(cov_m0_m1))
        else:
            cov_m0_m1 = self.CorrelationModel(x,evaluatePhase=self.phaseout,evaluateWave=self.waveout)[0]*m0err*m1err
            cov_m0_mhost = self.CorrelationModel(x,evaluatePhase=self.phaseout,evaluateWave=self.waveout)[1]*m0err*mhosterr
        modelerr=np.ones(m0err.shape)

        return(x,self.phaseout,self.waveout,m0,m0err,m1,m1err,mhost,mhosterr,cov_m0_m1,cov_m0_mhost,modelerr,
               clpars,clerr,clscat,resultsdict)

    def getErrsGN(self,m0var,m1var,m0m1cov):

        errs=[]
        for errpars in [m0var,m1var,m0m1cov]:
            if self.bsorder != 0:
                errs+=[  bisplev(self.phase,
                                 self.wave,
                                 (self.phaseknotloc,self.waveknotloc,errpars,self.bsorder,self.bsorder))]
            else:
                n_repeat_phase = int(self.phase.size/(self.phaseknotloc.size-1))+1
                n_repeat_phase_extra = -1*(n_repeat_phase*(self.phaseknotloc.size-1) % self.phase.size)
                if n_repeat_phase_extra == 0: n_repeat_phase_extra = None
                n_repeat_wave = int(self.wave.size/(self.waveknotloc.size-1))+1
                n_repeat_wave_extra = -1*(n_repeat_wave*(self.waveknotloc.size-1) % self.wave.size)
                if n_repeat_wave_extra == 0: n_repeat_wave_extra = None
                errs += [np.repeat(np.repeat(errpars.reshape([self.phaseknotloc.size-1,self.waveknotloc.size-1]),n_repeat_phase,axis=0),n_repeat_wave,axis=1)[:n_repeat_phase_extra,:n_repeat_wave_extra]]


        return(np.sqrt(errs[0]),np.sqrt(errs[1]),errs[2])

    
    def getPars(self,loglikes,x,nburn=500,mkplots=False):

        axcount = 0; parcount = 0
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_pages = PdfPages(f'{self.outputdir}/MCMC_hist.pdf')
        fig = plt.figure()

        m0pars = np.array([])
        m0err = np.array([])
        for i in self.im0:
            m0pars = np.append(m0pars,x[i,nburn:].mean())
            m0err = np.append(m0err,x[i,nburn:].std())
            if mkplots:
                if not parcount % 9:
                    subnum = axcount%9+1
                    ax = plt.subplot(3,3,subnum)
                    axcount += 1
                    md,std = np.mean(x[i,nburn:]),np.std(x[i,nburn:])
                    histbins = np.linspace(md-3*std,md+3*std,50)
                    ax.hist(x[i,nburn:],bins=histbins)
                    ax.set_title('M0')
                    if axcount % 9 == 8:
                        pdf_pages.savefig(fig)
                        fig = plt.figure()
                parcount += 1

        m1pars = np.array([])
        m1err = np.array([])
        parcount = 0
        for i in self.im1:
            m1pars = np.append(m1pars,x[i,nburn:].mean())
            m1err = np.append(m1err,x[i,nburn:].std())
            if mkplots:
                if not parcount % 9:
                    subnum = axcount%9+1
                    ax = plt.subplot(3,3,subnum)
                    axcount += 1
                    md,std = np.mean(x[i,nburn:]),np.std(x[i,nburn:])
                    histbins = np.linspace(md-3*std,md+3*std,50)
                    ax.hist(x[i,nburn:],bins=histbins)
                    ax.set_title('M1')
                    if axcount % 9 == 8:
                        pdf_pages.savefig(fig)
                        fig = plt.figure()
                parcount += 1

        # covmat (diagonals only?)
        m0_m1_cov = np.zeros(len(m0pars))
        chain_len = len(m0pars)
        m0mean = np.repeat(x[self.im0,nburn:].mean(axis=0),np.shape(x[self.im0,nburn:])[0]).reshape(np.shape(x[self.im0,nburn:]))
        m1mean = np.repeat(x[self.im1,nburn:].mean(axis=0),np.shape(x[self.im1,nburn:])[0]).reshape(np.shape(x[self.im1,nburn:]))
        m0var = x[self.im0,nburn:]-m0mean
        m1var = x[self.im1,nburn:]-m1mean
        for i in range(len(m0pars)):
            for j in range(len(m1pars)):
                if i == j: m0_m1_cov[i] = np.sum(m0var[j,:]*m1var[i,:])
        m0_m1_cov /= chain_len


        modelerrpars = np.array([])
        modelerrerr = np.array([])
        for i in np.where(self.parlist == 'modelerr')[0]:
            modelerrpars = np.append(modelerrpars,x[i,nburn:].mean())
            modelerrerr = np.append(modelerrerr,x[i,nburn:].std())

        clpars = np.array([])
        clerr = np.array([])
        for i in self.iCL:
            clpars = np.append(clpars,x[i,nburn:].mean())
            clerr = np.append(clpars,x[i,nburn:].std())

        clscatpars = np.array([])
        clscaterr = np.array([])
        for i in np.where(self.parlist == 'clscat')[0]:
            clscatpars = np.append(clpars,x[i,nburn:].mean())
            clscaterr = np.append(clpars,x[i,nburn:].std())



        result=np.mean(x[:,nburn:],axis=1)

        resultsdict = {}
        n_sn = len(self.datadict.keys())
        for k in self.datadict.keys():
            resultsdict[k] = {'x0':x[self.parlist == f'x0_{k}',nburn:].mean(),
                              'x1':x[self.parlist == f'x1_{k}',nburn:].mean(),
                              'c':x[self.parlist == f'c_{k}',nburn:].mean(),
                              'x0err':x[self.parlist == f'x0_{k}',nburn:].std(),
                              'x1err':x[self.parlist == f'x1_{k}',nburn:].std(),
                              'cerr':x[self.parlist == f'c_{k}',nburn:].std(),
                              }


        m0 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars,self.bsorder,self.bsorder))
        m0errp = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars+m0err,self.bsorder,self.bsorder))
        m0errm = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars-m0err,self.bsorder,self.bsorder))
        m0err = (m0errp-m0errm)/2.
        if len(m1pars):
            m1 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars,self.bsorder,self.bsorder))
            m1errp = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars+m1err,self.bsorder,self.bsorder))
            m1errm = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars-m1err,self.bsorder,self.bsorder))
            m1err = (m1errp-m1errm)/2.
        else:
            m1 = np.zeros(np.shape(m0))
            m1err = np.zeros(np.shape(m0))

        cov_m0_m1 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0_m1_cov,self.bsorder,self.bsorder))
        modelerr = bisplev(self.phase,self.wave,(self.errphaseknotloc,self.errwaveknotloc,modelerrpars,self.bsorder,self.bsorder))

        clscat = self.colorscatter(np.mean(x[:,nburn:],axis=1),self.wave)
        if not len(clpars): clpars = []

        for snpar in ['x0','x1','c']:
            subnum = axcount%9+1
            ax = plt.subplot(3,3,subnum)
            axcount += 1
            md = np.mean(x[self.parlist == f'{snpar}_{k}',nburn:])
            std = np.std(x[self.parlist == f'{snpar}_{k}',nburn:])
            histbins = np.linspace(md-3*std,md+3*std,50)
            ax.hist(x[self.parlist == f'{snpar}_{k}',nburn:],bins=histbins)
            ax.set_title(f'{snpar}_{k}')
            if axcount % 9 == 8:
                pdf_pages.savefig(fig)
                fig = plt.figure()


        pdf_pages.savefig(fig)          
        pdf_pages.close()
        
        return(result,self.phase,self.wave,m0,m0err,m1,m1err,cov_m0_m1,modelerr,
               clpars,clerr,clscat,resultsdict)

            
    def updateEffectivePoints(self,x):
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
                

        self.neffRaw=gaussian_filter1d(self.neffRaw,self.phaseSmoothingNeff,0)
        self.neffRaw=gaussian_filter1d(self.neffRaw,self.waveSmoothingNeff,1)

        self.neff=self.neffRaw.copy()
        self.neff[self.neff>self.neffMax]=np.inf

        if not np.any(np.isinf(self.neff)): log.warning('Regularization is being applied to the entire phase/wavelength space: consider lowering neffmax (currently {:.2e})'.format(self.neffMax))
        
        self.neff=np.clip(self.neff,self.neffFloor,None)

        
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
    
    def regularizationScale(self,x,regmethod='none'):
        if self.regularizationScaleMethod=='fixed':
            return self.guessScale   
        else:
            raise ValueError('Regularization scale method invalid: ',self.regularizationScaleMethod)

    def dyadicRegularization(self,x):

        scales=self.regularizationScale(x)
        componentidxs=[self.im0,self.im1]
        if self.host_component:
            componentidxs+=[self.imhost]
        
        for scale,idxs in zip(scales,componentidxs):
            coeffs=x[idxs]
            fluxes= self.componentderiv @ coeffs
            dfluxdwave=self.dcompdphasederiv @ coeffs
            dfluxdphase=self.dcompdwavederiv @ coeffs
            d2fluxdphasedwave=self.ddcompdwavedphase @ coeffs

            #Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
            normalization=np.sqrt(1/( (self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))**2.
            #0 if model is locally separable in phase and wavelength i.e. flux=g(phase)* h(wavelength) for arbitrary functions g and h
            numerator=(dfluxdphase *dfluxdwave -d2fluxdphasedwave *fluxes )
            yield normalization* (numerator / (scale**2 * self.neff.flatten()))      

    
    def phaseGradientRegularization(self, x):
        scales=self.regularizationScale(x)
        componentidxs=[self.im0,self.im1]
        if self.host_component:
            componentidxs+=[self.imhost]
        
        for scale,idxs in zip(scales,componentidxs):
            coeffs=x[idxs]
            dfluxdphase=self.dcompdphasederiv @ coeffs
            normalization=np.sqrt(1/( (self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))
            yield normalization* ( dfluxdphase/scale / self.neff.flatten())

    
    def waveGradientRegularization(self, x):
        scales=self.regularizationScale(x)
        componentidxs=[self.im0,self.im1]
        if self.host_component:
            componentidxs+=[self.imhost]
        
        for scale,idxs in zip(scales,componentidxs):
            coeffs=x[idxs]
            dfluxdwave=self.dcompdwavederiv @ coeffs
            normalization=np.sqrt(1/( (self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))
            yield normalization* ( dfluxdwave/scale / self.neff.flatten())
