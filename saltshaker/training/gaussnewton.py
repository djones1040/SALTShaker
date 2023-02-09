#!/usr/bin/env python

import numpy as np
from numpy.random import standard_normal
from numpy.linalg import inv,pinv,norm

import time,copy,extinction,pickle

from scipy.interpolate import splprep,splev,BSpline,griddata,bisplev,bisplrep,interp1d,interp2d
from scipy.integrate import trapz, simps
from scipy.optimize import minimize, least_squares,minimize_scalar,lsq_linear
from scipy.ndimage import gaussian_filter1d
from scipy.special import factorial
from scipy import linalg,sparse
from scipy.sparse import linalg as sprslinalg
import scipy.stats as ss

from sncosmo.salt2utils import SALT2ColorLaw
from sncosmo.models import StretchSource
from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
from sncosmo.utils import integration_grid

from saltshaker.config.configparsing import *

from saltshaker.util.synphot import synphot
from saltshaker.util.query import query_yes_no
from saltshaker.training import saltresids

from multiprocessing import Pool, get_context
from iminuit import Minuit
from datetime import datetime
from itertools import starmap
from os import path
from functools import partial
from collections import namedtuple

import jax
from jax import numpy as jnp


import sys
import iminuit,warnings
import logging
log=logging.getLogger(__name__)


def in_ipynb():
    try:
        cfg = get_ipython().config 
        return True

    except NameError:
        return False

if in_ipynb():
    from tqdm.notebook import tqdm,trange
else:
    from tqdm import tqdm,trange

usetqdm= sys.stdout.isatty() or in_ipynb()

#Which integer code corresponds to which reason for LSMR terminating
stopReasons=['x=0 solution','atol approx. solution','atol+btol approx. solution','ill conditioned','machine precision limit','machine precision limit','machine precision limit','max # of iteration']
        
gnfitresult=namedtuple('gnfitresult',['lsmrresult','postGN','gaussNewtonStep','resids','damping','reductionratio'])
lsmrresult=namedtuple('lsmrresult',['precondstep','stopsignal','itn','normr','normar','norma','conda','normx'] )

class SALTTrainingResult(object):
     def __init__(self, **kwargs):
         self.__dict__.update(kwargs)

def ensurepositivedefinite(matrix,maxiter=5):

    for i in range(maxiter):
        mineigenval=np.linalg.eigvalsh(matrix)[0]
        if mineigenval>0:
            return matrix
        else:
            if maxiter==0: 
                raise ValueError('Unable to make matrix positive semidefinite')
        matrix+=np.diag(-mineigenval*4* np.ones(matrix.shape[0]))

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
    design=sparse.diags(diagonals,offsets).tocsr()
    for i in range(window.size//2+1):
        design[i,:window.size//2+1]=gaussian_filter1d(1.*(np.arange(design.shape[0])== i ),5)[:window.size//2+1]
        design[-i-1,-(window.size//2+1) : ]=gaussian_filter1d(1.*(np.arange(design.shape[0])== i ),5)[:window.size//2+1][::-1]    
    return design

def isDiag(M):
    i, j = M.shape
    assert i == j 
    test = M.reshape(-1)[:-1].reshape(i-1, j+1)
    return ~np.any(test[:, 1:])

def ridders(f,central,h,maxn,tol):
    """Iterative method to evaluate the second derivative of a function f based on a stepsize h and a relative tolerance tol"""
    lookup={}
    def A(n,m):
        if (n,m) in lookup: return lookup[n,m]
        if n==1:
            result=(f((h/2**(m-1)))-2*central+f((-h/2**(m-1))))/(h/2**(m-1))**2
        elif n<1:
            return 0
        else:
            result =(4**(n-1)*A(n-1,m+1)-A(n-1,m))/(4**(n-1)-1)
        lookup[(n,m)]=result
        return result
    def AwithErr(n):
        diff=A(n,1)-A(n-1,1)
        return A(n,1),  norm(diff)/ min(norm(A(n,1)),norm(A(n-1,1)))
    best=AwithErr(2)
    result,err=best
    prev=best[1]
    diverging=0
    errs=[prev]
    for n in range(3,maxn+1):
        if err<tol:
            log.debug(f'Second directional derivative found to tolerance {tol} after {n} iterations')
            best= result,err
            break
        elif err>prev:
            diverging+=1
            if diverging>2:
                log.warning(f'Second directional derivative diverging after {n} iterations, tolerance is {best[1]}')
                break
        else:
            diverging=0
            if err<best[1]:
                best=result,err
                diverging=0
        result,err=AwithErr(n)
        errs+=[err]
        prev=err
    return best

class fitting:
    def __init__(self,n_components,n_colorpars,
                 n_phaseknots,n_waveknots,datadict):

        self.n_phaseknots = n_phaseknots
        self.n_waveknots = n_waveknots
        self.n_components = n_components
        self.n_colorpars = n_colorpars
        self.datadict = datadict

    def gaussnewton(self,gn,guess,
            gaussnewton_maxiter,
            getdatauncertainties=True):

        gn.debug = False
        convergenceresult = \
             gn.convergence_loop(
                  guess,loop_niter=gaussnewton_maxiter,getdatauncertainties=getdatauncertainties)
            
        return convergenceresult,\
            'Gauss-Newton was successful'
        
    def mcmc(self,saltfitter,guess,
             n_processes,n_mcmc_steps,
             n_burnin_mcmc,stepsizes=None):
        
        saltfitter.debug = False
#       if n_processes > 1:
#           with InterruptiblePool(n_processes) as pool:
#               x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
#                   modelerr,clpars,clerr,clscat,SNParams = \
#                   saltfitter.mcmcfit(
#                       guess,n_mcmc_steps,n_burnin_mcmc,pool=None,stepsizes=stepsizes)
#       else:
        x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
                modelerr,clpars,clerr,clscat,SNParams = \
                saltfitter.mcmcfit(
                    guess,n_mcmc_steps,n_burnin_mcmc,pool=None,stepsizes=stepsizes)

        return x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
            modelerr,clpars,clerr,clscat,SNParams,'Adaptive MCMC was successful'


class mcmc(saltresids.SALTResids):
    def __init__(self,guess,datadict,parlist,chain=[],loglikes=[],**kwargs):
        self.loglikes=loglikes
        self.chain=chain

        super().__init__(guess,datadict,parlist,**kwargs)
        
        
    def get_proposal_cov(self, n, beta=0.25):
        d, _ = self.M2_recent.shape
        init_period = self.nsteps_before_adaptive
        s_0, s_opt, C_0 = self.AMpars['sigma_0'], self.AMpars['sigma_opt'], self.AMpars['C_0']
        if n<= init_period or np.random.rand()<=beta:
            return np.sqrt(C_0), False
        else:
            # We can always divide M2 by n-1 since n > init_period
            return np.sqrt((s_opt/(self.nsteps_adaptive_memory - 1))*self.M2_recent), True
    
    def generate_AM_candidate(self, current, n, steps_from_gn=False):
        prop_std,adjust_flag = self.get_proposal_cov(n)
        
        candidate = np.zeros(self.npar)
        candidate = np.random.normal(loc=current,scale=np.diag(prop_std))
        for i,par in zip(range(self.npar),self.parlist):
            if self.adjust_snpars and (par == 'm0' or par == 'm1' or par == 'modelerr'):
                candidate[i] = current[i]
            elif self.adjust_modelpars and par != 'm0' and par != 'm1' and par != 'modelerr':
                candidate[i] = current[i]
            else:
                if not steps_from_gn and (par.startswith('modelerr') or par.startswith('x0') or par == 'm0' or par == 'clscat'):
                    candidate[i] = current[i]*10**(0.4*np.random.normal(scale=prop_std[i,i]))
                else:
                    pass
        return candidate
        
    def get_propcov_init(self,x,stepsizes=None):
        C_0 = np.zeros([len(x),len(x)])
        if stepsizes is not None:
            for i,par in zip(range(self.npar),self.parlist):
                C_0[i,i] = stepsizes[i]**2.
        else:
            for i,par in zip(range(self.npar),self.parlist):
                if par == 'm0':
                    C_0[i,i] = self.stepsize_magscale_M0**2.
                elif par.startswith('modelerr'):
                    C_0[i,i] = (self.stepsize_magscale_err)**2.
                elif par == 'm1':
                    C_0[i,i] = (self.stepsize_magadd_M1)**2.
                elif par.startswith('x0'):
                    C_0[i,i] = self.stepsize_x0**2.
                elif par.startswith('x1'):
                    C_0[i,i] = self.stepsize_x1**2.
                elif par == 'clscat':
                    C_0[i,i] = (self.stepsize_magscale_clscat)**2.
                elif par.startswith('c'): C_0[i,i] = (self.stepsize_c)**2.
                elif par.startswith('specrecal'): C_0[i,i] = self.stepsize_specrecal**2.
                elif par.startswith('modelcorr'):
                    C_0[i,i]= self.stepsize_errcorr**2
        self.AMpars = {'C_0':C_0,
                       'sigma_0':0.1/np.sqrt(self.npar),
                       'sigma_opt':2.38*self.adaptive_sigma_opt_scale/np.sqrt(self.npar)}
    
    def update_moments(self, sample, n):
        next_n = (n + 1)
        w = 1/next_n
        new_mean = self.mean + w*(sample - self.mean)
        delta_bf, delta_af = sample - self.mean, sample - new_mean
        self.M2 += np.outer(delta_bf, delta_af)
        self.mean = new_mean

        return
    
    def mcmcfit(self,x,nsteps,nburn,pool=None,debug=False,thin=1,stepsizes=None):
        npar = len(x)
        self.npar = npar
        self.chain,self.loglikes = [],[]
        # initial log likelihood
        if self.chain==[]:
            self.chain+=[x]
        if self.loglikes==[]:
            self.loglikes += [self.maxlikefit(x,pool=pool,debug=debug)]
        self.M0stddev = np.std(x[self.parlist == 'm0'])
        self.M1stddev = np.std(x[self.parlist == 'm1'])
        self.errstddev = self.stepsize_magscale_err
        self.M2 = np.zeros([len(x),len(x)])
        self.M2_recent = np.empty_like(self.M2)
        self.mean = x[:], 

        if stepsizes is not None:
            steps_from_gn = True
            stepsizes[stepsizes > 0.1] = 0.1
            stepsizes *= 1e-14
        else: steps_from_gn = False
        self.get_propcov_init(x,stepsizes=stepsizes)
        accept = 0
        nstep = 0
        accept_frac = 0.5
        accept_frac_recent = 0.5
        accepted_history = np.array([])
        n_adaptive = 0
        self.adjust_snpars,self.adjust_modelpars = False,False
        while nstep < nsteps:
            nstep += 1
            n_adaptive += 1
            
            if not nstep % 50 and nstep > 250:
                accept_frac_recent = len(accepted_history[-100:][accepted_history[-100:] == True])/100.
            if self.modelpar_snpar_tradeoff_nstep:
                if not nstep % self.modelpar_snpar_tradeoff_nstep and nstep > self.nsteps_before_modelpar_tradeoff:
                    if self.adjust_snpars: self.adjust_modelpars = True; self.adjust_snpars = False
                    else: self.adjust_modelpars = False; self.adjust_snpars = True

            if self.use_lsqfit:
                if not (nstep+1) % self.nsteps_between_lsqfit:
                    X = self.lsqguess(current=self.chain[-1],snpars=True)
                if not (nstep) % self.nsteps_between_lsqfit:
                    X = self.lsqguess(current=self.chain[-1],doMangle=True)
                else:
                    X = self.generate_AM_candidate(current=self.chain[-1], n=nstep, steps_from_gn=steps_from_gn)
            else:
                X = self.generate_AM_candidate(current=self.chain[-1], n=nstep, steps_from_gn=steps_from_gn)
            self.__components_time_stamp__ = time.time()
            
            # loglike
            this_loglike = self.maxlikefit(X,pool=pool,debug=debug)
            accept_bool = self.accept(self.loglikes[-1],this_loglike)
            if accept_bool:
                if not nstep % thin:

                    self.chain+=[X]
                self.loglikes+=[this_loglike]

                accept += 1
                log.info('step = %i, accepted = %i, acceptance = %.3f, recent acceptance = %.3f'%(
                    nstep,accept,accept/float(nstep),accept_frac_recent))
            else:
                if not nstep % thin:

                    self.chain+=[self.chain[-1]]
                self.loglikes += [self.loglikes[-1]]

            accepted_history = np.append(accepted_history,accept_bool)
            if not (nstep) % self.nsteps_between_lsqfit:
                self.updateEffectivePoints(self.chain[-1])
            self.update_moments(self.chain[-1], n_adaptive)
            if not n_adaptive % self.nsteps_adaptive_memory:
                n_adaptive = 0
                
                self.M2_recent = np.empty_like(self.M2)
                self.M2_recent[:] = self.M2
                self.mean = self.chain[-1][:]
                self.M2 = np.empty_like(self.M2)

        log.info('acceptance = %.3f'%(accept/float(nstep)))
        if nstep < nburn:
            raise RuntimeError('Not enough steps to wait %i before burn-in'%nburn)
        xfinal,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
            modelerr,clpars,clerr,clscat,SNParams = \
            self.getPars(self.loglikes,np.array(self.chain).T,nburn=int(nburn/thin))
        
        return xfinal,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
            modelerr,clpars,clerr,clscat,SNParams
        
    def accept(self, last_loglike, this_loglike):
        alpha = np.exp(this_loglike - last_loglike)
        return_bool = False
        if alpha >= 1:
            return_bool = True
        else:
            if np.random.rand() < alpha:
                return_bool = True
        return return_bool



class GaussNewton:

    configoptionnames={}
    
    def __init__(self,guess,saltresids,outputdir,options):
            self.debug=False
            self.lsqfit = False
            self.modelobj=saltresids
            self.outputdir=outputdir
            for key in configoptionnames:
                self.__dict__[key]=options.__dict__[key]
                
            self.rngkey=jax.random.PRNGKey(18327534917853348)

            self.GN_iter = {}
            self.damping={}
            self.randomvjpevalfuns={}
            self.directionaloptimization=True
            self.geodesiccorrection=False
            self.updatejacobian=True
            self._robustify = False
            self._writetmp = False
            self.chi2_diff_cutoff = .1
            self.fitOptions={}
            self.iModelParam=np.ones(self.modelobj.npar,dtype=bool)
            self.iModelParam[self.modelobj.imodelerr]=False
            self.iModelParam[self.modelobj.imodelcorr]=False
            self.iModelParam[self.modelobj.iclscat]=False
            self.iModelParam[self.modelobj.ixhost]=False
            self.cachedpreconevalfuns={}
            self.Xhistory=[]
            self.tryFittingAllParams=True
            fitlist = [('all parameters','all'),('all parameters grouped','all-grouped'),('supernova params','sn'),
                       (" x0",'x0'),('both components','components'),('component 0 piecewise','piecewisecomponent0'),('principal component 0','component0'),('x1','x1'),
                       ('component 1 piecewise','piecewisecomponent1'),('principal component 1','component1'),('color','color'),('color law','colorlaw'),
                       ('spectral recalibration const.','spectralrecalibration_norm'),('all spectral recalibration','spectralrecalibration'),
                       ('error model','modelerr'),('params with largest residuals','highestresids'),('pca parameters only','pcaparams'), 
                       ('noncomponent parameters','snparams+colorlaw+recal'),('components and recalibration','components+recal')]+ [(f'sn {sn}',sn) for sn in self.modelobj.datadict.keys()]

            for message,fit in fitlist:
                self.GN_iter[fit]=1
                self.damping[fit]=0.1
                if 'all' in fit or fit=='highestresids':
                    includePars=np.ones(self.modelobj.npar,dtype=bool)
                    includePars[self.modelobj.ixhost] = False # we never want to fit the host coordinate
                else:
                    includePars=np.zeros(self.modelobj.npar,dtype=bool)
                    if fit in self.modelobj.datadict:
                        sn=fit
                        includePars=np.array([ sn in name.split('_') for name in self.modelobj.parlist])
                    elif fit=='components+recal':
                        includePars[self.modelobj.im0]=True
                        includePars[self.modelobj.im1]=True
                        includePars[self.modelobj.imhost]=True
                        includePars[self.modelobj.ispcrcl]=True
                    elif 'pcaparams' == fit:
                        self.GN_iter[fit]=2
                        includePars[self.modelobj.im0]=True
                        includePars[self.modelobj.im1]=True      
                        includePars[self.modelobj.ix0]=True
                        includePars[self.modelobj.ix1]=True
                    elif 'components' in fit:
                        includePars[self.modelobj.im0]=True
                        includePars[self.modelobj.im1]=True
                        includePars[self.modelobj.imhost]=True
                    elif 'component0' in fit :
                        self.damping[fit]=1e-3
                        includePars[self.modelobj.im0]=True
                    elif 'component1' in fit:
                        self.damping[fit]=1e-3
                        includePars[self.modelobj.im1]=True
                    elif 'componenthost' in fit:
                        self.damping[fit]=1e-3
                        includePars[self.modelobj.imhost]=True
                    elif fit=='sn':
                        self.damping[fit]=1e-3
                        includePars[self.modelobj.ix0]=True
                        includePars[self.modelobj.ix1]=True
                    elif fit=='snparams+colorlaw+recal':
                        includePars[self.modelobj.ix0]=True
                        includePars[self.modelobj.ix1]=True
                        includePars[self.modelobj.ic]=True
                        includePars[self.modelobj.iCL]=True
                        includePars[self.modelobj.ispcrcl]=True                
                    elif fit=='x0':
                        self.damping[fit]=0
                        includePars[self.modelobj.ix0]=True
                    elif fit=='x1':
                        self.damping[fit]=1e-3
                        includePars[self.modelobj.ix1]=True
                    elif fit=='color':
                        self.damping[fit]=1e-3
                        includePars[self.modelobj.ic]=True
                    elif fit=='colorlaw':
                        includePars[self.modelobj.iCL]=True
                    elif fit=='spectralrecalibration':
                        if len(self.modelobj.ispcrcl):
                            includePars[self.modelobj.ispcrcl]=True
                        else:
                            self.modelobj.ispcrcl = []
                    elif fit=='spectralrecalibration_norm':
                        if len(self.modelobj.ispcrcl_norm):
                            includePars[self.modelobj.ispcrcl_norm]=True
                        else:
                            self.modelobj.ispcrcl = []
                    elif fit=='modelerr':
                        includePars[self.modelobj.imodelerr]=True
                        includePars[self.modelobj.imodelcorr]=True
                        includePars[self.modelobj.parlist=='clscat']=True
                    else:
                        raise NotImplementedError("""This option for a Gauss-Newton fit with a 
        restricted parameter set has not been implemented: {}""".format(fit))
                if self.fix_salt2modelpars:
                    includePars[self.modelobj.im0]=False
                    includePars[self.modelobj.im1]=False
                    includePars[self.modelobj.im0new]=True
                    includePars[self.modelobj.im1new]=True
                elif self.fix_salt2components:
                    includePars[self.modelobj.im0]=False
                    includePars[self.modelobj.im1]=False
                    includePars[self.modelobj.iCL]=False
                    
                self.fitOptions[fit]=(message,includePars)

            if kwargs['fitting_sequence'].lower() == 'default' or not kwargs['fitting_sequence']:
                self.fitlist = [('all'),
                                ('pcaparams'),
                                ('color'),('colorlaw'),
                                ('spectralrecalibration'),      
                                ('sn')]
            else:
                self.fitlist = [f for f in kwargs['fitting_sequence'].split(',')]
            
    @classmethod
    def add_training_options(cls,parser,config):
        if parser == None:
            parser = ConfigWithCommandLineOverrideParser(usage=usage, conflict_handler="resolve")
        
        def wrapaddingargument(*args,**kwargs):
            cls.configoptionnames.add(args[2])
            return generateerrortolerantaddmethod(parser)

        successful=wrapaddingargument(config,'trainparams','n_repeat',  type=int,
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
        successful=successful&wrapaddingargument(config,'trainparams','gaussnewton_maxiter',     type=int,
                                                help='maximum iterations for Gauss-Newton (default=%(default)s)')
                                
        successful=successful&wrapaddingargument(config,'trainparams','lsmrmaxiter', type=int,
                                                help=' Allowed number of iterations for the linear solver of the Gauss-Newton procedure to run  (default=%(default)s)')
        successful=successful&wrapaddingargument(config,'trainparams','preconditioningmaxiter', type=int,
                                                help=' Number of operations used to evaluate preconditioning for the problem  (default=%(default)s)')

        successful=successful&wrapaddingargument(config,'trainparams','dampingscalerate', type=float,
                                                help=' Parameter that controls how quickly damping is adjusted in optimizer  (default=%(default)s)')

        successful=successful&wrapaddingargument(config,'trainparams','preconditioningchunksize', type=int,
                                                help='Size of batches to evaluate preconditioning scales in Gauss-Newton proces. Increasing this value may increase memory performance at the cost of speed (default=%(default)s)')

        successful=successful&wrapaddingargument(config,'trainparams','fitting_sequence',  type=str,
                                                        help="Order in which parameters are fit, 'default' or empty string does the standard approach, otherwise should be comma-separated list with any of the following: all, pcaparams, color, colorlaw, spectralrecalibration, sn (default=%(default)s)")
                                                        
        if not successful: sys.exit(1)

        return parser
    
    
    def convergence_loop(self,guess,loop_niter=3,usesns=None,getdatauncertainties=True):
        lastResid = 1e20
        log.info('Initializing')
        start=datetime.now()
        
        stepsizes=None
    
        X = copy.deepcopy(guess[:])
        clscatzeropoint=X[self.modelobj.iclscat[-1]]
        nocolorscatter=clscatzeropoint==-np.inf
        if not nocolorscatter: log.debug('Turning off color scatter for convergence_loop')
        X[self.modelobj.iclscat[-1]]=-np.inf

        cacheduncertainties= self.modelobj.calculatecachedvals(X,target='variances')
        Xlast = copy.deepcopy(guess[:])
        if np.all(X[self.modelobj.ix1]==0) or np.all(X[self.modelobj.ic]==0):
            #If snparams are totally uninitialized
            log.info('Estimating supernova parameters x0,x1,c and spectral normalization')
            for fit in ['x0','color','x0','color','x1']:
                X,chi2_init,chi2=self.process_fit(
                    X,self.fitOptions[fit][1],{},fit=fit,doPriors=False,
                    doSpecResids=  (fit=='x0'),allowjacupdate=False)
        else:

            chi2_init=(self.modelobj.lsqwrap(X,cacheduncertainties ,usesns=usesns)**2).sum()
        
        log.info(f'starting loop; {loop_niter} iterations')
        chi2results=self.modelobj.getChi2Contributions(X)
        for name,chi2component,dof in chi2results:
            if name.lower()=='photometric':
                photochi2perdof=chi2component/dof
        tryfittinguncertainties=False
        for superloop in range(loop_niter):
            tstartloop = time.time()
            try:
                if ((not superloop % self.steps_between_errorfit) or tryfittinguncertainties)  and  self.fit_model_err and not \
                   self.fit_cdisp_only and photochi2perdof<self.model_err_max_chisq and not superloop == 0:
                    X=self.iterativelyfiterrmodel(X)
                    chi2results=self.modelobj.getChi2Contributions(X)
                    cacheduncertainties= self.modelobj.calculatecachedvals(X,target='variances')
                    tryfittinguncertainties=False
                else:
                    log.info('Reevaluted model error')
                    chi2results=self.modelobj.getChi2Contributions(X)
                    cacheduncertainties= self.modelobj.calculatecachedvals(X,target='variances')
                
                for name,chi2component,dof in chi2results:
                    log.info('{} chi2/dof is {:.1f} ({:.2f}% of total chi2)'.format(name,chi2component/dof,chi2component/sum([x[1] for x in chi2results])*100))
                    if name.lower()=='photometric':
                        photochi2perdof=chi2component/dof

                X,chi2,converged = self.robust_process_fit(X,cacheduncertainties,chi2_init,superloop,usesns=usesns)
                if chi2_init-chi2 < -1.e-6:
                    log.warning("MESSAGE WARNING chi2 has increased")
                elif np.abs(chi2_init-chi2) < self.chi2_diff_cutoff:
                    if np.abs(photochi2perdof-1)>.2:
                        tryfittinguncertainties=True
                    else:
                        log.info(f'chi2 difference less than cutoff {self.chi2_diff_cutoff}, exiting loop')
                        break

                log.info(f'finished iteration {superloop+1}, chi2 improved by {chi2_init-chi2:.1f}')
                log.info(f'iteration {superloop+1} took {time.time()-tstartloop:.3f} seconds')

                if converged:
                    log.info('Gauss-Newton optimizer could not further improve chi2')
                    break
                chi2_init = chi2
                stepsizes = self.getstepsizes(X,Xlast)
                Xlast = copy.deepcopy(X)

            except KeyboardInterrupt as e:
                if query_yes_no("Terminate optimization loop and begin writing output?"):
                    break
                else:
                    if query_yes_no("Enter pdb?"):
                        import pdb;pdb.set_trace()
                    else:
                        raise e
            except Exception as e:
                logging.exception('Error encountered in convergence_loop, exiting')
                raise e
        X[self.modelobj.iclscat[-1]]=clscatzeropoint
        try:
            if self.fit_model_err: X= self.fitcolorscatter(X)
        except Exception as e:
            logging.critical('Color scatter crashed during fitting, finishing writing output')
            logging.critical(e, exc_info=True)

        Xredefined=self.modelobj.priors.satisfyDefinitions(X,self.modelobj.SALTModel(X))
        logging.info('Checking that rescaling components to satisfy definitions did not modify photometry')
        try:
            unscaledresults={}
            scaledresults={}

            Xtmp,Xredefinedtmp = X.copy(),Xredefined.copy()
            if self.options['no_transformed_err_check']:
                log.warning('parameter no_transformed_err_check set to True.  Use this option with bootstrap errors *only*')
                Xtmp[self.imodelerr0] = 0
                Xredefinedtmp[self.imodelerr0] = 0
                Xtmp[self.imodelerr1] = 0
                Xredefinedtmp[self.imodelerr1] = 0
                Xtmp[self.imodelerrhost] = 0
                Xredefinedtmp[self.imodelerrhost] = 0
                
            for sn in self.modelobj.datadict:
                for flt in self.modelobj.datadict[sn].photdata:
                    lcdata=self.modelobj.datadict[sn].photdata[flt]
                    photresidsunscaled=lcdata.modelresidual(Xtmp)
                    photresidsrescaled=lcdata.modelresidual(Xredefinedtmp)
                    assert(np.allclose(photresidsunscaled['residuals'],photresidsrescaled['residuals'],rtol=0.001,atol=1e-4))
        except AssertionError:
            logging.critical('Rescaling components failed; photometric residuals have changed. Will finish writing output using unscaled quantities')
            Xredefined=X.copy()

        if getdatauncertainties:
            M0dataerr, M1dataerr, Mhostdataerr, cov_M0_M1_data, cov_M0_Mhost_data =self.datauncertaintiesfromhessianapprox(Xredefined)
        else:
            M0dataerr, M1dataerr, Mhostdataerr, cov_M0_M1_data, cov_M0_Mhost_data =None,None,None,None,None
        # M0/M1 errors
        xfinal,phase,wave,M0,M0modelerr,M1,M1modelerr,Mhost,Mhostmodelerr,cov_M0_M1_model,cov_M0_Mhost_model,\
            modelerr,clpars,clerr,clscat,SNParams = \
            self.modelobj.getPars(Xredefined)
        if M0dataerr is None:
            M0dataerr      = np.zeros((self.modelobj.phaseout.size,self.modelobj.waveout.size))
            cov_M0_M1_data = np.zeros((self.modelobj.phaseout.size,self.modelobj.waveout.size))
            M1dataerr      = np.zeros((self.modelobj.phaseout.size,self.modelobj.waveout.size))

            Mhostdataerr = np.zeros((self.modelobj.phaseout.size,self.modelobj.waveout.size))
            cov_M0_Mhost_data = np.zeros((self.modelobj.phaseout.size,self.modelobj.waveout.size))
            
        log.info('Total time spent in convergence loop: {}'.format(datetime.now()-start))
        

        return SALTTrainingResult(
            num_lightcurves=self.modelobj.num_lc,num_spectra=self.modelobj.num_spectra,num_sne=len(self.modelobj.datadict),
            parlist=self.modelobj.parlist,X=xfinal,X_raw=X,phase=phase,wave=wave,M0=M0,M0modelerr=M0modelerr,M0dataerr=M0dataerr,
            M1=M1,Mhost=Mhost,M1modelerr=M1modelerr,M1dataerr=M1dataerr,Mhostdataerr=Mhostdataerr,Mhostmodelerr=Mhostmodelerr,
            cov_M0_M1_model=cov_M0_M1_model,cov_M0_M1_data=cov_M0_M1_data,cov_M0_Mhost_model=cov_M0_Mhost_model,cov_M0_Mhost_data=cov_M0_Mhost_data,
            modelerr=modelerr,clpars=clpars,clerr=clerr,clscat=clscat,SNParams=SNParams,stepsizes=stepsizes)
        
        

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


    
    def fitcolorscatter(self,X,fitcolorlaw=False,rescaleerrs=True,maxiter=2000):
        message='Optimizing color scatter'
        if rescaleerrs:
            message+=' and scaling model uncertainties'
        if fitcolorlaw:
            message+=', with color law varying'
        log.info(message)
        if X[self.modelobj.iclscat[-1]]==-np.inf:
            X[self.modelobj.iclscat[-1]]=-8
        includePars=np.zeros(self.modelobj.parlist.size,dtype=bool)
        includePars[self.modelobj.iclscat]=True
        includePars[self.modelobj.iCL]=fitcolorlaw
        fixfluxes=not fitcolorlaw
        if fixfluxes: cachedfluxes = self.modelobj.calculatecachedvals(X,target='fluxes')
        else: cachedfluxes=None
        log.info('Initialized log likelihood: {:.2f}'.format(self.modelobj.maxlikefit(X)))
        
        X,minuitresult=self.minuitoptimize(X,includePars,cachedresults=cachedfluxes ,rescaleerrs=rescaleerrs,fixfluxes=fixfluxes, dopriors=False,dospec=False)
        log.info('Finished optimizing color scatter')
        log.debug(str(minuitresult))
        return X

    def iterativelyfiterrmodel(self,X):
        log.info('Optimizing model error')
        X=X.copy()
        imodelerr=np.zeros(self.modelobj.parlist.size,dtype=bool)
        imodelerr[self.modelobj.imodelerr]=True
        problemerrvals=(X<0)&imodelerr
        X[problemerrvals]=1e-3
        if np.isnan(X[self.modelobj.iclscat[-1]]):
            X[self.modelobj.iclscat[-1]]=-np.inf
        X0=X.copy()
        mapFun= starmap
        cachedfluxes= self.modelobj.calculatecachedvals(X,target='fluxes')
        
        partriplets= list(zip(np.where(self.modelobj.parlist=='modelerr_0')[0],np.where(self.modelobj.parlist=='modelerr_1')[0],np.where(self.modelobj.parlist=='modelcorr_01')[0]))
        if self.modelobj.host_component:
            partriplets= list(zip(np.where(self.modelobj.parlist=='modelerr_0')[0],np.where(self.modelobj.parlist=='modelerr_1')[0],np.where(self.modelobj.parlist=='modelcorr_01')[0],
                                  np.where(self.modelobj.parlist=='modelerr_host')[0],np.where(self.modelobj.parlist=='modelcorr_0host')[0]))
        
        for i,parindices in tqdm(list(enumerate(partriplets))):
            includePars=np.zeros(self.modelobj.parlist.size,dtype=bool)
            includePars[list(parindices)]=True

            X,minuitfitresult=self.minuitoptimize(X,includePars,cachedresults=cachedfluxes,fixfluxes=True,dopriors=False,dospec=False)

        log.info('Finished model error optimization')
        return X

    def minuitoptimize(self,X,includePars,rescaleerrs=False,tol=0.1,*args,**kwargs):
        X=X.copy()
        if includePars.dtype==bool:
            includePars= np.where(includePars)[0]
            
        if not rescaleerrs:
            def fn(Y):
                
                Xnew=X.copy()
                Xnew[includePars]=Y
                result=-self.modelobj.maxlikefit(Xnew,*args,**kwargs)
                return 1e10 if np.isnan(result) else result
            def grad(Y):
                Xnew=X.copy()
                Xnew[includePars]=Y
                grad=-self.modelobj.maxlikefit(Xnew,*args,**kwargs,diff='grad')
                grad=grad[includePars]
                return np.ones(Y.size) if np.isnan(grad ).any() else grad

        else:
            def fn(Y):
                Xnew=X.copy()
                Xnew[includePars]=Y[:-1]
                Xnew[self.modelobj.imodelerr]*=Y[-1]
                result=-self.modelobj.maxlikefit(Xnew,*args,**kwargs)
                return (1e10 if np.isnan(result) else result)

            def grad(Y):
                Xnew=X.copy()
                Xnew[includePars]=Y[:-1]
                Xnew[self.modelobj.imodelerr]*=Y[-1]
                grad=-self.modelobj.maxlikefit(Xnew,*args,**kwargs,diff='grad')
                grad=np.concatenate((grad[includePars], [np.dot(grad[self.modelobj.imodelerr], X[self.modelobj.imodelerr])]))
                return np.ones(Y.size) if np.isnan(grad ).any() else grad
        initvals=X[includePars].copy()

        if rescaleerrs:
            initvals=np.concatenate((initvals, [1]))

        m=Minuit(fn,initvals,grad=grad)
        m.errordef=0.5
        m.errors = np.tile(1e-2, initvals.size)
        m.tol=tol
        
        lowerlims=np.tile(-np.inf,includePars.size)
        upperlims=np.tile( np.inf,includePars.size)
        clscatindices=np.where(self.modelobj.parlist[includePars] == 'clscat')[0]
        if clscatindices.size>0:
            lowerlims[clscatindices[0]]=-1e-4
            lowerlims[clscatindices[1:-1]]=-1 
            lowerlims[clscatindices[-1]]=-10
            
            upperlims[clscatindices[0]]=1e-4
            upperlims[clscatindices[1:-1]]=1 
            upperlims[clscatindices[-1]]=2
        onebounded=np.array( [x.startswith('modelcorr') for x in  self.modelobj.parlist[includePars] ])
        lowerlims[onebounded]=-1
        upperlims[onebounded]=1
        positivebounded=np.array([x.startswith('modelerr') for x in self.modelobj.parlist[includePars]])
        lowerlims[positivebounded]=0
        upperlims[positivebounded]=1
        
        lowerlims[self.modelobj.parlist[includePars] == 'cl'] = -100
        upperlims[self.modelobj.parlist[includePars] == 'cl'] = 100
    

        if rescaleerrs:
            lowerlims=np.concatenate((lowerlims,[0]))
            upperlims=np.concatenate((upperlims,[2]))
        m.limits=list(zip(lowerlims,upperlims))
        try:
            result=m.migrad()
        except KeyboardInterrupt:
            logging.info('Keyboard interrupt, exiting minuit loop')
            return X,None
        X=X.copy()

        paramresults=np.array(m.values)
        if rescaleerrs:
            X[includePars]=paramresults[:-1]
            X[self.modelobj.imodelerr]*=paramresults[-1]
        else:
            X[includePars]=paramresults

        return X,result


    def minuitoptimize_components(self,X,includeM0Pars,includeM1Pars,uncertainties=None,varyParams=None,**kwargs):

        X=X.copy()
        if uncertainties is None: uncertainties={}
        def fn(Y):
            Xnew=X.copy()
            Xnew[includeM0Pars] += Y[0]
            Xnew[includeM1Pars] += Y[1]
            storedCopy = uncertainties.copy()
            if 'components' in storedCopy.keys():
                storedCopy.pop('components')
            result=-self.modelobj.maxlikefit(Xnew,{})
            return 1e10 if np.isnan(result) else result
            
        params=['x0','x1']
        minuitkwargs={'x0':0,'x1':1}
        #minuitkwargs.update({'error_x0': 1e-2,'error_x1': 1e-2})
        #minuitkwargs.update({'limit_x0': (-3,3),'limit_x1': (-3,3)})

        m=Minuit(fn,name=params,**minuitkwargs)
        m.errordef = 0.5
        m.errors = {'error_x0':1e-2,'error_x1':1e-2}
        m.limits = {'limit_x0':(-3,3),'limit_x1':(-3,3)}
        result,paramResults=m.migrad()

        if m.covariance:
            return paramResults[0].error**2.,paramResults[1].error**2.,m.covariance[('x0', 'x1')]
        else:
            return paramResults[0].error**2.,paramResults[1].error**2.,0.0

    
    def getstepsizes(self,X,Xlast):
        stepsizes = X-Xlast
        return stepsizes
        
    
    def robust_process_fit(self,X_init,uncertainties,chi2_init,niter,usesns=None):

        Xprop,chi2prop,chi2 = self.process_fit(X_init,self.fitOptions['all'][1],uncertainties,fit='all',usesns=usesns)
        if (chi2prop<chi2_init):
            return Xprop,chi2prop, False
        else:
            return X_init, chi2_init, True

    def linesearch(self,X,searchdir,*args,**kwargs):
        def opFunc(x):
            return ((self.modelobj.lsqwrap(X-(x*searchdir),*args,**kwargs))**2).sum()
        result,step,stepType=minimize_scalar(opFunc),searchdir,'Gauss-Newton'
        log.info('Linear optimization factor is {:.2f} x {} step'.format(result.x,stepType))
        log.info('Linear optimized chi2 is {:.2f}'.format(result.fun))
        return result.x*searchdir,result.fun
    
    def evaljacobipreconditioning(self,parindex,*args,**kwargs):
        jacrow=self.modelobj.lsqwrap(*args,**kwargs,diff='jvp',jit=False)((jnp.arange(self.modelobj.npar)==parindex)*1.)
        return  1/jnp.sqrt(jacrow@jacrow)
    
    
       

    def jacobipreconditioning(self,varyingParams,guess,uncertainties,dopriors=True,dospecresids=True,usesns=None):
        #Simple diagonal preconditioning matrix designed to make the jacobian better conditioned. Seems to do well enough! If output is showing significant condition number in J, consider improving this
        chunksize=self.preconditioningchunksize
        
        iterator = tqdm  if usetqdm else lambda x: x 
        targets=np.where(varyingParams)[0]
        if targets.size%chunksize!=0: 
        
            paddedtargets=np.concatenate((targets,np.tile( self.modelobj.npar,chunksize-targets.size%chunksize)))
        else: 
            paddedtargets=targets
        numchunks=(paddedtargets.size//chunksize)
        staticargs=(dopriors,dospecresids,usesns)
        if staticargs in self.cachedpreconevalfuns: 
            preconevalfun= self.cachedpreconevalfuns[staticargs]
            
        else:
            preconevalfun = jax.jit(jax.vmap(lambda parindex,x,y: self.evaljacobipreconditioning(parindex,x,y,*staticargs),
            
                    in_axes=(0,None,[[None]*len(self.modelobj.batchedphotdata),[None]*len(self.modelobj.batchedspecdata)])))
                    
            self.cachedpreconevalfuns[staticargs]=preconevalfun
        
        precon=jnp.concatenate([preconevalfun(paddedtargets[i*chunksize: (i+1)*chunksize] , guess,uncertainties) for i in iterator(range(numchunks)) ])
        return jnp.nan_to_num(precon[:targets.size])

    def stochasticbinormpreconditioning(self,includepars,*args,**kwargs):
        #https://web.stanford.edu/group/SOL/dissertations/bradley-thesis.pdf
        #Determines pre- and post- diagonal preconditioning matrices iteratively using only matrix-vector products
        #Requires many fewer iterations than the jacobi preconditioner
        if includepars.dtype==bool: includepars=np.where(includepars)[0]

        def substitute(x):
            y=np.copy(args[0])
            y[includepars]=x
            return y

        jshape= jax.eval_shape(lambda x: self.modelobj.lsqwrap(x,*args[1:],**kwargs) ,args[0]).shape[0],includepars.size

        jaclinop=sprslinalg.LinearOperator(matvec = lambda x: (self.modelobj.lsqwrap(*args,**kwargs,diff='jvp',jit=True)( substitute(x) )) ,

                                         rmatvec= lambda x: (self.modelobj.lsqwrap(*args,**kwargs,diff='vjp',jit=True)(x))[includepars] ,shape=(jshape))
        
        iterator = tqdm  if usetqdm else lambda x: x 
        r=np.ones(jshape[0])
        c=np.ones(jshape[1])

        #Number of matrix-vector products to perform
        nmv= jshape[1]//10
        for k in iterator(range( nmv)):
            omega=2**(-max(min(np.floor(np.log2(k+1))-1,4),1))
#                  Found that letting r vary wasn't giving much improvement, at least not obviously; might need to play with damping more. Regardless, for now I've fixed it to 1, and only doing postconditioning
#                 s=np.random.normal(size=jshape[1])/np.sqrt(c)
#                 y= jaclinop @ s
#                 r= (1-omega)*r/r.sum() + omega* y**2 / (y**2).sum()
            s= np.random.normal(size=(jshape[0])) / np.sqrt(r)
            y= jaclinop.T @ s
            cnew=(1-omega)*c/c.sum() + omega*y**2 /(y**2).sum()
            c=cnew
        x=1/np.sqrt(r)
        y=1/np.sqrt(c)

        return  y/np.median(y)*0.008


    def evalrandomvjp(self,key,*args,**kwargs):
        numresids=jax.eval_shape(lambda x: self.modelobj.lsqwrap(x,*args[1:],**kwargs) ,args[0]).shape[0]
        return self.modelobj.lsqwrap(*args,**kwargs,diff='vjp',jit=False)(jax.random.normal(key,shape=[numresids]))

    def vectorizedstochasticbinormpreconditioning(self,includepars,guess,uncertainties,dopriors=True,dospecresids=True,usesns=None,maxiter=None):
        if includepars.dtype==bool: includepars=np.where(includepars)[0]

        staticargs=(dopriors,dospecresids,usesns)
        jshape= jax.eval_shape(lambda x: self.modelobj.lsqwrap(x,uncertainties,dopriors,dospecresids,usesns) ,guess).shape[0],includepars.size

        if staticargs in self.randomvjpevalfuns: 
            preconevalfun= self.randomvjpevalfuns[staticargs]
            
        else:
            preconevalfun = jax.jit(jax.vmap(lambda key,x,y: self.evalrandomvjp(key,x,y,*staticargs),
            
                    in_axes=(0,None,[[None]*len(self.modelobj.batchedphotdata),[None]*len(self.modelobj.batchedspecdata)])))
                    
            self.randomvjpevalfuns[staticargs]=preconevalfun
        iterator = tqdm  if usetqdm else lambda x: x 
        r=np.ones(jshape[0])
        c=np.ones(jshape[1])
        

        if maxiter is None: 
            nmv= self.preconditioningmaxiter
        else:
            nmv= maxiter
        numblocks=(nmv//self.preconditioningchunksize) + ((nmv %self.preconditioningchunksize )>0)
        k=0
        for i in iterator(range(numblocks)):
            self.rngkey,veckey=jax.random.split(self.rngkey,2)

            veckey=jax.random.split(veckey,self.preconditioningchunksize)
            for y in preconevalfun(veckey,guess,uncertainties  ):
                k+=1
                omega=2**(-max(min(np.floor(np.log2(k))-1,4),1))
                y=y[includepars]
                c=(1-omega)*c/c.sum() + omega*y**2 /(y**2).sum()
                if np.isnan(c).any():
                    log.critical('NaN appeared in preconditioning term, setting to zero and attempting to reiterate')
                c=np.nan_to_num(c)
        y=1/np.sqrt(c)
        
        return  y/np.median(y)*0.008
        

        
    def constructoperator(self,precon,includepars,*args,**kwargs):
        if includepars.dtype==bool: includepars=np.where(includepars)[0]


        jshape= jax.eval_shape(lambda x: self.modelobj.lsqwrap(x,*args[1:],**kwargs) ,args[0]).shape[0],includepars.size
        @jax.jit
        def preconditioninverse(x):
            return jnp.zeros(self.modelobj.npar).at[includepars].set( precon *x)

        linop=sprslinalg.LinearOperator(matvec = lambda x: (self.modelobj.lsqwrap(*args,**kwargs,diff='jvp',jit=True)( preconditioninverse(x) )) ,

                                 rmatvec= lambda x: (self.modelobj.lsqwrap(*args,**kwargs,diff='vjp',jit=True)(x))[includepars]*precon ,shape=(jshape))

        return linop,preconditioninverse



    def gaussNewtonFit(self,initval,jacobian,preconinv,residuals,damping,lsqwrapargs, maxiter=None):

        tol=1e-8
        #import pdb; pdb.set_trace()
        initchi=(residuals**2).sum()
        if maxiter is None: maxiter= self.lsmrmaxiter
        result=lsmrresult(*sprslinalg.lsmr(jacobian,residuals,damp=damping,maxiter=maxiter,atol=tol,btol=tol))
        gaussNewtonStep= preconinv(result.precondstep)
        resids=self.modelobj.lsqwrap(initval-gaussNewtonStep,*lsqwrapargs[0],**lsqwrapargs[1])
        postGN=(resids**2).sum() #
        #if fit == 'all': import pdb; pdb.set_trace()
        log.debug(f'Attempting fit with damping {damping} gave chi2 {postGN} with {result.itn} iterations')
        reductionratio= (initchi -postGN)/(initchi-(result.normr**2))
        # np.unique(self.parlist[np.where(varyingParams != True)])
        return gnfitresult(result,postGN,gaussNewtonStep,resids,damping,reductionratio)
        
    def iteratedampings(self,fit,initval,jacobian,preconinv,residuals,lsqwrapargs):
        """Experiment with different amounts of damping in the fit"""
        scale=self.dampingscalerate
    
        oldChi=(residuals**2).sum()
        damping=self.damping[fit]
        currdamping=damping
        gnfitfun= lambda dampingval,**kwargs: self.gaussNewtonFit(initval, jacobian,preconinv,residuals,dampingval, lsqwrapargs, **kwargs)
        log.debug('Beginning iteration over damping')
        result=gnfitfun(damping )
        #Ratio of actual improvement in chi2 to how well the optimizer thinks it did
    
        #If the chi^2 is improving less than expected, check whether damping needs to be increased
        #If the new position is worse, need to increase the damping until the new position is better
        if (oldChi> result.postGN)  :
            newresult=gnfitfun(damping/scale)
            result=min([result,newresult],key=lambda x:x.postGN )
        if (oldChi< result.postGN) or (result.reductionratio<0.33 ) or np.isnan(result.postGN):
            maxiter=5
            for i in range(maxiter) :
                if result.postGN > 1e4*oldChi or np.isnan(result.postGN):
                    log.critical('Pathological result with chi2 {result.postGN:.3e}, attempting to iterate preconditioning')
                else:
             
                    if result.postGN > 100*oldChi:
                        damping*= 10
                    log.debug('Reiterating and increasing damping')
                    damping*=scale*11/9
                    result=min([result,newresult],key=lambda x:x.postGN )

                    if (oldChi>result.postGN): break
                    
            else:
                log.info(f'After increasing damping {maxiter} times, failed to find a result that improved chi2')
        log.debug(f'After iteration on input damping {currdamping:.2e} found best damping was {result.damping:.2e}')
        self.damping[fit]=result.damping
        return result

#     def geodesicgaussnewton(self):
#         directionalSecondDeriv,tol= ridders(lambda dx: self.lsqwrap(X+dx*gaussNewtonStep,uncertainties,**kwargs) ,residuals,.5,5,1e-8)
#         accelerationdir,stopsignal,itn,normr,normar,norma,conda,normx=sprslinalg.lsmr(precondjac,directionalSecondDeriv,damp=self.damping[fit],maxiter=2*min(jacobian.shape))
#         secondStep=np.zeros(X.size)
#         secondStep[varyingParams]=0.5*preconditioningmatrix*accelerationdir
# 
#         postgeodesic=(self.lsqwrap(X-gaussNewtonStep-secondStep,uncertainties,**kwargs)**2).sum() #doSpecResids
#         log.info('After geodesic acceleration correction chi2 is {:.2f}'.format(postgeodesic))
#         if postgeodesic<postGN :
#             chi2=postgeodesic
#             gaussNewtonStep=gaussNewtonStep+secondStep
#         else:
#             chi2=postGN
#             gaussNewtonStep=gaussNewtonStep

       
    def process_fit(self,initvals,iFit,uncertainties,fit='all',allowjacupdate=True,**kwargs):

        X=initvals.copy()
        varyingParams=iFit&self.iModelParam
        if 'usesns' in kwargs :
            if kwargs['usesns' ] is None:
                kwargs.pop('usesns')
            else:
                snnotinset=[sn for sn in self.modelobj.datadict if sn not in kwargs['usesns']]
                sndependentparams=np.prod([self.fitOptions[sn][1] for sn in snnotinset],axis=0).astype(bool)
                log.debug(f'Removing {sndependentparams.sum()} params because {len(snnotinset)} SNe are not included in this iteration')
                varyingParams=varyingParams&~sndependentparams
        residuals=self.modelobj.lsqwrap(X,uncertainties,**kwargs)
        oldChi=(residuals**2).sum()


        log.info('Number of parameters fit this round: {}'.format(varyingParams.sum()))
        log.info('Initial chi2: {:.2f} '.format(oldChi))
        log.info('Calculating preconditioning')

        preconditioning= self.vectorizedstochasticbinormpreconditioning(varyingParams,X,uncertainties,**kwargs)
        log.info('Finished preconditioning')

        jacobian,preconinv= self.constructoperator(preconditioning, varyingParams, X,uncertainties, **kwargs)
        lsqwrapargs=([uncertainties],kwargs)
        fittingfunction= (lambda *args: self.gaussNewtonFit(*args,0,lsqwrapargs)) if self.damping[fit]==0 else (lambda *args,**kwargs: self.iteratedampings(fit,*args,lsqwrapargs=lsqwrapargs,**kwargs))

        result= fittingfunction(X,jacobian,preconinv,residuals)
        log.debug(f'First Gauss-Newton step: LSMR results with damping factor {result.damping:.2e}: {stopReasons[result.lsmrresult.stopsignal]}, norm r {result.lsmrresult.normr:.2f}, norm J^T r {result.lsmrresult.normar:.2f}, norm J {result.lsmrresult.norma:.2f}, cond J {result.lsmrresult.conda:.2f}, norm step {result.lsmrresult.normx:.2f}, reduction ratio {result.reductionratio:.2f} required {result.lsmrresult.itn} iterations' )
        if result.lsmrresult.stopsignal==7: log.warning('Gauss-Newton solver reached max # of iterations')

        if np.any(np.isnan(result.gaussNewtonStep)):
            log.error('NaN detected in stepsize; exitting to debugger')
            import pdb;pdb.set_trace()
        X-=result.gaussNewtonStep

        log.info('After Gauss-Newton chi2 is {:.2f}'.format(result.postGN))


        if self.updatejacobian and allowjacupdate:  
            prevresult=result
            for i in range(10):
                self.Xhistory+=[(X,prevresult.postGN,prevresult)]
                
                jacobian,_= self.constructoperator(preconditioning, varyingParams, X,uncertainties, **kwargs)

                result=fittingfunction(X,jacobian,preconinv,prevresult.resids)
                
                chi2improvement=prevresult.postGN-result.postGN

                log.info(f'Reiterating with updated jacobian gives improvement {chi2improvement}')
                log.debug(f'On reiteration: LSMR results with damping factor {result.damping:.2e}: {stopReasons[result.lsmrresult.stopsignal]}, norm r {result.lsmrresult.normr:.2f}, norm J^T r {result.lsmrresult.normar:.2f}, norm J {result.lsmrresult.norma:.2f}, cond J {result.lsmrresult.conda:.2f}, norm step {result.lsmrresult.normx:.2f}, reduction ratio {result.reductionratio:.2f} required {result.lsmrresult.itn} iterations' )
                if chi2improvement<=0:
                    log.info('No improvement, finishing process_fit')
                    break
                else:
                    X-=result.gaussNewtonStep
                    chi2=result.postGN
                    prevresult=result
            self.Xhistory+=[(X ,prevresult.postGN,prevresult.lsmrresult)]

        if self.directionaloptimization:
            linearStep,chi2=self.linesearch(X,X-initvals,uncertainties,**kwargs)   
            X-=linearStep       


        with open(path.join(self.outputdir,'gaussnewtonhistory.pickle'),'wb') as file: pickle.dump(self.Xhistory,file)
        log.info('Chi2 diff, % diff')
        log.info(' '.join(['{:.2f}'.format(x) for x in [oldChi-chi2,(100*(oldChi-chi2)/oldChi)] ]))
        log.info('')
        return np.array(X.to_py()),chi2,oldChi

