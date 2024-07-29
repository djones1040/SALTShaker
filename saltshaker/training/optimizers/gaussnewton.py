#!/usr/bin/env python

import numpy as np
from numpy.random import standard_normal
from numpy.linalg import inv,pinv,norm

import time,copy,extinction,pickle

from scipy.interpolate import splprep,splev,BSpline,griddata,bisplev,bisplrep,interp1d,interp2d
from scipy.integrate import trapezoid as trapz, simpson as simps
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

from .__optimizers__ import salttrainingresult,salttrainingoptimizer

import sys
import iminuit,warnings
import logging
log=logging.getLogger(__name__)

from functools import reduce

from saltshaker.util.inpynb import in_ipynb

if in_ipynb:
    from tqdm.notebook import tqdm,trange
else:
    from tqdm import tqdm,trange

usetqdm= sys.stdout.isatty() or in_ipynb

#Which integer code corresponds to which reason for LSMR terminating
stopReasons=['x=0 solution','atol approx. solution','atol+btol approx. solution','ill conditioned','machine precision limit','machine precision limit','machine precision limit','max # of iteration']
        
gnfitresult=namedtuple('gnfitresult',['lsmrresult','postGN','gaussNewtonStep','resids','damping','reductionratio'])
lsmrresult=namedtuple('lsmrresult',['precondstep','stopsignal','itn','normr','normar','norma','conda','normx'] )



class gaussnewton(salttrainingoptimizer):

    configoptionnames=set()
    
    def __init__(self,guess,saltresids,outputdir,options):
        super().__init__(guess,saltresids,outputdir,options)
        for key in self.configoptionnames:
            self.__dict__[key]=options.__dict__[key]
        self.modelobj=saltresids
        self.outputdir=outputdir

        self.debug=False
        self.lsqfit = False
            
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

        if options.fitting_sequence.lower() == 'default' or not options.fitting_sequence:
            self.fitlist = [('all'),
                            ('pcaparams'),
                            ('color'),('colorlaw'),
                            ('spectralrecalibration'),      
                            ('sn')]
        else:
            self.fitlist = [f for f in options.fitting_sequence.split(',')]
            
    @classmethod
    def add_training_options(cls,parser,config):
        
        if parser == None:
            parser = ConfigWithCommandLineOverrideParser(usage=usage, conflict_handler="resolve")
        temp=generateerrortolerantaddmethod(parser)
        def wrapaddingargument(*args,**kwargs):
            cls.configoptionnames.add(args[2])
            return temp(*args,**kwargs)

        successful=True
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
    
    
    def optimize(self,guess,usesns=None):
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
        
        log.info(f'starting loop; {self.gaussnewton_maxiter} iterations')
        chi2results=self.modelobj.getChi2Contributions(X)
        for name,chi2component,dof in chi2results:
            if name.lower()=='photometric':
                photochi2perdof=chi2component/dof
        tryfittinguncertainties=False
        for superloop in range(self.gaussnewton_maxiter):
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


        log.info('Total time spent in convergence loop: {}'.format(datetime.now()-start))
        return X 


        
        
    
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
            iFit=iFit & ~self.ifixedparams
            includePars= np.where(includePars)[0]
        else:
            includePars=np.isin(iFit,np.where(~self.ifixedparams)[0])
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
        

        
    def constructoperator(self,precon,includepars,*args,includeresids=None,**kwargs):
        if includepars.dtype==bool: includepars=np.where(includepars)[0]


        jshape= jax.eval_shape(lambda x: self.modelobj.lsqwrap(x,*args[1:],**kwargs) ,args[0]).shape[0] if includeresids is None else includeresids.size,includepars.size
        @jax.jit
        def preconditioninverse(x):
            return jnp.zeros(self.modelobj.npar).at[includepars].set( precon *x)
        if includeresids is None:
            
            linop=sprslinalg.LinearOperator(matvec = lambda x: (self.modelobj.lsqwrap(*args,**kwargs,diff='jvp',jit=True)( preconditioninverse(x) )) ,

                                 rmatvec= lambda x: (self.modelobj.lsqwrap(*args,**kwargs,diff='vjp',jit=True)(x))[includepars]*precon ,shape=(jshape))
        else:
            blankarray=jnp.zeros(jshape[0])
            
            linop=sprslinalg.LinearOperator(matvec = lambda x: (self.modelobj.lsqwrap(*args,**kwargs,diff='jvp',jit=True)( preconditioninverse(x) )[includeresids ]) ,

                                 rmatvec= lambda x: (self.modelobj.lsqwrap(*args,**kwargs,diff='vjp',jit=True)(blankarray.at[includeresids].set(x)))[includepars]*precon
                                 
                                 ,shape=(jshape))

        return linop,preconditioninverse



    def gaussnewtonfit(self,initval,jacobian,preconinv,residuals,damping,lsqwrapargs, maxiter=None,includeresids=None):

        tol=1e-8
        #import pdb; pdb.set_trace()
        initchi=(residuals**2).sum()
        if maxiter is None: maxiter= self.lsmrmaxiter
        result=lsmrresult(*sprslinalg.lsmr(jacobian,residuals,damp=damping,maxiter=maxiter,atol=tol,btol=tol))
        gaussNewtonStep= preconinv(result.precondstep)
        resids=self.modelobj.lsqwrap(initval-gaussNewtonStep,*lsqwrapargs[0],**lsqwrapargs[1])[includeresids]
        if includeresids is None: pass
        else: resids=resids[includeresids]
        postGN=(resids**2).sum() #
        #if fit == 'all': import pdb; pdb.set_trace()
        log.debug(f'Attempting fit with damping {damping} gave chi2 {postGN} with {result.itn} iterations')
        reductionratio= (initchi -postGN)/(initchi-(result.normr**2))
        # np.unique(self.parlist[np.where(varyingParams != True)])
        return gnfitresult(result,postGN,gaussNewtonStep,resids,damping,reductionratio)
        
    def iteratedampings(self,fit,initval,jacobian,preconinv,residuals,lsqwrapargs,includeresids=None):
        """Experiment with different amounts of damping in the fit"""
        scale=self.dampingscalerate
    
        oldChi=(residuals**2).sum()
        damping=self.damping[fit]
        currdamping=damping
        gnfitfun= lambda dampingval,**kwargs: self.gaussnewtonfit(initval, jacobian,preconinv,residuals,dampingval, lsqwrapargs, **kwargs)
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
                    log.critical(f'Pathological result with chi2 {result.postGN:.3e}, attempting to iterate preconditioning')
                else:
             
                    if result.postGN > 100*oldChi:
                        damping*= 10
                    log.debug('Reiterating and increasing damping')
                    damping*=scale*11/9
                    newresult=gnfitfun(damping/scale)
                    result=min([result,newresult],key=lambda x:x.postGN )

                    if (oldChi>result.postGN): break
                    
            else:
                log.info(f'After increasing damping {maxiter} times, failed to find a result that improved chi2')
        log.debug(f'After iteration on input damping {currdamping:.2e} found best damping was {result.damping:.2e}')
        self.damping[fit]=result.damping
        return result

       
    def process_fit(self,initvals,iFit,uncertainties,fit='all',allowjacupdate=True,excludesn=None,**kwargs):

        X=initvals.copy()
        varyingParams=iFit&self.iModelParam & ~self.ifixedparams
        
        residuals=self.modelobj.lsqwrap(X,uncertainties,**kwargs)
     
        if excludesn:
            snx0params=np.array([parname.endswith(f'x0_{excludesn}') for parname in self.modelobj.parlist])
            indepentofexcluded=(self.lsqwrap(X,uncertainties,**kwargs,diff='jvp')( snx0params*1.)==0)
            includeresids=indepentofexcluded
        else:
            includeresids=np.ones(residuals.size,dtype=bool)
            
        includeresids=np.where(includeresids)[0]
        residuals=residuals[includeresids]
        
        oldChi=(residuals**2).sum()
        
        

        log.info('Number of parameters fit this round: {}'.format(varyingParams.sum()))
        log.info('Initial chi2: {:.2f} '.format(oldChi))
        log.info('Calculating preconditioning')

        preconditioning= self.vectorizedstochasticbinormpreconditioning(varyingParams,X,uncertainties,**kwargs)
        log.info('Finished preconditioning')

        jacobian,preconinv= self.constructoperator(preconditioning, varyingParams, X,uncertainties,includeresids=includeresids **kwargs)
        lsqwrapargs=([uncertainties],kwargs)
        fittingfunction= (lambda *args,**kwargs: self.gaussnewtonfit(*args,0,lsqwrapargs,**kwargs)) if self.damping[fit]==0 else (lambda *args,**kwargs: self.iteratedampings(fit,*args,lsqwrapargs=lsqwrapargs,**kwargs))

        result= fittingfunction(X,jacobian,preconinv,residuals,includeresids=includeresids)
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
                
                jacobian,_= self.constructoperator(preconditioning, varyingParams, X,uncertainties,includeresids=includeresids **kwargs)

                result=fittingfunction(X,jacobian,preconinv,prevresult.resids,includeresids=includeresids)
                
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

