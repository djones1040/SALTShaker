from saltshaker.util.inpynb import in_ipynb
from saltshaker.util.query import query_yes_no
from saltshaker.util.jaxoptions import jaxoptions
from jax import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
#config.update("jax_debug_nans", True)
#config.update("jax_disable_jit", True)

import pickle
from os import path
import os
import sys
import time
import logging

import matplotlib.pyplot as plt
interactive= sys.stdout.isatty()
try: from IPython import display
except: pass

from functools import reduce,partial



import numpy as np
from jax import numpy as jnp, lax, profiler

from saltshaker.config.configparsing import *

from .__optimizers__ import salttrainingresult,salttrainingoptimizer


from scipy.ndimage import gaussian_filter1d
from scipy import signal



log=logging.getLogger(__name__)


class rpropwithbacktracking(salttrainingoptimizer):
    configoptionnames=set()

    def __init__(self,guess,saltresids,outputdir,options):
        super().__init__(guess,saltresids,outputdir,options)
        self.saltobj=saltresids  
        for x in self.configoptionnames:
     	    self.__dict__[x]=getattr(options,x)
        self.outputdir=options.outputdir
        assert(0<self.searchsize<1)
        assert(0<self.searchtolerance<1)
        assert(0<self.etaminus<1)
        assert(self.etaplus>1)
        assert(self.learningratesinitscale>0)
        
        self.minlearning=0
        self.maxlearning=np.inf
        
        self.Xbounds= np.tile(-np.inf,self.saltobj.npar),np.tile(np.inf,self.saltobj.npar)
        self.Xbounds[0][self.saltobj.ispcrcl_coeffs]= -10
        self.Xbounds[1][self.saltobj.ispcrcl_coeffs]= 10
        self.Xbounds[0][self.saltobj.imodelcorr]=-1
        self.Xbounds[1][self.saltobj.imodelcorr]=1
        
        self.functionevals=0
        self.losshistory=[]
        self.Xhistory=[]
        
    @classmethod
    def add_training_options(cls,parser,config):
        """ Specifies to the parser all required configuration options"""
        if parser == None:
            parser = ConfigWithCommandLineOverrideParser(usage=usage, conflict_handler="resolve")
        temp=generateerrortolerantaddmethod(parser)
        def wrapaddingargument(*args,**kwargs):
            cls.configoptionnames.add(args[2])
            return temp(*args,**kwargs)


        successful=wrapaddingargument(config,'rpropconfig','gradientmaxiter',  type=int,
                                                help='Max number of gradient iterations allowed')
        successful=successful&wrapaddingargument(config,'rpropconfig','burninmaxiter',  type=int,default=100,
                                                help='Max number of gradient iterations allowed for the burnin of the flux model')
        successful=successful&wrapaddingargument(config,'rpropconfig','learningratesinitscale',  type=float,
                                                help="""Initial scale for the learning rates""")
        successful=successful&wrapaddingargument(config,'rpropconfig','searchsize',  type=float,
                                                help="""Size of steps to take in backtracking line-search (must be between 0 and 1)""")
        successful=successful&wrapaddingargument(config,'rpropconfig','searchtolerance',  type=float,
                                                help="""Size of  Armijo's criterion, smaller indicates looser constraints (must be between 0 and 1)""")
        successful=successful&wrapaddingargument(config,'rpropconfig','etaminus',  type=float,
                                                help="""Amount by which to decrease learning rates when applicable (must be between 0 and 1)""")
        successful=successful&wrapaddingargument(config,'rpropconfig','etaplus',  type=float,
                                                help="""Amount by which to increase learning rates when applicable (must be greater than 1)""")
        successful=successful&wrapaddingargument(config,'rpropconfig','convergencetolerance',  type=float,
                                                help=""" Convergence criterion, will abort when change in loss is consistently less than this (must be greater than 0)""")
        successful=successful&wrapaddingargument(config,'rpropconfig','memorydebug',  type=boolean_string, default=False,
                                                help=""" This is a flag to enable memory profiling""")

        if not successful: sys.exit(1)

        return parser




    def optimize(self,initvals):
        X=initvals.copy() #self.saltobj.constraints.transformtoconstrainedparams(jnp.array(initvals))

        residuals=self.saltobj.lsqwrap(X,self.saltobj.calculatecachedvals(X,'variances'),jit=False,dospecresids=self.saltobj.dospec)
        oldChi=(residuals**2).sum()
        log.info('Initial chi2: {:.2f} '.format(oldChi))

        rates=self.initializelearningrates(X)
        if self.memorydebug:
           profiler.save_device_memory_profile(path.join(self.outputdir, 'memory_init.profile'))
        try:
            #First fit with color scatter fixed
            if not np.isinf(X[self.saltobj.iclscat_0]):
                X[self.saltobj.iclscat_0]=-np.inf
            
            log.info('Fitting sn parameters')
            from functools import reduce
            snpars=reduce(lambda x,y:  x | np.isin(np.arange(self.saltobj.npar), y),[self.saltobj.icoordinates,self.saltobj.ix0,self.saltobj.ic],False)

            X,loss,rates=self.optimizeparams(X,snpars,rates,niter=self.burninmaxiter,usesecondary=False)
            rates=self.initializelearningrates(X)
            log.info('Burning in flux model')
            fitparams=self.saltobj.iModelParam
            X,loss,rates=self.optimizeparams(X,fitparams,rates,niter=self.burninmaxiter,usesecondary=False) #self.gradientmaxiter)
            log.info('Fitting all parameters')
            fitparams=~np.isin(np.arange(self.saltobj.npar),self.saltobj.iclscat)
            X,loss,rates=self.optimizeparams(X,fitparams,rates,niter=self.gradientmaxiter,usesecondary=len(self.saltobj.constraints.use_secondary_constraint_names)>0)
                        
            #Fit error model including color scatter
            log.info('Fitting color scatter')
            X=X.at[self.saltobj.iclscat[-1]].set(-4)
            
            fitparams=~self.saltobj.iModelParam
            fitparams=fitparams | np.isin(np.arange(self.saltobj.npar),self.saltobj.ic) | np.isin(np.arange(self.saltobj.npar),self.saltobj.iCL) 
            X,loss,rates=self.optimizeparams(X,fitparams,rates,niter=self.gradientmaxiter,usesecondary=len(self.saltobj.constraints.use_secondary_constraint_names)>0)
            
        except KeyboardInterrupt as e:
            if query_yes_no("Terminate optimization loop and begin writing output?"):
                X=self.Xhistory[np.argmin(self.losshistory)]
            else:
                if query_yes_no("Enter pdb?"):
                    import pdb;pdb.set_trace()
                else:
                    raise e
        except Exception as e:
            log.exception('Error encountered in optimization, exiting')
            raise e
        residuals=self.saltobj.lsqwrap(X,self.saltobj.calculatecachedvals(X,'variances'),jit=False,dospecresids=self.saltobj.dospec)
        newChi=(residuals**2).sum()

        log.info('Final chi2: {:.2f} '.format(newChi))
        
        chi2results=self.saltobj.getChi2Contributions(X,jit=False,dospecresids=self.saltobj.dospec)
        
        for name,chi2component,dof in chi2results:
            log.info('{} chi2/dof is {:.1f} ({:.2f}% of total chi2)'.format(name,chi2component/dof,chi2component/sum([x[1] for x in chi2results])*100))

        return X

    
    def initializelearningrates(self,X):
        """
        Determines starting learning rates for the gradient descent algorithm
        
        Parameters
        ----------
        X: array-like of size N
            Initial parameter vector
        ----------
        Returns
        rates: array-like of size N
            Initial choice for learning rates
        """
        learningrates=np.tile(np.nan,self.saltobj.npar)
        #Parameters I expect to be more or less normal distribution, where the std gives a reasonable size to start learning
        for idx in list(self.saltobj.icomponents):
            if np.std(X[idx])==0: learningrates[idx]=(np.std( X[self.saltobj.icomponents])*1e-2)
            else: learningrates[idx]=(np.std( X[idx])*.1)
            
            
        for idx in list(self.saltobj.icoordinates)+[ self.saltobj.ic
                ]:
            learningrates[idx]=(np.std( X[idx])*.1)
            
                
            
        learningrates[self.saltobj.imodelerr]=np.std( X[self.saltobj.imodelerr])*.1
        #x0 should very fractionally from the initialization
        x0=X[self.saltobj.ix0]
        for idx in self.saltobj.ix0:
            try:
                learningrates[idx]= .1* max(X[idx],x0[np.nonzero(x0)].min())
            except:
                learningrates[idx]= 1e-3
        #The rest of the parameters are mostly dimensionless coeffs of O(1)
        for idx in [self.saltobj.iCL,self.saltobj.ispcrcl_coeffs,self.saltobj.iclscat,self.saltobj.imodelcorr]:
            learningrates[idx]=1e-2
        #Check that all parameters get an initial guess
        try:
            assert(~np.any(np.isnan(learningrates)) and ~ np.any(learningrates<=0))
        except Exception as e:
            log.debug(f'Uninitialized learning rates: {", ".join(np.unique(self.saltobj.parlist[np.isnan(learningrates) |(learningrates<=0) ]))}')
            learningrates[np.isnan(learningrates) |(learningrates<=0) ]=1e-5
        return learningrates*self.learningratesinitscale
                
        
    def lossfunction(self,params,*args,excludesn=None,**kwargs):
        """ 
        Passthrough function to maxlikefit, with an excludeSN keyword that removes the log-likelihood of a single SN.
        """
        if 'diff' in kwargs and "grad" in kwargs['diff']: self.functionevals +=3
        else: self.functionevals+=1

        result= self.saltobj.constrainedmaxlikefit(params,*args,**kwargs)

        if excludesn: 
            singleresult=self.saltobj.datadict[excludesn].modelloglikelihood(params,*args,**kwargs)
            try: 
                result=( x-y for x,y in zip(result,singleresult))
            except:
                result=result-singleresult

        try:
            return (-x for x in result)
        except:
            return -result
            
    
    def optimizeparams(self,initvals,iFit,initrates,niter=100,debug=False,**kwargs):
        """
        Iterate parameters by gradient descent algorithm until convergence or a maximum number of iterations is reached
        
        Parameters
        ----------
        initvals : array-like of size N
            Initial values of model parameters
        iFit : array of integers or booleans of size N
            Index array indicating which parameters are to be fit
        initrates : array-like of size N
            Initial learning rates for each parameter
        niter : integer
            Maximum number of iterations allowed before automatic termination
        Returns
        -------
        X : array-like of size N
            Optimized parameter vector 
        loss: float
            Loss at local minimum
        rates: array-like of size N
            Final learning rates for each parameter as determined by the algorithm
        """

        if ( iFit.dtype == int):
            iFit=np.isin( np.arange(X.size),iFit )
        assert((iFit.dtype==bool))
        iFit=iFit & ~self.ifixedparams

        log.info('Number of parameters fit this round: {}'.format(iFit.sum()))
        startlen=len(self.Xhistory)
        rates=initrates*iFit
        #Set convergence criteria
        #Low pass filter to ensure that 
#         convergencefilt=signal.butter(4,2e-2,output='sos')
        numconvergence=10
            
        starttime=time.time()
        initvals=jnp.array(initvals)
        
        X, Xprev,loss,sign= initvals,initvals, np.inf,np.zeros(initvals.size)
        rates=jnp.array(rates)

        def iteration(X, Xprev,loss,sign,rates):
            #Proposes a new value based on sign of gradient
            Xnew,newloss, newsign, newgrad,newrates  = self.rpropiter(X, Xprev,loss,sign,rates,**kwargs)

            #Take the direction proposed and do a line-search in that direction
            searchdir=jnp.select([~jnp.isinf(X),jnp.isinf(X)], [ Xnew-X, 0])
            if newgrad @ searchdir < 0:
                gamma=self.twowaybacktracking(X,newloss,newgrad,searchdir,**kwargs)
                newrates*=gamma
            else:
                #If the line is opposite the gradient, check that it actually improves the result
                unimproved= self.lossfunction(Xnew,**kwargs) > newloss
                if unimproved :
                    backwards=newgrad*searchdir > 0
                    gamma=self.twowaybacktracking(X,newloss,newgrad, searchdir.at[backwards].set(0 ),**kwargs)
                    newrates=newrates.at[~backwards].set(newrates[~backwards]*gamma)
                else:
                    gamma=1

            Xnew= X+ gamma*searchdir

            return Xnew, X, newloss,newsign, newgrad,newrates
            
        for i in range(niter):
            if i%20 == 19:
                trials=[( reinitialized,*iteration(X, Xprev,loss,sign, rateguess)) for reinitialized,rateguess in [(False,rates), (True,iFit*self.initializelearningrates(X)*1e-2)]]
                log.debug('Experimenting with reinitialization')                
                reinitialized,X,Xprev,loss,sign,grad,rates = min(trials,key=lambda x: self.lossfunction(x[1],**kwargs) )
                if reinitialized: log.debug('Reinitialized learning rates')
            else:
                X,Xprev,loss,sign,grad,rates = iteration(X, Xprev,loss,sign,rates)
                
            constrainedparams=  np.concatenate([self.saltobj.ic,self.saltobj.icoordinates])
            if not np.allclose(X[constrainedparams],Xprev[constrainedparams]):
                X=self.saltobj.constraints.transformtoconstrainedparams(X)
            self.losshistory+=[loss]
            self.Xhistory+=[X]
            
            if i==0:
                log.debug(f'First iteration took {time.time()-starttime:1f} seconds')
            else:
                if len(self.losshistory)> numconvergence+10:
                    convergencecriterion= np.abs(self.losshistory[-numconvergence] - loss)
                    if np.isnan(loss) or np.all( np.array(self.losshistory[-numconvergence*2+1:]) > self.losshistory[-numconvergence*2] ):
                    
                        convergencecriterion=0
                else:
                    convergencecriterion=np.inf
                if (i%100 == 99) and self.memorydebug:
                   profiler.save_device_memory_profile(path.join(self.outputdir, f'memory_eval{self.functionevals}.profile'))
                
                #signal.sosfilt(convergencefilt, -np.diff((self.losshistory[-numconvergence:]) ))[-1]
                outtext=f'Iteration {i} , function evaluations {self.functionevals}, convergence criterion {convergencecriterion:.2g}, last diff {self.losshistory[-2]-loss:.2g}, gradient magnitude {jnp.dot(grad[np.nonzero(rates)],grad[np.nonzero(rates)]):.2g}, rates magnitude {jnp.dot(rates,rates):.2g}'
                if interactive:
                    sys.stdout.write(f'\r\x1b[1K'+outtext.ljust(os.get_terminal_size().columns))
                log.debug(outtext)
                
                if i> numconvergence+10:
                    if np.all(convergencecriterion< self.convergencetolerance):
                        sys.stdout.write('\n')
                        log.info('Convergence achieved')
                        break

            if in_ipynb:
                if i==0: continue
                plt.clf()
                diffs= np.diff(self.losshistory[-100:])
                plt.plot( np.arange(diffs.size) + i-diffs.size,diffs,'r-')
                
                plt.plot( np.arange(diffs.size) + i-diffs.size,-diffs,'b-')
                plt.xlabel('Iteration')
                plt.ylabel('Change in Loss')
                plt.yscale('log')
                #if i>0: plt.text(0.7,.8,f'Last diff: {-np.diff(self.losshistory[-2:])[0]:.2g}',transform=plt.gca().transAxes)
                display.display(plt.gcf())
                display.clear_output(wait=True)
        else: 
            sys.stdout.write('\n')
            log.info('Optimizer encountered iteration limit')
        final= startlen+np.argmin( self.losshistory[startlen:])
        X,loss=self.Xhistory[final],self.losshistory[final]
        with open(path.join(self.outputdir,'gradienthistory.pickle'),'wb') as file:
            pickle.dump((self.Xhistory,self.losshistory),file)
        return self.saltobj.constraints.transformtoconstrainedparams(X),loss,rates.at[rates==0].set(initrates[rates==0])
        

    def twowaybacktracking(self,X,loss,grad, searchdir,*args,**kwargs):
    
        """
        Based on a proposed parameter update, determines whether Armijo's criterion is satisfied; if so attempts to increase learning rate. If false, decreases learning rate until satisfied
    
        Armijo's criterion checks whether the gradient accurately models the proposed update; when satisfied indicates efficient convergence
        
        Parameters
        ----------
        X : array-like
            Initial values of model parameters
        loss : 
            Loss of model at initial values
        grad :
            Gradient at initial values
        searchdir :
            Proposed direction for an update to the parameters. Dot product with `grad` must be positive.
        Returns
        -------
        gamma : float
            Scalar multiplier for parameter and learning rate update 
        
        References
        https://doi.org/10.1007/s00245-020-09718-8

            """
        t= - self.searchtolerance * grad @ searchdir
        prevgamma=1/self.searchsize
        gamma=1
        if t<0:
            raise ValueError('Bad search direction provided')            
            
        Xprop= X + gamma * searchdir 
        proploss=self.lossfunction(Xprop, *args,**kwargs)

        log.debug(f'Entering two way backtracking with initial loss {loss:.2g}, termination criterion {t:.2g}, and initial diff {loss-proploss:.2g}')
        
        if loss-proploss >= gamma*t:
            for i in range(5):
                if loss-proploss >= gamma*t:
                    prevgamma=gamma
                    gamma=gamma/self.searchsize 
                    Xprop= X + gamma * searchdir 
                    proploss=self.lossfunction(Xprop,*args,**kwargs)
                else:
                    break
            log.debug(f'final gamma factor {gamma:.2g}')
            return prevgamma
        else:
            for i in range(10):
                if (loss-proploss < gamma*t or  np.isnan(loss-proploss)) and ( loss-proploss != 0):
                    prevgamma=gamma
                    gamma=gamma*self.searchsize
                    Xprop= X + gamma * searchdir 
                    proploss=self.lossfunction(Xprop,*args,**kwargs)
                else:
                    break
            log.debug(f'final gamma factor {prevgamma:.2g}')
            return gamma

    
        
        
    def rpropiter(self,X, Xprev, prevloss,prevsign, learningrates,*args,**kwargs):
    
        """
        Implementation of the iRProp+ with weight backtracking algorithm. Based on the sign of the gradient chooses in which direction to update the parameters as well as increasing or decreasing learning rate on a parameter by parameter basis. Parameters with gradients in a single direction will be consistently increased, while those with changing gradients (indicating jumps over minima) will be decreased and/or updates reverted.
        
        Parameters
        ----------
        
        X : array-like
            Initial values of parameters
        Xprev: array-like
            Previous values of parameters
        prevloss : float
            Loss of model at `Xprev`
        prevsign : array-like
            Sign of gradient of loss at `Xprev` (-1,0,1)
        learningrates : array-like
            Semipositive learning rates for each parameter
            
        Returns
        --------
        
        Xnew : array-like
            Suggested parameter vector
        loss: float
            Loss of model at `X`
        sign: array-like
            Sign of gradient of loss at `X` (-1,0,1)
        grad: array-like
            Gradient of loss at `X` 
        learningrates : array-like
            Semipositive learning rates for each parameter
        
        References
        https://doi.org/10.1016/S0925-2312(01)00700-7
        """
        lossval,grad=  self.lossfunction(X,*args,**kwargs, diff='valueandgrad')
        
        # if gradient is NaN, jax had some trouble...
        sign=jnp.nan_to_num(jnp.sign(grad))
        indicatorvector= prevsign *sign

        greater= indicatorvector >0
        less=   indicatorvector <0
        eq=     indicatorvector==0
        greatereq = (eq) | (greater)

        learningrates = jnp.clip (jnp.select([less,eq,greater], [learningrates*self.etaminus,learningrates,learningrates*self.etaplus ] ), self.minlearning,self.maxlearning)

        Xnew=jnp.select( [less,greatereq], [
            lax.cond(lossval>prevloss, lambda x,y:x , lambda x,y: y, Xprev, X), 
            X-(sign *learningrates)
        ])
        
        #Set sign to 0 after a previous change
        sign= (sign * greatereq)
        return jnp.clip(Xnew,*self.Xbounds), lossval, sign, grad, learningrates
