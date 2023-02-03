import numpy as np
from jax import numpy as jnp
from jax import lax

import logging
log=logging.getLogger(__name__)


class rpropwithbacktracking:
    def __init__(self,saltobj,etaminus=.5,etaplus=1.2, searchtolerance=np.exp(-.5),searchsize=np.exp(-.5)):
        self.saltobj=saltobj        
        self.etaplus=etaplus
        self.etaminus=etaminus
        self.minlearning=0
        self.maxlearning=np.inf
        self.Xbounds= np.tile(-np.inf, saltobj.npar),np.tile(np.inf, saltobj.npar)
        self.Xbounds[0][saltobj.ispcrcl_coeffs]= -10
        self.Xbounds[1][saltobj.ispcrcl_coeffs]= 10
        self.searchtolerance= searchtolerance
        self.searchsize = searchsize
        self.functionevals=0
        self.losshistory=[]
        self.Xhistory=[]
        
    def process_fit(self,initvals,iFit,learningrates=None,niter=100,**kwargs):
        X=initvals.copy()
        log.info('Entering process_fit')
        if ( iFit.dtype == int):
            iFit=np.isin( np.arange(X.size),iFit )
        assert((iFit.dtype==bool))
        
        if learningrates is None:
    
            learningrates=np.tile(np.nan,X.size)
            for idx in [self.saltobj.im0,self.saltobj.im1,self.saltobj.imhost, self.saltobj.ix1,self.saltobj.ic,self.saltobj.ixhost
                    ,self.saltobj.imodelerr]:
                learningrates[idx]=np.std( X[idx])*.1
            for idx in [self.saltobj.ix0]:
                learningrates[idx]= .1* X[idx] 
            for idx in [self.saltobj.iCL,self.saltobj.ispcrcl_coeffs,self.saltobj.iclscat,self.saltobj.imodelcorr]:
                learningrates[idx]=1e-2
            
        rates= learningrates.copy()
#         import pdb;pdb.set_trace()
        assert(not np.isnan(learningrates).any() )
        
        rates=rates*iFit
        
        residuals=self.saltobj.lsqwrap(X,self.saltobj.calculatecachedvals(X,'variances'),**kwargs)
        oldChi=(residuals**2).sum()
        
        log.info('Number of parameters fit this round: {}'.format(iFit.sum()))
        log.info('Initial chi2: {:.2f} '.format(oldChi))
                
        X, Xprev,loss,sign= X,X, np.inf,np.zeros(X.size)
        for i in range(niter):
            try:
                Xnew,newloss, newsign, newgrad,newrates  = self.rpropiter(X, Xprev,loss,sign,rates)
        
                searchdir=np.select([~np.isinf(X),np.isinf(X)], [ Xnew-X, 0])
                gamma=self.twowaybacktracking(X,newloss,newgrad,searchdir)
                
                newrates*=gamma
                Xnew= X+ gamma*searchdir
                if newloss==loss:
                    log.info('Convergence achieved')
                    break
                loss,sign,grad,rates=newloss,newsign, newgrad,newrates
                Xprev=X
                X=Xnew
                
                self.losshistory+=[loss]
                self.Xhistory+=[X]
            except KeyboardInterrupt:
                if query_yes_no("Terminate optimization loop and begin writing output?"):
                    break
                else:
                    if query_yes_no("Enter pdb?"):
                        import pdb;pdb.set_trace()
                    else:
                        raise e
            except Exception as e:
                log.exception('Error encountered in convergence_loop, exiting')
                raise e
        residuals=self.saltobj.lsqwrap(X,self.saltobj.calculatecachedvals(X,'variances'),**kwargs)
        newChi=(residuals**2).sum()
        log.info('Final chi2: {:.2f} '.format(newChi))

        return X,loss,rates
        

    def twowaybacktracking(self,X,loss,grad, searchdir,debug=False,*args,**kwargs):
        t= - self.searchtolerance * grad @ searchdir
        prevgamma=1/self.searchsize
        gamma=1
        if debug: log.debug('t: ',t)
        Xprop= X + gamma * searchdir 
        proploss=-self.saltobj.maxlikefit(Xprop,*args,**kwargs)
        self.functionevals+=1
        if loss-proploss >= gamma*t:
            for i in range(5):
                if loss-proploss >= gamma*t:
                    if debug: log.debug(gamma, gamma*t, loss-proploss)
                    prevgamma=gamma
                    gamma=gamma/self.searchsize 
                    Xprop= X + gamma * searchdir 
                    proploss=-self.saltobj.maxlikefit(Xprop,*args,**kwargs)
                    self.functionevals+=1
                else:
                    break
        else:
            while (loss-proploss < gamma*t or  np.isnan(loss-proploss)) and ( loss-proploss != 0):
                if debug: log.debug(gamma,gamma*t, loss-proploss)
                prevgamma=gamma
                gamma=gamma*self.searchsize
                Xprop= X + gamma * searchdir 
                proploss=-self.saltobj.maxlikefit(Xprop,*args,**kwargs)
                self.functionevals+=1
    
        if debug: log.debug(gamma, loss-proploss)
        if debug: log.debug('Final',prevgamma,prevgamma*t, loss- ( -self.saltobj.maxlikefit(X + prevgamma * searchdir ,*args,**kwargs)))
        return prevgamma
    
        
        
    def rpropiter(self,X, Xprev, prevloss,prevsign, learningrates,*args,**kwargs):
        lossval,grad=  self.saltobj.maxlikefit(X,*args,**kwargs, diff='valueandgrad')
        self.functionevals+=3
        lossval=-lossval
        grad=-grad
        sign=jnp.sign(grad)
        indicatorvector= prevsign *sign

        greater= indicatorvector >0
        less=   indicatorvector <0
        eq=     indicatorvector==0
        greatereq = (eq) | (greater)

        learningrates = jnp.clip (jnp.select([less,eq,greater], [learningrates*self.etaminus,learningrates,learningrates*self.etaplus ] ), self.minlearning,self.maxlearning)

        Xnew=jnp.select( [less,greatereq], [ lax.cond(lossval>prevloss, lambda x,y:x , lambda x,y: y, Xprev, X), 
            X-(sign *learningrates)
        ])
        sign= (sign * greatereq)
        return jnp.clip(Xnew,*self.Xbounds), lossval, sign, grad, learningrates
