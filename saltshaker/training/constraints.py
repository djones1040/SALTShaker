
import numpy as np

from jax import numpy as jnp
import jax
from jax import lax
from jax.experimental import sparse 
from saltshaker.util.jaxoptions import jaxoptions

from scipy.interpolate import splprep,splev,bisplev,bisplrep,interp1d,interp2d,RegularGridInterpolator,RectBivariateSpline
from scipy import stats

from functools import partial,reduce
from inspect import signature

__possibleconstraints__=dict()
import logging
log=logging.getLogger(__name__)


def constraint(fun):
    """Decorator to register a given function as a valid prior"""
    #Check that the method accepts 2 inputs
    assert(len(signature(fun).parameters)==2)
    __possibleconstraints__[fun.__name__]=fun
    return fun


class SALTconstraints:

    def __init__(self,residsobj):
        for k in residsobj.__dict__.keys():
            self.__dict__[k] = residsobj.__dict__[k]
        self.saltresids=residsobj

        self.constraints={ key: partial(__possibleconstraints__[key],self) for key in __possibleconstraints__}
        if self.constraint_names[0] == '':
            self.use_constraint_names = []
        else:
            self.use_constraint_names= [x.strip() for x in self.constraint_names]
        self.use_secondary_constraint_names= [x.strip() for x in self.secondary_constraint_names if len(x.strip())>0]
        
        intmult = (self.wave[1]-self.wave[0])*self.fluxfactor['default']['B']
        fluxDeriv= np.zeros(self.im0.size)
        for i in range(self.im0.size):
            derivInterp = bisplev(np.array([0]),self.wave,(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder))
            fluxDeriv[i] = np.sum( self.kcordict['default']['Bpbspl'] * derivInterp)*intmult 
        
        self.__maximumlightpcderiv__=sparse.BCOO.fromdense(fluxDeriv)

    @partial(jaxoptions,static_argnums=[0,2],static_argnames=['usesecondary'],jitdefault=True)
    def transformtoconstrainedparams(self,guess,usesecondary=True):
        return reduce( lambda value,name: self.constraints[name](value), self.use_secondary_constraint_names + self.use_constraint_names if usesecondary else self.use_constraint_names ,  guess)

    @constraint
    def centeranddecorrelatedcolorsandcoords(self,guess):
        idxs=np.concatenate([self.ic[:1,:],self.icoordinates])
        coordinates=guess[idxs]
        from jax.scipy import linalg as jlin
        from functools import reduce
        
        #decorrelate and center the distribution of the coordinates
        def choldecorrelate(data):
            coordinates=data
            coordinates=coordinates-jnp.mean(coordinates,axis=1)[:,np.newaxis]
            chol=jlin.cholesky(jnp.cov(coordinates),lower=True)
            return jlin.solve_triangular(chol, coordinates, lower=True)

        decorrelated=reduce(lambda x,i: choldecorrelate(x),np.arange(3),coordinates)
        for i,idx,corrected in zip(range(decorrelated.shape[0]),idxs,decorrelated):
            if i >= self.ic[:1,:].shape[0]:
                guess=guess.at[idx].set(corrected)
            else:
                guess=guess.at[idx].set(corrected *jnp.std(guess[idx]))

        return guess
    
    @constraint
    def fixbluecolorscatter(self,guess):
        """ Set the value of the color scatter at the minimum wavelength covered by the model to 1"""
        clscatpars=guess[self.iclscat]
        
        firstterms=self.saltresids.colorscatter(guess.at[self.iclscat[0]].set(0),self.wave.min())
        lastterm=self.saltresids.colorscatter(guess.at[self.iclscat].set(jnp.zeros(clscatpars.size).at[0].set(1)),self.wave.min())
        return jax.lax.cond(firstterms==0,
            lambda : guess,
            lambda : guess.at[self.iclscat[0]].set(-jnp.log(firstterms)/jnp.log(lastterm))
        )
        
    @constraint
    def fixbbandfluxes(self,guess):
        #set M0 flux to fiducial standard, and the other components to 0 B-band flux

        bstdflux=(10**((self.m0guess-27.5)/-2.5) )
        bflux= self.__maximumlightpcderiv__ @ guess[self.im0]
        guess=guess.at[self.icomponents].set( guess[self.icomponents] *bstdflux/bflux) #*  -self.bstdflux
        guess=guess.at[self.ix0].set(guess[self.ix0]*bflux/bstdflux)
     
        
        for i,comp in enumerate(self.icomponents[1:]):
            bflux=self.__maximumlightpcderiv__ @ guess[comp]
            ratio=bflux/bstdflux
            indices=np.array([self.datadict[self.parlist[x0ind].split('_')[1]].icoordinates[i] for x0ind in self.ix0])
            guess=guess.at[self.ix0].set(guess[self.ix0]*(1+ratio*guess[indices]))
            guess=guess.at[self.icoordinates].set( guess[self.icoordinates]/(1+ratio*guess[self.icoordinates[i][np.newaxis,:]]))
            guess=guess.at[comp].set(guess[comp]-  ratio * guess[self.im0])
        return guess
    
    @constraint
    def fixinitialflux(self,guess):
        return guess.at[self.icomponents[:,:(self.waveknotloc.size-self.bsorder) ]].set(0)
        
    @constraint
    def fixinitialderivative(self,guess):
        numwavepars=(self.waveknotloc.size-self.bsorder)
        return guess.at[self.icomponents[:,numwavepars:2*numwavepars ]].set(0)
    
    
    def enforcefinaldefinitions(self,X,components,checkerrors=True):
        X=np.array(X)
        if checkerrors:
            try:
                Xredefined=self.enforcefinaldefinitions(X,self.saltresids.SALTModel(X),False)
                log.info('Checking that rescaling components to satisfy definitions did not modify photometry')
                unscaledresults={}
                scaledresults={}

                Xtmp,Xredefinedtmp = X.copy(),Xredefined.copy()
                if self.no_transformed_err_check:
                    log.warning('parameter no_transformed_err_check set to True.  Use this option with bootstrap errors *only*')
                    Xtmp[self.imodelerr0] = 0
                    Xredefinedtmp[self.imodelerr0] = 0
                    Xtmp[self.imodelerr1] = 0
                    Xredefinedtmp[self.imodelerr1] = 0
                    Xtmp[self.imodelerrhost] = 0
                    Xredefinedtmp[self.imodelerrhost] = 0

                assert(np.allclose(self.saltresids.batchedphotlikelihood(Xredefined),
                self.saltresids.batchedphotlikelihood(X),rtol=1e-4,atol=1e-4))
                
                Xfinal= Xredefined.copy()
            except NotImplementedError:
                log.critical("Final definitions and rescaling has not been implemented for this model configuration")
                Xfinal=X.copy()
            except AssertionError:
                log.critical('Rescaling components failed; photometric residuals have changed. Will finish writing output regardless')
                Xfinal=Xredefined.copy()
            return Xfinal
        else:
            if self.n_components==2:
                return self.onedfinaldefinitions(X,components)
            elif self.n_components==3:
                return self.twodfinaldefinitions(X,components)
            else:
                raise NotImplementedError()
            
    def onedfinaldefinitions(self,X,components):
        """Ensures that the definitions of M1,M0,x0,x1 are satisfied"""
        X=X.copy()
        int1d = interp1d(self.phase,components[0],axis=0,assume_sorted=True)
        m0Bflux = np.sum(self.kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
            (self.wave[1]-self.wave[0])*self.fluxfactor['default']['B']

        int1d = interp1d(self.phase,components[1],axis=0,assume_sorted=True)
        m1Bflux = np.sum(self.kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
            (self.wave[1]-self.wave[0])*self.fluxfactor['default']['B']
        ratio=m1Bflux/m0Bflux
        #Define M1 to have no effect on B band at t=0

        if not self.host_component:
            # re-scaling M1 isn't going to work in the host component case
            for sn in self.datadict:
                ix0=np.array([(x==f'x0_{sn}' ) or (x.startswith(f'specx0_{sn}_')) for x in self.parlist])
                X[ix0]*=(1+ratio*X[f'x1_{sn}'==self.parlist])
            X[self.ix1]/=1+ratio*X[self.ix1]
            X[self.im1]-=ratio*X[self.im0]
        else:
            # we need to modify x1 and M_host to remove correlations
            # this causes problems for the errors, but can be used in conjunction with bootstrapping
            alpha = ((X[self.ix1]*X[self.ixhost]).sum() / (X[self.ixhost]**2).sum())
            X[self.ix1] = X[self.ix1] - alpha*X[self.ixhost]
            X[self.imhost] = X[self.imhost] + alpha*X[self.im1]


        ####This code will not work if the model uncertainties are not 0th order (simple interpolation)
        if self.errbsorder==0:
            if self.n_errorsurfaces>1:
                m0variance=X[self.imodelerr0]**2
                m0m1covariance=X[self.imodelerr1]*X[self.imodelerr0]*X[self.imodelcorr01]
                m1variance=X[self.imodelerr1]**2
            
                if not self.host_component:
                    # re-scaling M1 isn't going to work in the host component case
                    m1variance+=-2*ratio*m0m1covariance+ratio**2*m0variance
                    m0m1covariance-=m0variance*ratio
                else:
                    mhostvariance=X[self.imodelerrhost]**2
                    m0mhostcovariance=X[self.imodelerrhost]*X[self.imodelerr0]*X[self.imodelcorr0host]
        else:
            log.critical('RESCALING ERROR TO SATISFY DEFINITIONS HAS NOT BEEN IMPLEMENTED')

        #Define x1 to have mean 0
        #m0 at peak is not modified, since m1B at peak is defined as 0
        #Thus does not need to be recalculated for the last definition
        meanx1=np.mean(X[self.ix1])
        X[self.im0]+= meanx1*X[self.im1]
        X[self.ix1]-=meanx1
        if (self.errbsorder==0 )and (self.n_errorsurfaces>1):
            m0variance+=2*meanx1*m0m1covariance+meanx1**2*m1variance
            m0m1covariance+=m1variance*meanx1
        else:
            log.critical('RESCALING ERROR TO SATISFY DEFINITIONS HAS NOT BEEN IMPLEMENTED')

        
        #Define x1 to have std deviation 1
        x1std = np.std(X[self.ix1])
        if x1std == x1std and x1std != 0.0:
            X[self.im1]*= x1std
            X[self.ix1]/= x1std
        if (self.errbsorder==0) and (self.n_errorsurfaces>1):
            m1variance*=x1std**2
            m0m1covariance*=x1std
        else:
            log.critical('RESCALING ERROR TO SATISFY DEFINITIONS HAS NOT BEEN IMPLEMENTED')

        #Define m0 to have a standard B-band magnitude at peak
        self.bstdflux=(10**((self.m0guess-27.5)/-2.5) )
        fluxratio=self.bstdflux/m0Bflux
        X[self.im0]*=fluxratio
        X[self.im1]*= fluxratio
        if self.host_component: X[self.imhost]*= fluxratio
        X[self.ix0]/=fluxratio
        if (self.errbsorder==0) and (self.n_errorsurfaces>1):
            m1variance*=fluxratio**2
            m0variance*=fluxratio**2
            m0m1covariance*=fluxratio**2
            if self.host_component:
                mhostvariance*=fluxratio**2
                m0mhostcovariance*=fluxratio**2
        else:
            log.critical('RESCALING ERROR TO SATISFY DEFINITIONS HAS NOT BEEN IMPLEMENTED')


        if (self.errbsorder==0) and (self.n_errorsurfaces>1):
            X[self.imodelerr0]= np.sqrt(m0variance)
            X[self.imodelcorr01]= m0m1covariance/np.sqrt(m0variance*m1variance)
            X[self.imodelerr1]=np.sqrt(m1variance)
            if self.host_component:
                X[self.imodelerrhost]=np.sqrt(mhostvariance)
                X[self.imodelcorr0host]= m0mhostcovariance/np.sqrt(m0variance*mhostvariance)

        return X
        
        
    def twodfinaldefinitions(self,guess,components):
        if self.host_component: raise NotImplementedError()
        if self.n_errorsurfaces not in [1,3]: raise NotImplementedError()
       # guess=self.fixbbandfluxes(jnp.array(guess))
        #guess=self.transformtoconstrainedparams(guess)
        guess=np.array(guess)
        #KDE estimator for mutual entropy in 2D
        def mutualinformation(theta,coords):
            rot=np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            coords=rot @ coords
            x1,x2=coords
            return np.mean(stats.gaussian_kde(coords).logpdf(coords)
                           -(stats.gaussian_kde(x1).logpdf(x1)+stats.gaussian_kde(x2).logpdf(x2)))
        #Find the rotation angle that minimizes mutual information
        from scipy.optimize import minimize_scalar
        theta=minimize_scalar(lambda theta: mutualinformation(theta,guess[self.icoordinates]) ,bounds=[-np.pi/4,np.pi/4],method='bounded').x
        #Rotation matrix
        rot=np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        x1,x2= rot @ guess[self.icoordinates]
        
        #Assign the most skewed parameter as x1
        #statistic= lambda x: stats.normaltest(np.clip(x,*np.percentile(x,[2,98]))).statistic
        
        #Assign the component with largest mean squared flux as M1
        m1,m2= rot @ guess[self.icomponents[1:]]
        if  np.mean(m1**2)< np.mean(m2**2):
            rot=rot[::-1]
            x2,x1=x1,x2
        
        m1,m2= rot @ guess[self.icomponents[1:]]
        #x1 should be positive with the flux variation in B band at 15 days
        m1deltaBat15=(np.sum(self.kcordict['default']['Bpbspl'] * bisplev(np.array([15]),self.wave,(self.phaseknotloc,self.waveknotloc,m1,self.bsorder,self.bsorder))))
        #x2 will be defined with positive flux variation in B band at 15 days
        m2deltaBatearly=(np.sum(self.kcordict['default']['Bpbspl'] * bisplev(np.array([15]),self.wave,(self.phaseknotloc,self.waveknotloc,m2,self.bsorder,self.bsorder))))

        rot = np.diag([np.sign(m1deltaBat15),np.sign((m2deltaBatearly))]) @ rot 

        #apply the rotation to the parameters
        guess[self.icomponents[1:]]= rot @ guess[self.icomponents[1:]]
        guess[self.icoordinates]=rot @ guess[self.icoordinates]
        if self.n_errorsurfaces==3:
            m1var=guess[self.parlist == 'modelerr_1']**2
            m2var=guess[self.parlist == 'modelerr_2']**2
            m1m2covar=guess[self.parlist == 'modelerr_1']*guess[self.parlist == 'modelerr_2']* guess[self.parlist == 'modelcorr_12']
            varmat=np.array([[m1var,m1m2covar],[m1m2covar,m2var]])
            result=np.array(jax.vmap(lambda x: rot @ x @ rot.T, in_axes=2,out_axes=2)(varmat))
            m1var,m1m2covar=result[0]
            m2var=result[1,1]
            guess[self.parlist == 'modelerr_1']=np.sqrt(m1var)
            guess[self.parlist == 'modelerr_2']=np.sqrt(m2var)
            guess[self.parlist == 'modelcorr_12']=m1m2covar/np.sqrt(m1var*m2var)
            
        return guess

