import numpy as np
from jax import numpy as jnp
import jax
from jax import lax
from jax.experimental import sparse 


from functools import partial
from saltshaker.util.jaxoptions import jaxoptions

from inspect import signature

from scipy.interpolate import splprep,splev,bisplev,bisplrep,interp1d,interp2d,RegularGridInterpolator,RectBivariateSpline
from sncosmo.salt2utils import SALT2ColorLaw
from scipy.special import factorial
import logging
log=logging.getLogger(__name__)

__priors__=dict()
def prior(prior):
    """Decorator to register a given function as a valid prior"""
    #Check that the method accepts 4 inputs: a saltresids object, a width, parameter vector, model components
    assert(len(signature(prior).parameters)==3 or len(signature(prior).parameters)==4)
    __priors__[prior.__name__]=prior
    return prior


#This is taken from https://arxiv.org/pdf/1712.05151.pdf
b=1.5
c=4
q1=1.540793 
q2 = 0.8622731
def psi(z):
    result=np.zeros(z.size)
    absz=np.abs(z)
    trans=(b<absz)&(c>absz)
    interior=b>=absz
    tanhterm=np.tanh(q2*(c-absz[trans]))
    result[trans]=q1*tanhterm*np.sign(z[trans])
    result[interior]=z[interior]
    deriv=np.zeros(z.size)
    deriv[interior]=1
    deriv[trans]=-q1*(1-tanhterm**2)*q2
    return result,deriv

def jpsi(z):
    return lax.cond(c>jnp.abs(z),
             lambda x : lax.cond( 
                jnp.abs(x)>b,
                 lambda y: q1*jnp.tanh(q2*(c-jnp.abs(y)))*jnp.sign(y),
             lambda y: y, x)
             , lambda x:0.,
             z)
jpsi=jax.vmap(jpsi)

def robustcorrelation(x,y):
    assert(x.size==y.size)
    xdev=x-jnp.median(x)
    xscale=1.48*jnp.median(jnp.abs(xdev))
    
    ydev=y-jnp.median(y)
    yscale=1.48*jnp.median(jnp.abs(ydev))
    #Attempt to construct a robust correlator against the optimizer choosing certain points to blow up by using a median estimator, then transforming points. This (hopefully) gives relatively smooth derivatives
    #The psi function is identity for objects within 2 MAD, and transitions to 0 for extreme objects
    #This derivative then isn't wholly accurate since median estimator derivatives aren't accounted for. Maybe it's fine???????
    
    xtrans=jpsi(xdev/xscale)
    ytrans=jpsi(ydev/yscale)
    
    return jnp.mean(xtrans*ytrans)


class SALTPriors:

    def __init__(self,SALTResidsObj):
        for k in SALTResidsObj.__dict__.keys():
            self.__dict__[k] = SALTResidsObj.__dict__[k]
        self.SALTModel = SALTResidsObj.SALTModel
        self.SALTModelDeriv = SALTResidsObj.SALTModelDeriv
        
        self.priors={ key: partial(__priors__[key],self) for key in __priors__}
        if len(self.usePriors) != len(self.priorWidths):
            raise RuntimeError('length of priors does not equal length of prior widths!')

  
        self.lowresphase=self.phaseRegularizationPoints
        self.lowreswave=self.waveRegularizationPoints
        
        lower=np.tile(-np.inf,self.npar)
        upper=np.tile(np.inf,self.npar)
        widths=np.tile(np.inf,self.npar)

        for parname,(lbound,ubound,width) in zip(self.boundedParams,self.bounds):
            iPar = self.__dict__['i%s'%parname]
            lower[iPar]=lbound
            upper[iPar]=ubound
            if not np.isinf(widths[iPar]).all():
                log.warning(f'Multiple bounds are being applied to parameter {parname}')
            widths[iPar]=width
            
        self.isbounded=np.where(~np.isinf(widths))[0]
        self.parameterbounds= lower[self.isbounded],upper[self.isbounded],widths[self.isbounded]
     
        m0Bderivjac= np.zeros(self.im0.size)
        passbandColorExp = self.kcordict['default']['Bpbspl']
        intmult = (self.wave[1]-self.wave[0])*self.fluxfactor['default']['B']
        for i in range(self.im0.size):
            waverange=self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
            phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-self.bsorder-1),i//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
            #Check if this filter is inside values affected by changes in knot i
            minlam=np.min(self.kcordict['default']['Bwave'][self.kcordict['default']['Btp'] > 0.01])
            maxlam=np.max(self.kcordict['default']['Bwave'][self.kcordict['default']['Btp'] > 0.01])

            if waverange[0] > maxlam or waverange[1] < minlam:
                pass
            if (0>=phaserange[0] ) & (0<=phaserange[1]):
                #Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
                #Integrate only over wavelengths within the relevant range
                inbounds=(self.wave>waverange[0]) & (self.wave<waverange[1])
                derivInterp = bisplev(np.array([0]),self.wave[inbounds],(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder),dx=1) 
                m0Bderivjac[i] = np.sum( passbandColorExp[inbounds] * derivInterp)*intmult 
        self.__peakpriorderiv__=sparse.BCOO.fromdense(m0Bderivjac)
   
        fluxDeriv= np.zeros(self.im0.size)
        for i in range(self.im0.size):
            waverange=self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
            phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-self.bsorder-1),i//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
            #Check if this filter is inside values affected by changes in knot i
            minlam=np.min(self.kcordict['default']['Bwave'][self.kcordict['default']['Btp'] > 0.01])
            maxlam=np.max(self.kcordict['default']['Bwave'][self.kcordict['default']['Btp'] > 0.01])
            if waverange[0] > maxlam or waverange[1] < minlam:
                pass
            if (0>=phaserange[0] ) & (0<=phaserange[1]):
                #Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
                #Integrate only over wavelengths within the relevant range
                inbounds=(self.wave>waverange[0]) & (self.wave<waverange[1])
                derivInterp = bisplev(np.array([0]),self.wave[inbounds],(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder))
                fluxDeriv[i] = np.sum( passbandColorExp[inbounds] * derivInterp)*intmult 
        self.__maximumlightpcderiv__=sparse.BCOO.fromdense(fluxDeriv)


        thinning=4
        jacobian=np.zeros((2*self.wave[::thinning].size,self.im0.size))
        for i in range(self.im0.size):
            jacobian[:,i] = bisplev(self.phase[:2],self.wave[::thinning],(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder)).flatten()
        self.__initialphasepcderiv__=sparse.BCOO.fromdense(jacobian)
        
        
        self.__componentderivs__=SALTResidsObj.componentderiv

        self.bstdflux=(10**((self.m0guess-27.5)/-2.5) )
                        
        self.priorexecutionlist=list(zip(self.usePriors,self.priorWidths))
        
        self.numresids=jax.eval_shape(self.priorresids,np.random.normal(1e-1,size=self.npar)).shape[0]

        
    @partial(jaxoptions, static_argnums=[0],static_argnames= ['self'],diff_argnum=1)        
    def priorresids(self,x):
        """Given a parameter vector returns a residuals vector representing the priors"""

        residuals=[]
        for prior,width in self.priorexecutionlist:
            try:
                priorFunction=self.priors[prior]
            except:
                raise ValueError('Invalid prior supplied: {}'.format(prior)) 
            residuals+=[jnp.atleast_1d(priorFunction(width,x))]
        residuals+=[self.boundedpriorresids(x)]
        return jnp.concatenate(  residuals)

    @partial(jaxoptions, static_argnums=[0],static_argnames= ['self'],diff_argnum=1)        
    def priorloglike(self,x):
        """Given a parameter vector returns a residuals vector representing the priors"""

        loglike=0.
        for prior,width in self.priorexecutionlist:
            try:
                priorFunction=self.priors[prior]
            except:
                raise ValueError('Invalid prior supplied: {}'.format(prior)) 
            loglike+=-(priorFunction(width,x)**2).sum()/2
        loglike+=-(self.boundedpriorresids(x)**2).sum()/2
        return loglike

    def boundedpriorresids(self,x):
        """Given a parameter vector, returns a residuals vector, nonzero where parameters are outside bounds specified at configuration """
        lbound,ubound,widths=self.parameterbounds
        xprime=x[self.isbounded]
        return (jnp.clip(xprime-lbound,None,0)+jnp.clip(xprime-ubound,0,None))/widths

        
    @prior
    def colorstretchcorr(self,width,x):
        """x1 should have no inner product with c"""
        x1=x[self.ix1]
        c=x[self.ic]
        if x1.size!=c.size: return np.array([])
        return robustcorrelation(x1,c)/width

    @prior
    def hoststretchcorr(self,width,x):
        """x1 should have no inner product with xhost"""

        x1=x[self.ix1]
        xhost=x[self.ixhost]
        if x1.size!=xhost.size: return np.array([])      
        return robustcorrelation(x1,xhost)/width

    @prior
    def hostcolorcorr(self,width,x):
        """xhost should have no inner product with c"""

        xhost=x[self.ixhost]
        c=x[self.ic]
        if c.size!=xhost.size: return np.array([])
        return robustcorrelation(xhost,c)/width

    @prior
    def peakprior(self,width,x):
        """ At t=0, minimize time derivative of B-band lightcurve"""
        return self.__peakpriorderiv__ @ x[self.im0] /(self.bstdflux*width)
        
        
    @prior
    def m0prior(self,width,x):
        """Prior on the magnitude of the M0 component at t=0"""
        return (self.__maximumlightpcderiv__ @ x[self.im0] -self.bstdflux)/ (self.bstdflux* width)

    @prior
    def m1prior(self,width,x):
        """M1 should have zero flux at t=0 in the B band"""        
        return self.__maximumlightpcderiv__ @ x[self.im1] / (self.bstdflux* width)

    @prior
    def hostpeakprior(self,width,x):
        """host component should have zero flux at t=0 in the B band.  Not sure if needed"""
        return self.__maximumlightpcderiv__ @ x[self.imhost] / (self.bstdflux* width)

    
    @prior
    def recalprior(self,width,x):
        """Prior on all spectral recalibration"""
        residuals=[]
        thinning=4
        for snid,sn in self.datadict.items():
            for k,spectrum in sn.specdata.items():

                coeffs=x[spectrum.ispcrcl]
                recalterm=spectrum.recaltermderivs[::thinning,:] @ coeffs

                residuals+=[(recalterm/width)]
        return jnp.concatenate(residuals)
    
    @prior
    def m0positiveprior(self,width,x):
        """Prior that m0 is not negative"""
        return jnp.clip(components[0],None,0).flatten()/width
        
    @prior
    def colormean(self,width,x):
        """Prior such that the mean of the color population is 0"""
        return jnp.mean(x[self.ic])/width

    @prior
    def x1mean(self,width,x):
        """Prior such that the mean of the x1 population is 0"""
        return jnp.mean(x[self.ix1])/width

    # for now host coords should be fixed
    # but in future can allow these to change within their measurement uncertainties
    #@prior
    #def xhostmean(self,width,x,components):
    #   """Prior such that the host coordinate squared is 1"""
    #   xhost=(x[self.ixhost]**2.-1)
    #   residual = xhosttot/width
    #   jacobian=np.zeros(self.npar)
    #   jacobian[self.ixhost] = 2*x[self.ixhost]/width # 1/len(self.datadict.keys())/width
    #   return residual,xhost,jacobian

    
    @prior
    def x1std(self,width,x):
        """Prior such that the standard deviation of the x1 population is 1"""
        return (jnp.std(x[self.ix1])-1)/width

    @prior
    def m0endalllam(self,width,x):
        """Prior such that at early times there is no flux"""
        return self.__initialphasepcderiv__ @ x[self.im0]/width

    @prior
    def m1endalllam(self,width,x):
        """Prior such that at early times there is no flux"""
        return self.__initialphasepcderiv__ @ x[self.im1]/width
                    
        
        

    
        
        
    def satisfyDefinitions(self,X,components):
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
        if self.errbsorder==0:
            m0variance+=2*meanx1*m0m1covariance+meanx1**2*m1variance
            m0m1covariance+=m1variance*meanx1
        else:
            log.critical('RESCALING ERROR TO SATISFY DEFINITIONS HAS NOT BEEN IMPLEMENTED')

        
        #Define x1 to have std deviation 1
        x1std = np.std(X[self.ix1])
        if x1std == x1std and x1std != 0.0:
            X[self.im1]*= x1std
            X[self.ix1]/= x1std
        if self.errbsorder==0:
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
        if self.errbsorder==0:
            m1variance*=fluxratio**2
            m0variance*=fluxratio**2
            m0m1covariance*=fluxratio**2
            if self.host_component:
                mhostvariance*=fluxratio**2
                m0mhostcovariance*=fluxratio**2
        else:
            log.critical('RESCALING ERROR TO SATISFY DEFINITIONS HAS NOT BEEN IMPLEMENTED')

#       Define color to have 0 mean
#       centralwavelength=self.waveBinCenters[np.arange(self.im0.size)%self.waveBinCenters.size])
#       
#       X[self.im0]*=
#       stats.pearsonr(X[self.ix1],X[self.ic])
#       
#       X[self.ic]-=X[self.ix1]
        if self.errbsorder==0:
            X[self.imodelerr0]= np.sqrt(m0variance)
            X[self.imodelcorr01]= m0m1covariance/np.sqrt(m0variance*m1variance)
            X[self.imodelerr1]=np.sqrt(m1variance)
            if self.host_component:
                X[self.imodelerrhost]=np.sqrt(mhostvariance)
                X[self.imodelcorr0host]= m0mhostcovariance/np.sqrt(m0variance*mhostvariance)

        return X
