import numpy as np
from inspect import signature
from functools import partial
from scipy.interpolate import splprep,splev,bisplev,bisplrep,interp1d,interp2d,RegularGridInterpolator,RectBivariateSpline
from sncosmo.salt2utils import SALT2ColorLaw
from scipy.special import factorial
import logging
log=logging.getLogger(__name__)

__priors__=dict()
def prior(prior):
    """Decorator to register a given function as a valid prior"""
    #Check that the method accepts 4 inputs: a saltresids object, a width, parameter vector, model components
    assert(len(signature(prior).parameters)==4 or len(signature(prior).parameters)==5)
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


class SALTPriors:

    def __init__(self,SALTResidsObj):
        for k in SALTResidsObj.__dict__.keys():
            self.__dict__[k] = SALTResidsObj.__dict__[k]
        self.SALTModel = SALTResidsObj.SALTModel
        self.SALTModelDeriv = SALTResidsObj.SALTModelDeriv
        self.regularizationScale=SALTResidsObj.regularizationScale
        
        self.priors={ key: partial(__priors__[key],self) for key in __priors__}
        phase=self.phaseRegularizationPoints
        wave=self.waveRegularizationPoints
        components=self.SALTModel(self.guess,evaluatePhase=phase,evaluateWave=wave)

        for prior in self.usePriors:
            result=self.priors[prior](1,self.guess,components)
        self.numBoundResids=0
        for boundedparam in self.boundedParams:
            #width,bound,x,par
            result=self.boundedprior(0.1,(0,1),self.guess,boundedparam)
            self.numBoundResids += result[0].size
    
    
    @prior
    def colorstretchcorr(self,width,x,components):
        """x1 should have no inner product with c"""
        x1=x[self.ix1]
        c=x[self.ic]
        if np.all(x1==0) or np.all(c==0):
            return np.zeros(1),np.zeros(1),np.zeros((1,self.npar))      
        x1dev=x1-np.median(x1)
        x1scale=1.48*np.median(np.abs(x1dev))
        
        cdev=c-np.median(c)
        cscale=1.48*np.median(np.abs(cdev))
        #Attempt to construct a robust correlator against the optimizer choosing certain points to blow up by using a median estimators, then transforming points. This (hopefully) gives relatively smooth derivatives
        #The psi function is identity for objects within 2 MAD, and transitions to 0 for extreme objects
        #This derivative then isn't wholly accurate since median estimator derivatives aren't accounted for. Maybe it's fine???????
        
        x1transformed,x1deriv=psi(x1dev/x1scale)
        ctransformed,cderiv=psi(cdev/cscale)
        corr=np.mean(x1transformed*ctransformed)
        jacobian=np.zeros((1,self.npar))
        jacobian[0,self.ix1]=ctransformed*x1deriv /(x1.size*x1scale)
        jacobian[0,self.ic]=x1transformed*cderiv/(x1.size*cscale)
        return corr/width,corr,jacobian/width

    @prior
    def hoststretchcorr(self,width,x,components):
        """x1 should have no inner product with xhost"""

        x1=x[self.ix1]
        xhost=x[self.ixhost]
        if np.all(x1==0) or np.all(xhost==0):
            return np.zeros(1),np.zeros(1),np.zeros((1,self.npar))      
        x1dev=x1-np.median(x1)
        x1scale=1.48*np.median(np.abs(x1dev))
        
        hostdev=xhost-np.median(xhost)
        hostscale=1.48*np.median(np.abs(hostdev))
        #Attempt to construct a robust correlator against the optimizer choosing certain points to blow up by using a median estimators, then transforming points. This (hopefully) gives relatively smooth derivatives
        #The psi function is identity for objects within 2 MAD, and transitions to 0 for extreme objects
        #This derivative then isn't wholly accurate since median estimator derivatives aren't accounted for. Maybe it's fine???????
        
        x1transformed,x1deriv=psi(x1dev/x1scale)
        hosttransformed,hostderiv=psi(hostdev/hostscale)
        corr=np.mean(x1transformed*hosttransformed)
        jacobian=np.zeros((1,self.npar))
        jacobian[0,self.ix1]=hosttransformed*x1deriv /(x1.size*x1scale)
        #jacobian[0,self.ixhost]=x1transformed*hostderiv/(x1.size*hostscale)

        return corr/width,corr,jacobian/width

    @prior
    def hostcolorcorr(self,width,x,components):
        """xhost should have no inner product with c"""

        xhost=x[self.ixhost]
        c=x[self.ic]
        if np.all(xhost==0) or np.all(c==0):
            return np.zeros(1),np.zeros(1),np.zeros((1,self.npar))      
        xhostdev=x1-np.median(xhost)
        xhostscale=1.48*np.median(np.abs(xhostdev))
        
        cdev=c-np.median(c)
        cscale=1.48*np.median(np.abs(cdev))
        #Attempt to construct a robust correlator against the optimizer choosing certain points to blow up by using a median estimators, then transforming points. This (hopefully) gives relatively smooth derivatives
        #The psi function is identity for objects within 2 MAD, and transitions to 0 for extreme objects
        #This derivative then isn't wholly accurate since median estimator derivatives aren't accounted for. Maybe it's fine???????
        
        xhosttransformed,xhostderiv=psi(xhostdev/xhostscale)
        ctransformed,cderiv=psi(cdev/cscale)
        corr=np.mean(xhosttransformed*ctransformed)
        jacobian=np.zeros((1,self.npar))
        #jacobian[0,self.ixhost]=ctransformed*xhostderiv /(xhost.size*xhostscale)
        jacobian[0,self.ic]=xhosttransformed*cderiv/(xhost.size*cscale)

        return corr/width,corr,jacobian/width
    
    
    @prior
    def m0m1prior(self,width,x,components):
        """M1 should have no inner product with the effect of reddening on M0"""
        scale,scaleDeriv=self.regularizationScale(components,componentsatreg)
        
        
        
        
        colorlaw= -0.4 * SALT2ColorLaw(self.colorwaverange, x[self.parlist == 'cl'])(wave)
        coloreffect=(np.exp(0.4*colorlaw)[np.newaxis,:]-1)
        reddenedM0= components[0]* coloreffect
        m0m1=(reddenedM0*components[1]).sum()
        m0m1scale=np.sqrt(scale[0]*scale[1])
        corr=np.array([m0m1/m0m1scale])
        #Derivative with respect to m0
        
        
        jacobian=np.zeros((1,self.npar))
        jacobian[:,self.im0]= (((components[1] * coloreffect )[:,:,np.newaxis]*self.regularizationDerivs[0]).sum(axis=(0,1)) -corr*scaleDeriv[0]*0.5*scale[1]/m0m1scale)/m0m1scale
        jacobian[:,self.im1]=  (((reddenedM0)[:,:,np.newaxis]*self.regularizationDerivs[0]).sum(axis=(0,1))-corr*scaleDeriv[1]*0.5*scale[0]/m0m1scale)/m0m1scale
        jacobian[:,self.iCL]= (( components[1]*reddenedM0)[:,:,np.newaxis]* 0.4*self.colorLawDerivInterp(wave)[np.newaxis,:,:]).sum(axis=(0,1))/m0m1scale
        residual=corr/width
        
        return corr/width,corr,jacobian/width   

    @prior
    def m0m1nocprior(self,width,x,components):
        """M1 should have no inner product with M0"""
        scale,scaleDeriv=self.regularizationScale(components,componentsatreg)       
        
        
        m0m1=(components[0]*components[1]).sum()
        m0m1scale=np.sqrt(scale[0]*scale[1])
        corr=np.array([m0m1/m0m1scale])
        #Derivative with respect to m0
        
        
        jacobian=np.zeros((1,self.npar))
        jacobian[:,self.im0]= (((components[1] )[:,:,np.newaxis]*self.regularizationDerivs[0]).sum(axis=(0,1)) -corr*scaleDeriv[0]*0.5*scale[1]/m0m1scale)/m0m1scale
        jacobian[:,self.im1]=  ((components[0][:,:,np.newaxis]*self.regularizationDerivs[0]).sum(axis=(0,1))-corr*scaleDeriv[1]*0.5*scale[0]/m0m1scale)/m0m1scale
        residual=corr/width
        
        return corr/width,corr,jacobian/width   


    @prior
    def peakprior(self,width,x,components):
        """ At t=0, minimize time derivative of B-band lightcurve"""
        M0deriv=self.SALTModelDeriv(x,1,0,self.phase,self.wave)[0]
        int1d = interp1d(self.phase,M0deriv,axis=0,assume_sorted=True)
        m0Bderiv = np.sum(self.kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
            (self.wave[1]-self.wave[0])*self.fluxfactor['default']['B']
        bStdFlux=(10**((self.m0guess-27.5)/-2.5) )
        #This derivative is constant, and never needs to be recalculated, so I store it in a hidden attribute
        try:
            m0Bderivjac = self.__peakpriorderiv__.copy()
        except:
            m0Bderivjac= np.zeros(self.npar)
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
                    derivInterp = interp1d(self.phase,bisplev(self.phase,self.wave[inbounds],(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder),dx=1),axis=0,kind=self.interpMethod,bounds_error=False,fill_value="extrapolate",assume_sorted=True)
                    m0Bderivjac[self.im0[i]] = np.sum( passbandColorExp[inbounds] * derivInterp(0))*intmult 
            self.__peakpriorderiv__=m0Bderivjac.copy()
        #import pdb;pdb.set_trace()
        value=m0Bderiv/bStdFlux
        return value/width,value, m0Bderivjac/(bStdFlux*width)
    
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
        bStdFlux=(10**((self.m0guess-27.5)/-2.5) )
        fluxratio=bStdFlux/m0Bflux
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
        
    @prior
    def m0prior(self,width,x,components):
        """Prior on the magnitude of the M0 component at t=0"""
        int1d = interp1d(self.phase,components[0],axis=0,assume_sorted=True)
        m0Bflux = np.sum(self.kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
            (self.wave[1]-self.wave[0])*self.fluxfactor['default']['B']
        #m0Bflux=np.clip(m0Bflux,(10**((self.m0guess-27.5)/-2.5) ),None)
        m0B= -2.5*np.log10(m0Bflux)+27.5
        bStdFlux=(10**((self.m0guess-27.5)/-2.5) )
        residual = (m0Bflux-bStdFlux) / (width*bStdFlux)
        #This derivative is constant, and never needs to be recalculated, so I store it in a hidden attribute
        try:
            fluxDeriv= self.__m0priorfluxderiv__.copy()
        except:
            fluxDeriv= np.zeros(self.npar)
            passbandColorExp = self.kcordict['default']['Bpbspl']
            intmult = (self.wave[1] - self.wave[0])*self.fluxfactor['default']['B']
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
                    derivInterp = interp1d(self.phase,self.regularizationDerivs[0][:,inbounds,i],axis=0,kind=self.interpMethod,bounds_error=False,fill_value="extrapolate",assume_sorted=True)
                    fluxDeriv[self.im0[i]] = np.sum( passbandColorExp[inbounds] * derivInterp(0))*intmult 
            self.__m0priorfluxderiv__=fluxDeriv.copy()
        
        jacobian=fluxDeriv/ (bStdFlux* width)
        return residual,m0B,jacobian



        
    @prior
    def m1prior(self,width,x,components):
        """M1 should have zero flux at t=0 in the B band"""
        pbspl=(self.kcordict['default']['Bpbspl']*(self.wave[1]-self.wave[0])*self.fluxfactor['default']['B'])[np.newaxis,:]
        int1d = interp1d(self.phase,components[1],axis=0,assume_sorted=True)
        m1flux = np.sum(pbspl*int1d([0]), axis=1)
        bStdFlux=(10**((self.m0guess-27.5)/-2.5) )
        residual = (m1flux) / (width*bStdFlux)
        #This derivative is constant, and never needs to be recalculated, so I store it in a hidden attribute
        try:
            fluxDeriv= self.__m1priorfluxderiv__.copy()
        except:
            fluxDeriv= np.zeros((pbspl.shape[0],self.npar))
            for i in range(self.im1.size):
                waverange=self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
                phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-self.bsorder-1),i//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
                #Check if this filter is inside values affected by changes in knot i
                minlam=min([np.min(self.kcordict['default'][filt+'wave'][self.kcordict['default'][filt+'tp'] > 0.01]) for filt in ['B']]) 
                maxlam=max([np.max(self.kcordict['default'][filt+'wave'][self.kcordict['default'][filt+'tp'] > 0.01]) for filt in ['B']]) 
                if waverange[0] > maxlam or waverange[1] < minlam:
                    pass
                if (0>=phaserange[0] ) & (0<=phaserange[1]):
                    #Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
                    #Integrate only over wavelengths within the relevant range
                    inbounds=(self.wave>waverange[0]) & (self.wave<waverange[1])
                    derivInterp = interp1d(self.phase,self.regularizationDerivs[0][:,inbounds,i],axis=0,kind=self.interpMethod,bounds_error=False,fill_value="extrapolate",assume_sorted=True)
                    fluxDeriv[:,self.im1[i]] = np.sum( pbspl[:,inbounds]* derivInterp([0]),axis=1) 
            self.__m1priorfluxderiv__=fluxDeriv.copy()
        
        jacobian=fluxDeriv/ (bStdFlux* width)
        
        return residual,m1flux/bStdFlux,jacobian

    @prior
    def hostpeakprior(self,width,x,components):
        """host component should have zero flux at t=0 in the B band.  Not sure if needed"""
        pbspl=(self.kcordict['default']['Bpbspl']*(self.wave[1]-self.wave[0])*self.fluxfactor['default']['B'])[np.newaxis,:]
        int1d = interp1d(self.phase,components[2],axis=0,assume_sorted=True)
        m1flux = np.sum(pbspl*int1d([0]), axis=1)
        bStdFlux=(10**((self.m0guess-27.5)/-2.5) )
        residual = (m1flux) / (width*bStdFlux)
        #This derivative is constant, and never needs to be recalculated, so I store it in a hidden attribute
        try:
            fluxDeriv= self.__m1priorfluxderiv__.copy()
        except:
            fluxDeriv= np.zeros((pbspl.shape[0],self.npar))
            for i in range(self.im1.size):
                waverange=self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
                phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-self.bsorder-1),i//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
                #Check if this filter is inside values affected by changes in knot i
                minlam=min([np.min(self.kcordict['default'][filt+'wave'][self.kcordict['default'][filt+'tp'] > 0.01]) for filt in ['B']]) 
                maxlam=max([np.max(self.kcordict['default'][filt+'wave'][self.kcordict['default'][filt+'tp'] > 0.01]) for filt in ['B']]) 
                if waverange[0] > maxlam or waverange[1] < minlam:
                    pass
                if (0>=phaserange[0] ) & (0<=phaserange[1]):
                    #Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
                    #Integrate only over wavelengths within the relevant range
                    inbounds=(self.wave>waverange[0]) & (self.wave<waverange[1])
                    derivInterp = interp1d(self.phase,self.regularizationDerivs[0][:,inbounds,i],axis=0,kind=self.interpMethod,bounds_error=False,fill_value="extrapolate",assume_sorted=True)
                    fluxDeriv[:,self.im1[i]] = np.sum( pbspl[:,inbounds]* derivInterp([0]),axis=1) 
            self.__m1priorfluxderiv__=fluxDeriv.copy()
        
        jacobian=fluxDeriv/ (bStdFlux* width)
        
        return residual,m1flux/bStdFlux,jacobian

    
    @prior
    def recalprior(self,width,x,components):
        """Prior on all spectral recalibration"""
        results=[]
        for snid,sn in self.datadict.items():
            for k,spectrum in sn.specdata.items():
                ispcrcl=np.where(self.parlist==f'specrecal_{snid}_{k}')[0]
                coeffs=x[ispcrcl]
                pow=coeffs.size-np.arange(coeffs.size)
                #Select equidistantly spaced points along the recalibration exponent equal to 2+the number of coefficients
                wavepoints=np.percentile(spectrum.wavelength,100.*np.arange(coeffs.size+3)/(coeffs.size+2))
                
                recalCoord=((wavepoints-np.mean(spectrum.wavelength))/self.specrange_wavescale_specrecal)
                drecaltermdrecal=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
                recalterm=(drecaltermdrecal*coeffs[np.newaxis,:]).sum(axis=1)
                
                jacobian=np.zeros((recalterm.size,self.npar))
                jacobian[:,ispcrcl]=drecaltermdrecal
                results+=[(recalterm/width,recalterm,jacobian/width)]
        residuals,values,jacobian=zip(*results)
        return np.concatenate([np.array([x]) if x.shape==() else x for x in residuals]),np.concatenate([np.array([x]) if x.shape==() else x for x in values]),np.concatenate([x if len(x.shape)==2 else x[np.newaxis,:] for x in jacobian])
    
    @prior
    def m0positiveprior(self,width,x,components):
        """Prior that m0 is not negative"""
        val=components[0].copy().flatten()
        isnegative=val<0
        val[~isnegative]=0
        jacobian=np.zeros((val.size,self.npar))
        jacobian[:,self.im0]=self.regularizationDerivs[0].reshape((-1,self.im0.size)).copy()
        jacobian[~isnegative,:]=0
        return val/width,val,jacobian
        
    @prior
    def colormean(self,width,x,components):
        """Prior such that the mean of the color population is 0"""
        mean=np.mean(x[self.ic])
        residual = mean/width
        jacobian=np.zeros(self.npar)
        jacobian[self.ic] = 1/len(self.datadict.keys())/width
        return residual,mean,jacobian

    @prior
    def x1mean(self,width,x,components):
        """Prior such that the mean of the x1 population is 0"""
        x1mean=np.mean(x[self.ix1])
        residual = x1mean/width
        jacobian=np.zeros(self.npar)
        jacobian[self.ix1] = 1/len(self.datadict.keys())/width
        return residual,x1mean,jacobian

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
    def x1std(self,width,x,components):
        """Prior such that the standard deviation of the x1 population is 1"""
        x1s=x[self.ix1]
        x1mean=np.mean(x1s)
        x1std=np.std(x1s)
        residual = (x1std-1)/width
        jacobian=np.zeros(self.npar)
        if x1std!=0:
            jacobian[self.ix1] = (x1s-x1mean)/(x1s.size*x1std*width)
        return residual,x1std,jacobian

    @prior
    def m0endalllam(self,width,x,components):
        """Prior such that at early times there is no flux"""
        thinning=4
        try:
            jacobian= self.__m0endalllamderiv__.copy()
        except:
            jacobian=np.zeros((2*self.wave[::thinning].size,self.npar))
            for i, jacind in enumerate(self.im0):
                jacobian[:,jacind] = bisplev(self.phase[:2],self.wave[::thinning],(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder)).flatten()
            self.__m0endalllamderiv__=jacobian.copy()
        value=np.dot(jacobian[:,self.im0],x[self.im0])
        residual = value/width
        jacobian/=width
        return residual,value,jacobian

    @prior
    def m1endalllam(self,width,x,components):
        """Prior such that at early times there is no flux"""
        thinning=4
        try:
            jacobian= self.__m1endalllamderiv__.copy()
        except:
            jacobian=np.zeros((2*self.wave[::thinning].size,self.npar))
            for i, jacind in enumerate(self.im1):
                jacobian[:,jacind] = bisplev(self.phase[:2],self.wave[::thinning],(self.phaseknotloc,self.waveknotloc,np.arange(self.im1.size)==i,self.bsorder,self.bsorder)).flatten()
            self.__m1endalllamderiv__=jacobian.copy()
        value=np.dot(jacobian[:,self.im1],x[self.im1])
        residual = value/width
        jacobian/=width
        return residual,value,jacobian  

    def boundedprior(self,width,bound,x,par):
        """Flexible prior that sets Gaussian bounds on parameters"""
        #import pdb;pdb.set_trace()

        lbound,ubound = bound
        
        iPar = self.__dict__['i%s'%par]
        if iPar.dtype==bool and iPar.sum() ==0: return np.zeros(0),np.zeros(0),sparse.csr_matrix((0,self.npar))
        iOut = (x[iPar] < lbound) | (x[iPar] > ubound)
        iLow = (x[iPar] < lbound)
        iHigh = (x[iPar] > ubound)
        residual = np.zeros(iPar.size)
        residual[iLow] = (x[iPar][iLow]-lbound)/(width)
        residual[iHigh] = (x[iPar][iHigh]-ubound)/(width)

        jacobian = np.zeros((x[iPar].size,self.npar))
        jacobian[iLow,iPar[iLow]] = 1/(width)
        jacobian[iHigh,iPar[iHigh]] = 1/(width)
        
        return residual,x[iPar],jacobian    

    def getBounds(self,bounds,boundparams):
        lower=np.ones(self.npar)*-np.inf
        upper=np.ones(self.npar)*np.inf
        for bound,par in zip(bounds,boundparams):
            lbound,ubound,width = bound
            iPar = self.__dict__['i%s'%par]
            lower[iPar]=lbound
            upper[iPar]=ubound
        return lower,upper
    
    def priorResids(self,priors,widths,x):
        """Given a list of names of priors and widths returns a residuals vector, list of prior values, and Jacobian """
        phase=self.phaseRegularizationPoints
        wave=self.waveRegularizationPoints
        components=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)

        results=[]
        debugstring='Prior values are '
        chi2string='Prior chi2 are '
        for prior,width in zip(priors,widths):
            try:
                priorFunction=self.priors[prior]
            except:
                raise ValueError('Invalid prior supplied: {}'.format(prior)) 
            results+=[priorFunction(width,x,components)]
            
            if results[-1][0].size==0: continue
            if results[-1][0].size==1:
                debugstring+='{}: {:.2e},'.format(prior,float(results[-1][1]))
            chi2string+='{}: {:.2e},'.format(prior,(results[-1][0]**2).sum())
                #debugstring+=f'{prior}: '+' '.join(['{:.2e}'.format(val) for val in results[-1][1]])+','
        
        log.debug(debugstring)
        log.debug(chi2string)
        residuals,values,jacobian=zip(*results)
        return np.concatenate([np.array([x]) if x.shape==() else x for x in residuals]),np.concatenate([np.array([x]) if x.shape==() else x for x in values]),np.concatenate([x if len(x.shape)==2 else x[np.newaxis,:] for x in jacobian])


    def BoundedPriorResids(self,bounds,boundparams,x):
        """Given a list of names of priors and widths returns a residuals vector, list of prior values, and Jacobian """

        components = self.SALTModel(x)
        debugstring='Values outside bounds: '
        results=[]
        for bound,par in zip(bounds,boundparams):

            result=self.boundedprior(bound[-1],(bound[0],bound[1]),x,par)
            results+=[result]
            numResids=result[0].size
            if numResids==0: continue
            if result[0].any():
                debugstring+='{} {}, '.format(par,np.nonzero(result[0])[0].size)
        residuals,values,jacobian=zip(*results)
        residuals,values,jacobian=np.concatenate([np.array([x]) if x.shape==() else x for x in residuals]),np.concatenate([np.array([x]) if x.shape==() else x for x in values]),np.concatenate([x if len(x.shape)==2 else x[np.newaxis,:] for x in jacobian])
        if residuals.any():
            log.debug(debugstring[:-1])
        return residuals,values,jacobian
