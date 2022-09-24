from saltshaker.util.readutils import SALTtrainingSN,SALTtraininglightcurve,SALTtrainingspectrum

from sncosmo.salt2utils import SALT2ColorLaw

from scipy.special import factorial
from scipy.interpolate import splprep,splev,bisplev,bisplrep,interp1d,interp2d,RegularGridInterpolator,RectBivariateSpline
from scipy import sparse as scisparse
import numpy as np

from jax import numpy as jnp
import jax
from jax.experimental import sparse
from jax import lax
from jax.scipy import linalg as jaxlinalg

from functools import partial
from saltshaker.util.jaxoptions import jaxoptions

import extinction
import copy
import warnings
import logging
log=logging.getLogger(__name__)

warnings.simplefilter('ignore',category=FutureWarning)

from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
recalmax=20

_SCALE_FACTOR = 1e-12

def __anyinnonzeroareaforsplinebasis__(phase,wave,phaseknotloc,waveknotloc,bsorder,i):
    phaseindex,waveindex=i//(waveknotloc.size-bsorder-1), i% (waveknotloc.size-bsorder-1)
    return ((phase>=phaseknotloc[phaseindex])&(phase<=phaseknotloc[phaseindex+bsorder+1])).any() and ((wave>=waveknotloc[waveindex])&(wave<=waveknotloc[waveindex+bsorder+1])).any() 

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


class SALTfitcachelightcurve(SALTtraininglightcurve):

    __slots__ = ['idx','pbspl','denom','lambdaeff','bsplinecoeffshape','colorlawderiv'
    ,'im0','im1','iCL','ix0','ix1','ic','z','splinebasisconvolutions',
    'lambdaeffrest','varianceprefactor',
    'errorgridshape',
    'imodelcorr01','imodelerr0','imodelerr1','iclscat']
    

    def __init__(self,sn,lc,residsobj,kcordict):
        for key,val in lc.__dict__.items():
            self.__dict__[key]=copy.deepcopy(val)
        z = sn.zHelio
        
        #Define quantities for synthetic photometry
        filtwave = kcordict[sn.survey][lc.filt]['filtwave']
        filttrans = kcordict[sn.survey][lc.filt]['filttrans']
        
        self.idx   = (sn.obswave>= kcordict[sn.survey][lc.filt]['minlam']) & \
                     (sn.obswave<= kcordict[sn.survey][lc.filt]['maxlam'])  # overlap range
        pbspl = np.interp(sn.obswave[self.idx],filtwave,filttrans)
        pbspl *= sn.obswave[self.idx]
        denom = np.trapz(pbspl,sn.obswave[self.idx])
        pbspl /= denom*HC_ERG_AA
        self.id= f'{sn.snid}_{lc.filt}'
        
        self.pbspl = pbspl
        self.denom = denom
        self.lambdaeff = kcordict[sn.survey][lc.filt]['lambdaeff']    
        self.lambdaeffrest= self.lambdaeff/(1+z)    
        self.bsplinecoeffshape=(residsobj.phaseBins[0].size,residsobj.waveBins[0].size)        
        self.z=z
        
        self.preintegratebasis=residsobj.preintegrate_photometric_passband
        
        self.ix1=sn.ix1
        self.icoordinates=sn.icoordinates
        self.icomponents=residsobj.icomponents
        
        self.iCL=residsobj.iCL
        self.ix0=sn.ix0
        self.ic=sn.ic
        self.iclscat=residsobj.iclscat
        
        clippedphase=np.clip(self.phase,residsobj.phase.min(),residsobj.phase.max())
#########################################################################################
        #Evaluating a bunch of quantities used in the flux model
        #Evaluate derivatives of the color law, here assumed to be linear in coefficients  
        #Either evaluate on fine-grained wavelength grid, or at center of spline basis functions      
        self.colorlawderiv=residsobj.colorlawderiv
        self.colorlawzero=residsobj.colorlawzero
        if not self.preintegratebasis:
            self.colorlawderiv=self.colorlawderiv[self.idx]
            self.colorlawzero=self.colorlawzero[self.idx]
        #Calculate zero point of flux
        dwave=sn.dwave
        fluxfactor=residsobj.fluxfactor[sn.survey][self.filt]

        wave=residsobj.wave[self.idx]
        #Evaluate the b-spline basis functions for this passband
        #Evaluate parameters only for relevant portions of phase/wavelength space        
        inds=np.array(range(residsobj.im0.size))
        phaseind,waveind=inds//(residsobj.waveknotloc.size-residsobj.bsorder-1),inds%(residsobj.waveknotloc.size-residsobj.bsorder-1)
        inphase=((clippedphase[:,np.newaxis]>= residsobj.phaseknotloc[np.newaxis,phaseind])&(clippedphase[:,np.newaxis]<=residsobj.phaseknotloc[np.newaxis,phaseind+residsobj.bsorder+1])).any(axis=0)
        inwave=((wave.max()>=residsobj.waveknotloc[waveind])&(wave.min()<=residsobj.waveknotloc[waveind+residsobj.bsorder+1]))

        isrelevant=inphase&inwave
        #Array output indices match time along 0th axis, wavelength along 1st axis
        derivInterp=np.zeros((clippedphase.size,self.idx.sum(),residsobj.im0.size))        
        for i in np.where(isrelevant)[0]:
                derivInterp[:,:,i] = bisplev(clippedphase ,wave,(residsobj.phaseknotloc,residsobj.waveknotloc,np.arange(residsobj.im0.size)==i, residsobj.bsorder,residsobj.bsorder))

        self.splinebasisconvolutions=[]
        #Redden passband transmission by MW extinction, multiply by scalar factors
        reddenedpassband=sn.mwextcurve[self.idx]*self.pbspl*dwave*fluxfactor*_SCALE_FACTOR/(1+self.z)
        self.prefactor=reddenedpassband
    
        for pdx in range(len(lc)):
            #For photometry past the edge of the phase grid, extrapolate a linear decline in magnitudes
            if self.phase[pdx]>sn.obsphase.max():
                decayFactor= 10**(-0.4*residsobj.extrapolateDecline*(self.phase[pdx]-sn.obsphase.max()))
            else:
                decayFactor=1
            if self.preintegratebasis:
                self.splinebasisconvolutions+=[decayFactor*(np.dot(derivInterp[pdx,:,:].T,reddenedpassband))]
            else:
                
                self.splinebasisconvolutions+=[decayFactor*(derivInterp[pdx,:,:]*reddenedpassband[:,np.newaxis])]
        self.splinebasisconvolutions=sparse.BCOO.fromdense( np.stack(self.splinebasisconvolutions))
#########################################################################################
        
        #Quantities used in computation of model uncertainties
        #Color law derivatives at filter effective wavelength
        self.colorlawderivlambdaeff=np.array([
        SALT2ColorLaw(residsobj.colorwaverange, np.arange(self.iCL.size)==i)(np.array([self.lambdaeffrest]))-\
                SALT2ColorLaw(residsobj.colorwaverange, np.zeros(self.iCL.size))(np.array([self.lambdaeffrest])) for i in range(self.iCL.size)])[:,0]
        self.colorlawzerolambdaeff = SALT2ColorLaw(residsobj.colorwaverange, np.zeros(self.iCL.size))(np.array([self.lambdaeffrest]))[0]
        #Prefactor for variance
        self.varianceprefactor=fluxfactor*(self.pbspl.sum())*dwave* _SCALE_FACTOR*sn.mwextcurveint(self.lambdaeff) /(1+self.z)
        
        #Identify the relevant error model parameters
        errorwaveind=np.searchsorted(residsobj.errwaveknotloc,self.lambdaeffrest)-1
        errorphaseind=(np.searchsorted(residsobj.errphaseknotloc,clippedphase)-1)
        self.errorgridshape=(residsobj.errphaseknotloc.size-1,residsobj.errwaveknotloc.size-1)
        waveindtemp=np.array([errorwaveind for x in errorphaseind])
        ierrorbin=np.ravel_multi_index((errorphaseind,waveindtemp),self.errorgridshape)
        self.imodelcorrs=[(0,1,residsobj.imodelcorr01[ierrorbin])]    
        self.imodelerrs=[residsobj.imodelerr0[ierrorbin] ,residsobj.imodelerr1[ierrorbin]    ]
        if residsobj.host_component:
            self.imodelcorrs+=[(0,2,residsobj.imodelcorr0host[ierrorbin])]
            self.imodelerrs+=[residsobj.imodelerrhost[ierrorbin]]
        self.imodelerrs=np.array(self.imodelerrs)
        
        pow=self.iclscat.size-1-np.arange(self.iclscat.size)
        colorscateval=((self.lambdaeffrest-5500)/1000)
        
        self.clscatderivs=((colorscateval)  ** (pow)) / factorial(pow)

    
    @partial(jaxoptions, static_argnums=[0],static_argnames= ['self'],jac_argnums=1)                      
    def modelflux(self,pars):
        #Define parameters
        x0,c=pars[np.array([self.ix0,self.ic])]
        #Evaluate the coefficients of the spline bases
        
        coordinates=jnp.concatenate((jnp.ones(1), pars[self.icoordinates]))
        components=pars[self.icomponents]
        fluxcoeffs=jnp.dot(coordinates,components)*x0
        
        #Evaluate color law at the wavelength basis centers
        colorlaw=jnp.dot(self.colorlawderiv,pars[self.iCL])+self.colorlawzero
        #Exponentiate and multiply by color
        colorexp= 10. ** (  -0.4*colorlaw* c)
        if self.preintegratebasis:
            #Redden flux coefficients
            fluxcoeffsreddened= (colorexp[np.newaxis,:]*fluxcoeffs.reshape( self.bsplinecoeffshape)).flatten()
            #Multiply spline bases by flux coefficients
            
            
            return self.splinebasisconvolutions @ fluxcoeffsreddened
        else:    
            #Integrate basis functions over wavelength and sum over flux coefficients
            return ( self.splinebasisconvolutions @ fluxcoeffs) @ colorexp 
            
    @partial(jaxoptions, static_argnums=[0],static_argnames= ['self'],jac_argnums=1)                      
    def modelfluxvariance(self,pars):
        x0,c=pars[np.array([self.ix0,self.ic])]
        #Evaluate color law at the wavelength basis centers
        colorlaw=jnp.dot(self.colorlawderivlambdaeff,pars[self.iCL])+self.colorlawzerolambdaeff
        #Exponentiate and multiply by color
        colorexp= 10. ** (  -0.4*colorlaw* c)
  
          #Evaluate model uncertainty

        coordinates=jnp.concatenate((jnp.ones(1), pars[self.icoordinates]))
        errs= pars[self.imodelerrs]
        errorsurfaces=jnp.dot(coordinates,errs)**2
        for i,j,corridx in self.imodelcorrs:
            errorsurfaces= errorsurfaces+2*coordinates[i]*coordinates[j]* errs[i]*errs[j]
              
        modelfluxvar=colorexp**2 * self.varianceprefactor**2 * x0**2* errorsurfaces
        return jnp.clip(modelfluxvar,0,None)

    @partial(jaxoptions, static_argnums=[0],static_argnames= ['self'],jac_argnums=1)                       
    def colorscatter(self,pars):
        clscatpars = pars[self.iclscat]
        return  jnp.exp(self.clscatderivs @ clscatpars)

    @partial(jaxoptions, static_argnums=[0,3,4],static_argnames= ['self','fixuncertainties','fixfluxes'],jac_argnums=1)        
    def modelloglikelihood(self,x,cachedresults=None,fixuncertainties=False,fixfluxes=False):
        resids=self.modelresidual(x,cachedresults,fixuncertainties,fixfluxes)
        return resids['lognorm']- (resids['residuals']**2).sum() / 2.  
        
        
    @partial(jaxoptions, static_argnums=[0,3,4],static_argnames= ['self','fixuncertainties','fixfluxes'],jac_argnums=1)        
    def modelresidual(self,x,cachedresults=None,fixuncertainties=False,fixfluxes=False):
   
        if fixfluxes:
            modelflux=cachedresults
        else:
            modelflux=self.modelflux(x)

        if fixuncertainties:
             modelvariance,clscat=cachedresults
        else:
            modelvariance=self.modelfluxvariance(x)
            clscat=self.colorscatter(x)

        variance=self.fluxcalerr**2 + modelvariance            
        sigma=jnp.sqrt(variance)
        
        #if clscat>0, then need to use a cholesky matrix to find pulls
        def choleskyresidsandnorm( variance,clscat,modelflux):
#             cholesky=jaxrankOneCholesky(variance,clscat**2,modelflux)
            cholesky=jaxlinalg.cholesky(jnp.diag(variance)+ clscat**2*jnp.outer(modelflux,modelflux),lower=True)
            return {'residuals':jaxlinalg.solve_triangular(cholesky, modelflux-self.fluxcal,lower=True), 
            'lognorm': -jnp.log(jnp.diag(cholesky)).sum()}
        
        def diagonalresidsandnorm(variance,clscat,modelflux):
            sigma=jnp.sqrt(variance)
            return {'residuals':(modelflux-self.fluxcal)/sigma,'lognorm': -jnp.log(sigma).sum()}
        return lax.cond(clscat==0, diagonalresidsandnorm, choleskyresidsandnorm, 
             variance,clscat,modelflux )
#         return lax.cond(clscat==0, diagonalresidsandnorm, choleskyresidsandnorm, 
#              variance,clscat,modelflux )
  
  

    
class SALTfitcachespectrum(SALTtrainingspectrum):
# 'phase','wavelength','flux','fluxerr','tobs','mjd'
    __slots__ = [
        'restwavelength',
        'im0','im1','iCL','ix1','ic','ispecx0','ispcrcl',
        'z','spectrumid','mwextcurve',
        'pcderivsparse','recaltermderivs',
        'bsplinecoeffshape','varianceprefactor',
    'errorgridshape',
    'imodelcorr01','imodelerr0','imodelerr1']
    
    def __init__(self,sn,spectrum,k,residsobj):
        z = sn.zHelio
        self.z=z

        self.flux = spectrum.flux
        self.phase = spectrum.phase
        self.wavelength = spectrum.wavelength
        self.restwavelength= spectrum.wavelength/ (1+self.z)
        self.fluxerr = spectrum.fluxerr
        self.tobs = spectrum.tobs
        self.id='{}_{}'.format(sn.snid,k)
        
        self.ix1=sn.ix1
        self.iCL=residsobj.iCL
        self.ic=sn.ic
        self.ispecx0=np.where(residsobj.parlist==f'specx0_{self.id}')[0][0]
        self.ispcrcl=np.where(residsobj.parlist==f'specrecal_{self.id}')[0]
        self.bsplinecoeffshape=(residsobj.phaseBins[0].size,residsobj.waveBins[0].size)        

        self.icomponents=residsobj.icomponents
        self.icoordinates=sn.icoordinates
        self.mwextcurve=sn.mwextcurveint(spectrum.wavelength)

        wave=self.restwavelength
        #Evaluate the b-spline basis functions for this passband
        #Evaluate parameters only for relevant portions of phase/wavelength space        
        inds=np.array(range(residsobj.im0.size))
        phaseind,waveind=inds//(residsobj.waveknotloc.size-residsobj.bsorder-1),inds%(residsobj.waveknotloc.size-residsobj.bsorder-1)
        inphase=((self.phase>= residsobj.phaseknotloc[phaseind])&(self.phase<=residsobj.phaseknotloc[phaseind+residsobj.bsorder+1]))
        inwave=((wave.max()>=residsobj.waveknotloc[waveind])&(wave.min()<=residsobj.waveknotloc[waveind+residsobj.bsorder+1]))

        isrelevant=inphase&inwave

        derivInterp=np.zeros((spectrum.wavelength.size,residsobj.im0.size))
        for i in np.where(isrelevant)[0]:
                derivInterp[:,i] = bisplev(spectrum.phase,self.restwavelength,(residsobj.phaseknotloc,residsobj.waveknotloc,np.arange(residsobj.im0.size)==i, residsobj.bsorder,residsobj.bsorder))
        derivInterp=derivInterp*(_SCALE_FACTOR/(1+self.z)*self.mwextcurve)[:,np.newaxis]
        self.pcderivsparse=sparse.BCOO.fromdense(derivInterp)
        self.spectrumid=k
        

        pow=self.ispcrcl.size-jnp.arange(self.ispcrcl.size)
        recalCoord=(self.wavelength-jnp.mean(self.wavelength))/residsobj.specrange_wavescale_specrecal
        #recalCoord=(residsobj.waveBinCenters-jnp.mean(self.wavelength))/residsobj.specrange_wavescale_specrecal
        self.recaltermderivs=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
        

        self.varianceprefactor= _SCALE_FACTOR*sn.mwextcurveint(self.wavelength) /(1+self.z)
        
        errorwaveind=np.searchsorted(residsobj.errwaveknotloc,self.restwavelength)-1
        errorphaseind=(np.searchsorted(residsobj.errphaseknotloc,self.phase)-1)
        self.errorgridshape=(residsobj.errphaseknotloc.size-1,residsobj.errwaveknotloc.size-1)
        phaseindtemp=np.tile(errorphaseind,errorwaveind.size )
        ierrorbin=np.ravel_multi_index((phaseindtemp,errorwaveind),self.errorgridshape)
        self.imodelcorrs=[(0,1,residsobj.imodelcorr01[ierrorbin])]    
        self.imodelerrs=[residsobj.imodelerr0[ierrorbin] ,residsobj.imodelerr1[ierrorbin]    ]
        if residsobj.host_component:
            self.imodelcorrs+=[(0,2,residsobj.imodelcorr0host[ierrorbin])]
            self.imodelerrs+=[residsobj.imodelerrhost[ierrorbin]]
        self.imodelerrs=np.array(self.imodelerrs)

        
    @partial(jaxoptions, static_argnums=[0],static_argnames= ['self'],jac_argnums=1)                      
    def modelflux(self,pars):
        x0=pars[self.ispecx0]
        #Define recalibration factor
        coeffs=pars[self.ispcrcl]
        recalterm=jnp.dot(self.recaltermderivs,coeffs)
        recalterm=jnp.clip(recalterm,-recalmax,recalmax)
        recalexp=jnp.exp(recalterm)

        coordinates=jnp.concatenate((jnp.ones(1), pars[self.icoordinates]))
        components=pars[self.icomponents]

        fluxcoeffs=jnp.dot(coordinates,components)*x0

        return recalexp*(self.pcderivsparse @ (fluxcoeffs))

    @partial(jaxoptions, static_argnums=[0],static_argnames= ['self'],jac_argnums=1)                              
    def modelfluxvariance(self,pars):
        x0=pars[self.ispecx0]
        #Define recalibration factor
        coeffs=pars[self.ispcrcl]
        recalterm=jnp.dot(self.recaltermderivs,coeffs)
        recalterm=jnp.clip(recalterm,-recalmax,recalmax)
        recalexp=jnp.exp(recalterm)
        coordinates=jnp.concatenate((jnp.ones(1), pars[self.icoordinates]))
        #Evaluate model uncertainty
        errs= pars[self.imodelerrs]
        errorsurfaces=jnp.dot(coordinates,errs)**2
        for i,j,corridx in self.imodelcorrs:
            errorsurfaces= errorsurfaces+2*coordinates[i]*coordinates[j]* errs[i]*errs[j]
        modelfluxvar=recalexp**2 * self.varianceprefactor**2 * x0**2* errorsurfaces
        return jnp.clip(modelfluxvar,0,None)

    @partial(jaxoptions, static_argnums=[0,3,4],static_argnames= ['self','fixuncertainties','fixfluxes'],jac_argnums=1)        
    def modelresidual(self,x,cachedresults=None,fixuncertainties=False,fixfluxes=False):
            if fixfluxes:
                modelflux=cachedresults
            else:
                modelflux=self.modelflux(x)
        
            if fixuncertainties:
                 modelvariance=cachedresults
            else:
                modelvariance=self.modelfluxvariance(x)
        
            variance=self.fluxerr**2 + modelvariance
          
            uncertainty=jnp.sqrt(variance)

            return {'residuals':  (modelflux-self.flux)/uncertainty,
                        'lognorm': -jnp.log(uncertainty).sum()}
                        
    @partial(jaxoptions, static_argnums=[0,3,4],static_argnames= ['self','fixuncertainties','fixfluxes'],jac_argnums=1)        
    def modelloglikelihood(self,x,cachedresults=None,fixuncertainties=False,fixfluxes=False):
        resids=self.modelresidual(x,cachedresults,fixuncertainties,fixfluxes)
        return resids['lognorm']- (resids['residuals']**2).sum() / 2.  
 
        
class SALTfitcacheSN(SALTtrainingSN):
    """Class to store SN data in addition to cached results useful in speeding up the fitter
    """
    __slots__ = ['photdata','specdata','ix0','ix1','ic','mwextcurve','mwextcurveint','dwave','obswave','obsphase']
    
    def __init__(self,sndata,residsobj,kcordict):
        for key,val in sndata.__dict__.items():
            if key=='photdata' or  key=='specdata':
                pass
            else:
                self.__dict__[key]=copy.deepcopy(val)
        self.obswave = residsobj.wave*(1+self.zHelio)
        self.obsphase = residsobj.phase*(1+self.zHelio)
        self.dwave = residsobj.wave[1]*(1+self.zHelio) - residsobj.wave[0]*(1+self.zHelio)
        self.mwextcurve   = 10**(-0.4*extinction.fitzpatrick99(self.obswave,sndata.MWEBV*3.1))
        self.mwextcurveint = interp1d(
            self.obswave,self.mwextcurve ,kind=residsobj.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
        

        self.ix0=np.where(residsobj.parlist==f'x0_{self.snid}')[0][0]
        self.ix1=np.where(residsobj.parlist==f'x1_{self.snid}')[0][0]
        self.ixhost=np.where(residsobj.parlist==f'xhost_{self.snid}')[0]
        if len(self.ixhost): self.ixhost = self.ixhost[0]
        self.ic=np.where(residsobj.parlist==f'c_{self.snid}')[0][0]

        self.icoordinates=np.array([self.ix1]+([self.ixhost]if residsobj.host_component else []))
       
        self.photdata={flt: SALTfitcachelightcurve(self,sndata.photdata[flt],residsobj,kcordict ) for flt in sndata.photdata}
        self.specdata={k: SALTfitcachespectrum(self,sndata.specdata[k],k,residsobj ) for k in sndata.specdata }
        