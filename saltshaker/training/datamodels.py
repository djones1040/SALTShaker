from saltshaker.util.readutils import SALTtrainingSN,SALTtraininglightcurve,SALTtrainingspectrum


from scipy.special import factorial
from scipy.interpolate import bisplev, interp1d
from scipy import sparse as scisparse
import numpy as np

from jax import numpy as jnp
import jax
from jax.experimental import sparse
from jax.scipy import linalg as jaxlinalg
from jax.tree_util import register_pytree_node_class


from functools import partial
from saltshaker.util.jaxoptions import jaxoptions,sparsejaxoptions

import extinction
import warnings
import logging
import abc

log=logging.getLogger(__name__)

warnings.simplefilter('ignore',category=FutureWarning)

from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
recalmax=20

_SCALE_FACTOR = 1e-12

def __anyinnonzeroareaforsplinebasis__(phase,wave,phaseknotloc,waveknotloc,bsorder,i):
    phaseindex,waveindex=i//(waveknotloc.size-bsorder-1), i% (waveknotloc.size-bsorder-1)
    return ((phase>=phaseknotloc[phaseindex])&(phase<=phaseknotloc[phaseindex+bsorder+1])).any() and ((wave>=waveknotloc[waveindex])&(wave<=waveknotloc[waveindex+bsorder+1])).any() 

def mvn_likelihood_simplified_cov(x, mu, D_diag, u):
    """
    Compute the likelihood of a multivariate normal distribution with
    covariance matrix Sigma = diag(D_diag) + u*u^T
    
    Parameters:
    -----------
    x : jnp.ndarray
        Observed vector (or batch of vectors) of shape (..., d)
    mu : jnp.ndarray
        Mean vector of shape (d,)
    D_diag : jnp.ndarray
        Diagonal elements of the diagonal matrix D, shape (d,)
    u : jnp.ndarray
        Vector for the outer product, shape (d,)
        
    Returns:
    --------
    likelihood : jnp.ndarray
        The likelihood value(s)
    """
    # Get dimensions
    d = mu.shape[0]
    
    # Center the data
    x_centered = x - mu
    
    # Compute D^(-1)
    D_inv_diag = 1.0 / D_diag
    
    # Compute u^T D^(-1) u
    u_D_inv_u = jnp.sum(u * D_inv_diag * u)
    
    # Compute log determinant using the matrix determinant lemma
    # |D + uu^T| = |D| * (1 + u^T D^(-1) u)
    log_det = jnp.sum(jnp.log(D_diag)) + jnp.log(1.0 + u_D_inv_u)
    
    # Compute D^(-1) x
    D_inv_x = x_centered * D_inv_diag
    
    # Compute u^T D^(-1) x
    u_D_inv_x = jnp.sum(u * D_inv_x, axis=-1)
    
    # Compute the quadratic form (x - μ)^T Σ^(-1) (x - μ) using Sherman-Morrison formula
    # (D + uu^T)^(-1) = D^(-1) - (D^(-1)u u^T D^(-1))/(1 + u^T D^(-1) u)
    quad_form_1 = jnp.sum(x_centered * D_inv_x, axis=-1)
    quad_form_2 = (u_D_inv_x ** 2) / (1.0 + u_D_inv_u)
    quad_form = quad_form_1 - quad_form_2
    
    # Compute log likelihood
    log_likelihood = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det + quad_form)
    
    # Return likelihood
    return (log_likelihood)

def toidentifier(input):
    return "x"+str(abs(hash(input)))


@register_pytree_node_class
class SALTparameters:          
    __slots__=['x0','coordinates','components','c','CL',
    'modelcorrs','modelerrs','spcrcl','clscat','surverrfloor']
    __ismapped__={'x0','c','coordinates','spcrcl','surverrfloor'}
    def __init__(self,data,parsarray):
        for var in self.__slots__:
            #determine appropriate indices
            indexvar=f'i{var}'
            if isinstance(data,dict) and indexvar in data:
                idxs=data[indexvar]
            elif '__indexattributes__' in dir(data) and indexvar in data.__indexattributes__:
                idxs=getattr(data,indexvar)
            else:
                idxs=np.array([])
            
            if idxs.size>0:
                vals=parsarray[idxs]
            else:
                vals=np.array([])
            setattr(self,var, vals)
        
    def tree_flatten(self):
        children =tuple(getattr(self,x) for x in self.__slots__)
        aux_data =tuple()
        return (children, aux_data)
    
    @property
    def mappingaxes(self):
        #Return None if the attribute is empty, otherwise check the ismapped attribute for if it's supposed to change from data to data
        return [(( 0 if x in self.__ismapped__ else None) if getattr(self,x).size>0 else None) for x in self.__slots__]

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        self=cls.__new__(cls)
        for attr,val in zip(self.__slots__,children):
            setattr(self,attr,val)
        return self


class modeledtrainingdata(metaclass=abc.ABCMeta):
    
    __slots__=[]

    @property
    @abc.abstractmethod
    def __staticattributes__(self):
        """List of which attributes should be considered static for jax compilation"""
        pass
             
    @property
    @abc.abstractmethod
    def __dynamicattributes__(self):
        """List of which attributes should be considered dynamic for jax compilation"""
        pass
    
    @abc.abstractmethod
    def modelresidual(self,x,cachedresults=None,fixuncertainties=False,fixfluxes=False):
        """Calculate residuals and log-normalization term for this data given the model"""
        pass    

    @abc.abstractmethod
    def modelfluxvariance(self,pars):
        """Calculate the predicted model variance given the data and model"""
        pass
        
    @abc.abstractmethod
    def modelflux(self,pars):
        """Calculate the predicted flux given the data and model"""
        pass
    
    def modelloglikelihood(self,x,cachedresults=None,fixuncertainties=False,fixfluxes=False):
        resids=self.modelresidual(x,cachedresults,fixuncertainties,fixfluxes)
        
        return resids['lognorm']- ((resids['residuals']**2).sum() / 2.)

    def tree_flatten(self):
        children =tuple(getattr(self,x) for x in self.__dynamicattributes__)
        aux_data =tuple(getattr(self,x) for x in self.__staticattributes__)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        self=cls.__new__(cls)
        for attr,val in zip(self.__dynamicattributes__,children):
            setattr(self,attr,val)
        for attr,val in zip(self.__staticattributes__,aux_data):
            setattr(self,attr,val)
        return self

    def unpack(self):
        return tuple(getattr(self,x) for x in self.__slots__)
 
    @classmethod
    def repack(cls, data):
        self=cls.__new__(cls)
        for attr,val in zip(self.__slots__,data):
            setattr(self,attr,val)
        return self
       
    @abc.abstractmethod
    def __len__(self):
        pass
        
    def determineneededparameters(self,modelobj):
        return []
    
@register_pytree_node_class
class modeledtraininglightcurve(modeledtrainingdata):
    __indexattributes__=['iCL','ix0','ic','icoordinates', 'icomponents','imodelcorrs', 'imodelerrs','iclscat','ipad','isurverrfloor']
    
    __dynamicattributes__= [
        'phase','fluxcal','fluxcalerr',
        'lambdaeff','lambdaeffrest',
        'errordesignmat','pcderivsparse',
        'varianceprefactor',
        'clscatderivs',
        'wavebasis',
        'padding',
    ]+__indexattributes__
    __staticattributes__=[
        'preintegratebasis',
        'imodelcorrs_coordinds',
        'bsplinecoeffshape','errorgridshape','uniqueid',
        'colorlawfunction'
    ]
    
    __slots__ = __dynamicattributes__+__staticattributes__
    
    __ismapped__={
    'ix0','ic','icoordinates','isurverrfloor','ipad','phase','fluxcal','fluxcalerr',
        'lambdaeff','lambdaeffrest','errordesignmat','pcderivsparse',
        'varianceprefactor',
        'clscatderivs',
        'padding','uniqueid'
    }

    def __init__(self,sn,lc,residsobj,kcordict,padding=0):
        for attr in lc.__slots__:
            if attr in self.__slots__:
                setattr(self,attr,getattr(lc,attr))
        
        z = sn.zHelio
        padding=max(0,padding)
        self.padding=padding
        self.ipad= np.arange(len(lc)+padding )>= len(lc)
        self.uniqueid= f'{sn.snid}_{lc.filt}'
        #Define quantities for synthetic photometry
        filtwave = kcordict[sn.survey][lc.filt]['filtwave']
        filttrans = kcordict[sn.survey][lc.filt]['filttrans']
        
        waveidxs   = (sn.obswave>= kcordict[sn.survey][lc.filt]['minlam']) & \
                     (sn.obswave<= kcordict[sn.survey][lc.filt]['maxlam'])  # overlap range
        pbspl = np.interp(sn.obswave[waveidxs],filtwave,filttrans)
        pbspl *= sn.obswave[waveidxs]
        denom = np.trapz(pbspl,sn.obswave[waveidxs])
        pbspl /= denom*HC_ERG_AA
        
        self.lambdaeff = kcordict[sn.survey][lc.filt]['lambdaeff']    
        self.lambdaeffrest= self.lambdaeff/(1+z)    
        self.bsplinecoeffshape=(residsobj.phaseBins[0].size,residsobj.waveBins[0].size)        
        
        self.preintegratebasis=residsobj.preintegrate_photometric_passband
        
        self.icoordinates=sn.icoordinates
        self.icomponents=residsobj.icomponents
        
        self.iCL=residsobj.iCL
        self.ix0=sn.ix0
        self.ic=sn.ic
        self.iclscat=residsobj.iclscat
        self.imodelcorrs= np.array([np.arange(x,y+1) for x,y in (zip(residsobj.corrmin,residsobj.corrmax))])
        self.imodelcorrs_coordinds= np.array([(-1,comb[1]) if 'host'== comb[0] else ((comb[0],-1) if 'host'== comb[1] else comb) for comb in residsobj.corrcombinations]
           )
                
        self.imodelerrs= np.array([np.arange(x,y+1) for x,y in (zip(residsobj.errmin,residsobj.errmax))])
        self.isurverrfloor= np.where(residsobj.parlist== f'surverrfloor_{lc.filt}')[0]
        
        self.wavebasis= residsobj.wavebasis
        
        self.colorlawfunction=residsobj.colorlawfunction
        clippedphase=np.clip(self.phase,residsobj.phase.min(),residsobj.phase.max())
#########################################################################################
        #Evaluating a bunch of quantities used in the flux model
        #Evaluate derivatives of the color law, here assumed to be linear in coefficients  
        #Either evaluate on fine-grained wavelength grid, or at center of spline basis functions      
        #Calculate zero point of flux
        dwave=sn.dwave
        fluxfactor=residsobj.fluxfactor[sn.survey][lc.filt]

        wave=residsobj.wave[waveidxs]
        #Evaluate the b-spline basis functions for this passband
        #Evaluate parameters only for relevant portions of phase/wavelength space        
        inds=np.array(range(residsobj.im0.size))
        phaseind,waveind=inds//(residsobj.waveknotloc.size-residsobj.bsorder-1),inds%(residsobj.waveknotloc.size-residsobj.bsorder-1)
        inphase=((clippedphase[:,np.newaxis]>= residsobj.phaseknotloc[np.newaxis,phaseind])&(clippedphase[:,np.newaxis]<=residsobj.phaseknotloc[np.newaxis,phaseind+residsobj.bsorder+1])).any(axis=0)
        inwave=((wave.max()>=residsobj.waveknotloc[waveind])&(wave.min()<=residsobj.waveknotloc[waveind+residsobj.bsorder+1]))

        isrelevant=inphase&inwave
        #Array output indices match time along 0th axis, wavelength along 1st axis
        derivInterp=np.zeros((clippedphase.size,waveidxs.sum(),residsobj.im0.size))        
        for i in np.where(isrelevant)[0]:
                derivInterp[:,:,i] = bisplev(clippedphase ,wave,(residsobj.phaseknotloc,residsobj.waveknotloc,np.arange(residsobj.im0.size)==i, residsobj.bsorder,residsobj.bsorder))

        splinebasisconvolutions=[]
        #Redden passband transmission by MW extinction, multiply by scalar factors
        reddenedpassband=sn.mwextcurve[waveidxs]*pbspl*dwave*fluxfactor*_SCALE_FACTOR/(1+z)
    
        for pdx in range(len(lc)):
            #For photometry past the edge of the phase grid, extrapolate a linear decline in magnitudes
            if self.phase[pdx]>sn.obsphase.max():
                decayFactor= 10**(-0.4*residsobj.extrapolateDecline*(self.phase[pdx]-sn.obsphase.max()))
            else:
                decayFactor=1
            if self.preintegratebasis:
                splinebasisconvolutions+=[decayFactor*(np.dot(derivInterp[pdx,:,:].T,reddenedpassband))]
            else:
                
                splinebasisconvolutions+=[decayFactor*(derivInterp[pdx,:,:]*reddenedpassband[:,np.newaxis])]
        self.pcderivsparse=(sparse.BCOO.fromdense( np.stack(splinebasisconvolutions+ [np.zeros(splinebasisconvolutions[-1].shape) ] *padding)))
#########################################################################################
        
        #Quantities used in computation of model uncertainties
        #Color law derivatives at filter effective wavelength

        #Prefactor for variance
        self.varianceprefactor=fluxfactor*(pbspl.sum())*dwave* _SCALE_FACTOR*sn.mwextcurveint(self.lambdaeff) /(1+z)
        #Identify the relevant error model parameters
        self.errorgridshape=(residsobj.errphaseknotloc.size-1,residsobj.errwaveknotloc.size-1)
        if residsobj.errbsorder==0:
            errorwaveind=np.searchsorted(residsobj.errwaveknotloc,self.lambdaeffrest)-1
            errorphaseind=(np.searchsorted(residsobj.errphaseknotloc,clippedphase)-1)
            waveindtemp=np.array([errorwaveind for x in errorphaseind])
            ierrorbin=np.ravel_multi_index((errorphaseind,waveindtemp),self.errorgridshape)
            
            errordesignmat=scisparse.lil_matrix((len(lc)+padding,residsobj.imodelerr0.size ))
            errordesignmat[np.arange(0,len(lc)),ierrorbin ]= 1
            self.errordesignmat= sparse.BCOO.from_scipy_sparse(errordesignmat)
        else:
            inds=np.array(range(residsobj.imodelerr0.size))
            derivInterp=np.zeros((clippedphase.size,residsobj.imodelerr0.size))     
            phaseind,waveind=inds//(residsobj.errwaveknotloc.size-residsobj.errbsorder-1),inds%(residsobj.errwaveknotloc.size-residsobj.errbsorder-1)
            inphase=((clippedphase[:,np.newaxis]>= residsobj.phaseknotloc[np.newaxis,phaseind])&(clippedphase[:,np.newaxis]<=residsobj.phaseknotloc[np.newaxis,phaseind+residsobj.bsorder+1])).any(axis=0)
            inwave=(( self.lambdaeffrest>=residsobj.waveknotloc[waveind])&(self.lambdaeffrest<=residsobj.waveknotloc[waveind+residsobj.bsorder+1]))
            isrelevant=inphase&inwave
            for i in np.where(isrelevant)[0]:
                derivInterp[:,i] = bisplev(clippedphase ,self.lambdaeffrest,(residsobj.errphaseknotloc,residsobj.errwaveknotloc,np.arange(residsobj.imodelerr0.size)==i, residsobj.errbsorder,residsobj.errbsorder))
            self.errordesignmat= sparse.BCOO.fromdense(np.concatenate((derivInterp,np.zeros((padding,residsobj.imodelerr0.size)))))
        
        pow=self.iclscat.size-1-np.arange(self.iclscat.size)
        colorscateval=((self.lambdaeffrest-5500)/1000)
        
        self.clscatderivs=((colorscateval)  ** (pow)) / factorial(pow)
        for attr in lc.__slots__:
            if attr in lc.__listdatakeys__ and attr in self.__slots__:
                setattr(self,attr,np.concatenate((getattr(self,attr),np.zeros(padding))))
        self.fluxcalerr[self.ipad]=1
        
    def __len__(self):
        return self.fluxcal.size

    def modelflux(self,pars):
        if not isinstance(pars,SALTparameters):
            pars=SALTparameters(self,pars)
        #Evaluate the coefficients of the spline bases
        
        coordinates=jnp.concatenate((jnp.ones(1), pars.coordinates))

        fluxcoeffs=jnp.dot(coordinates,pars.components)*pars.x0
        #Evaluate color law at the wavelength basis centers
        colorlaw= sum([fun(c,cl,self.wavebasis) for fun,c,cl in zip(self.colorlawfunction,pars.c, pars.CL)])
        colorexp= 10. ** (  -0.4*colorlaw)

        if self.preintegratebasis:
            #Redden flux coefficients
            fluxcoeffsreddened= (colorexp[np.newaxis,:]*fluxcoeffs.reshape( self.bsplinecoeffshape)).flatten()
            #Multiply spline bases by flux coefficients
            return jnp.clip(self.pcderivsparse @ fluxcoeffsreddened,0,None)
        else:    
            #Integrate basis functions over wavelength and sum over flux coefficients
            return jnp.clip(( self.pcderivsparse @ fluxcoeffs) @ colorexp ,0,None)

    def modelfluxvariance(self,pars):
        if not isinstance(pars,SALTparameters):
            pars=SALTparameters(self,pars)

        #Exponentiate and multiply by color
        
        colorlaw= sum([fun(c,cl,self.lambdaeffrest) for fun,c,cl in zip(self.colorlawfunction,pars.c, pars.CL)])
        colorexp= 10. ** (  -0.4*colorlaw)

          #Evaluate model uncertainty

        coordinates=jnp.concatenate((jnp.ones(1), pars.coordinates ))
        
        errorsurfaces=((coordinates[:len(pars.modelerrs),np.newaxis]*pars.modelerrs)**2 ).sum(axis=0)
        for (i,j),correlation in zip(self.imodelcorrs_coordinds,pars.modelcorrs):
            errorsurfaces= errorsurfaces+2 *correlation*coordinates[i]*coordinates[j]* pars.modelerrs[i]*pars.modelerrs[j]
        errorsurfaces=self.errordesignmat @ errorsurfaces

        modelfluxvar=jnp.clip(colorexp**2 * self.varianceprefactor**2 * pars.x0**2* errorsurfaces ,0,None)
        if self.isurverrfloor.size>0:
            return jnp.hypot(modelfluxvar, pars.surverrfloor * self.modelflux(pars) )
        else:
            return modelfluxvar
            
    def colorscatter(self,pars):
        if not isinstance(pars,SALTparameters):
            pars=SALTparameters(self,pars)
        return  jnp.exp(self.clscatderivs @ pars.clscat)

    def modelloglikelihood(self, x, cachedresults=None, fixuncertainties=False, fixfluxes=False):
        if fixfluxes:
            modelflux=cachedresults
        else:
            modelflux=self.modelflux(x)

        if fixuncertainties:
            if isinstance(cachedresults,tuple):
                modelvariance,clscat=cachedresults
            else:
                modelvariance,clscat=cachedresults,0
        else:
            modelvariance=self.modelfluxvariance(x)
            clscat=self.colorscatter(x)

        variance=self.fluxcalerr**2 + modelvariance  
        zeropoint= jax.scipy.stats.norm.logpdf(self.fluxcalerr*(~self.ipad),0,self.fluxcalerr).sum()
        return mvn_likelihood_simplified_cov(self.fluxcal,modelflux,D_diag=variance,u=clscat*modelflux)-zeropoint
    
    def modelresidual(self,x,cachedresults=None,fixuncertainties=False,fixfluxes=False):
   
        if fixfluxes:
            modelflux=cachedresults
        else:
            modelflux=self.modelflux(x)

        if fixuncertainties:
            if isinstance(cachedresults,tuple):
                modelvariance,clscat=cachedresults
            else:
                modelvariance,clscat=cachedresults,0
        else:
            modelvariance=self.modelfluxvariance(x)
            clscat=self.colorscatter(x)

        variance=self.fluxcalerr**2 + modelvariance          
        numresids=(~self.ipad).sum() 
        zeropoint= ( -jnp.log(self.fluxcalerr).sum() - numresids/2)
        
        #if clscat>0, then need to use a cholesky matrix to find pulls
        def choleskyresidsandnorm( variance,clscat,modelflux):
            cholesky=jaxlinalg.cholesky(jnp.diag(variance)+ clscat**2*jnp.outer(modelflux,modelflux),lower=True)
            return {'residuals':jnp.nan_to_num(jaxlinalg.solve_triangular(cholesky, modelflux-self.fluxcal,lower=True),nan=0), 
                    'lognorm': -jnp.log(jnp.diag(cholesky)).sum()-zeropoint}
        
        return choleskyresidsandnorm(variance,clscat,modelflux)
#         return lax.cond(clscat==0, diagonalresidsandnorm, choleskyresidsandnorm, 
#              variance,clscat,modelflux )
    def dumptostring(self,pars):
        if not isinstance(pars,SALTparameters):
            pars=SALTparameters(self,pars)
        fluxes=self.modelflux(pars)
        variances=self.modelfluxvariance(pars)
        res=self.modelresidual(pars)
        for i in np.where(~self.ipad)[0]:
            yield (f"{self.uniqueid.split('_')[0]: >20} {self.uniqueid.split('_')[1]: >20} {self.phase[i]: >11.1f} {self.lambdaeffrest: >11.0f} {self.fluxcal[i]: >11.4e} {self.fluxcalerr[i]: >11.4e} {fluxes[i]: >11.4e} {np.sqrt(variances[i]): >12.4e} {res['residuals'][i]: >11.2e}")
  
  

 
@register_pytree_node_class
class modeledtrainingspectrum(modeledtrainingdata):
    __indexattributes__=[
        'ix0','ispcrcl','icomponents', 'icoordinates',
        'ipad','iCL','ic','imodelerrs','imodelcorrs','imodelcorrs_coordinds']
    __dynamicattributes__ = [
        'flux', 'wavelength','phase', 'fluxerr', 'restwavelength',
        'recaltermderivs',
        'varianceprefactor','errordesignmat',
        'pcderivsparse','spectralsuppression'
     ]
    __staticattributes__=[
        'padding',
        'bsplinecoeffshape','errorgridshape',
        'uniqueid','colorlawfunction','n_specrecal'
    ]+__indexattributes__
    __slots__ = __dynamicattributes__+__staticattributes__

    __ismapped__={
        'ix0','ic','ispcrcl','icoordinates','ipad','phase','flux','fluxerr',
        'restwavelength','recaltermderivs','pcderivsparse','errordesignmat',
       'uniqueid','n_specrecal'
    }
    
    def __init__(self,sn,spectrum,k,residsobj,padding=0):
        for attr in spectrum.__slots__:
            if attr in self.__slots__:
                    setattr(self,attr,getattr(spectrum,attr))
        z = sn.zHelio
        self.n_specrecal = spectrum.n_specrecal

        padding=max(0,padding)
        self.ix0=np.where(residsobj.parlist==f'specx0_{sn.snid}_{k}')[0][0]
        self.ispcrcl=np.where(residsobj.parlist==f'specrecal_{sn.snid}_{k}')[0]
        self.bsplinecoeffshape=(residsobj.phaseBins[0].size,residsobj.waveBins[0].size)        
        self.padding=padding
        self.ipad= np.arange(len(spectrum)+padding )>= len(spectrum)
        self.uniqueid= f'{sn.snid}_{k}'
        self.spectralsuppression=min(np.sqrt(residsobj.num_phot/residsobj.num_spec)*residsobj.spec_chi2_scaling,1)

        self.iCL=residsobj.iCL
        self.ic=sn.ic
        
        self.colorlawfunction=residsobj.colorlawfunction
        self.icomponents=residsobj.icomponents
        self.icoordinates=sn.icoordinates
        mwextcurve=sn.mwextcurveint(spectrum.wavelength)

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
        derivInterp=derivInterp*(_SCALE_FACTOR/(1+z)*mwextcurve)[:,np.newaxis]
        self.pcderivsparse=sparse.BCOO.fromdense(np.concatenate((derivInterp,np.zeros((padding,residsobj.im0.size)))))        

        pow=self.ispcrcl.size-np.arange(self.ispcrcl.size)
        recalCoord=(self.wavelength-np.mean(self.wavelength))/residsobj.specrange_wavescale_specrecal
        #recalCoord=(residsobj.waveBinCenters-jnp.mean(self.wavelength))/residsobj.specrange_wavescale_specrecal
        self.recaltermderivs=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
        self.recaltermderivs=np.concatenate((self.recaltermderivs,np.zeros((padding,pow.size))))

        self.varianceprefactor= _SCALE_FACTOR*sn.mwextcurveint(self.wavelength) /(1+z)
        self.errorgridshape=(residsobj.errphaseknotloc.size-residsobj.errbsorder-1,residsobj.errwaveknotloc.size-residsobj.errbsorder-1)
        self.varianceprefactor= np.concatenate((self.varianceprefactor, np.zeros(padding)))
        if residsobj.errbsorder==0:
            errorwaveind=np.searchsorted(residsobj.errwaveknotloc,self.restwavelength)-1
            errorphaseind=(np.searchsorted(residsobj.errphaseknotloc,self.phase)-1)
            phaseindtemp=np.tile(errorphaseind,errorwaveind.size )
            ierrorbin=np.ravel_multi_index((phaseindtemp,errorwaveind),self.errorgridshape)
            errordesignmat=scisparse.lil_matrix((len(spectrum)+padding,residsobj.imodelerr0.size ))
            errordesignmat[np.arange(0,len(spectrum)),ierrorbin ]= 1
            self.errordesignmat= sparse.BCOO.from_scipy_sparse(errordesignmat)
        else:
            inds=np.array(range(residsobj.imodelerr0.size))
            derivInterp=np.zeros((spectrum.wavelength.size,residsobj.imodelerr0.size))     
            phaseind,waveind=inds//(residsobj.errwaveknotloc.size-residsobj.errbsorder-1),inds%(residsobj.errwaveknotloc.size-residsobj.errbsorder-1)
            inphase=((self.phase>= residsobj.phaseknotloc[np.newaxis,phaseind])&(self.phase<=residsobj.phaseknotloc[np.newaxis,phaseind+residsobj.errbsorder+1])).any(axis=0)
            inwave=(( self.restwavelength.max()>=residsobj.waveknotloc[waveind])&(self.restwavelength.min()<=residsobj.waveknotloc[waveind+residsobj.errbsorder+1]))
            isrelevant=inphase&inwave
            for i in np.where(isrelevant)[0]:
                derivInterp[:,i] = bisplev(spectrum.phase,self.restwavelength,(residsobj.errphaseknotloc,residsobj.errwaveknotloc,np.arange(residsobj.imodelerr0.size)==i, residsobj.errbsorder,residsobj.errbsorder))
            self.errordesignmat= sparse.BCOO.fromdense(np.concatenate((derivInterp,np.zeros((padding,residsobj.imodelerr0.size)))))

        self.imodelerrs = np.array([(np.where(residsobj.parlist == f'specmodelerr_{i}')[0]) for i in range(residsobj.n_errorsurfaces)])
        self.imodelcorrs = np.array([( np.where(residsobj.parlist == 'specmodelcorr_{}{}'.format(i,j))[0]) for i,j in residsobj.corrcombinations])
        self.imodelcorrs_coordinds= np.array([(-1,comb[1]) if 'host'== comb[0] else ((comb[0],-1) if 'host'== comb[1] else comb) for comb in residsobj.corrcombinations])
        for attr in spectrum.__slots__:
            if attr in spectrum.__listdatakeys__ and attr in self.__slots__:
                setattr(self,attr,np.concatenate((getattr(self,attr),np.zeros(padding))))
        self.fluxerr[self.ipad]=1
        
    def __len__(self):
        return self.flux.size
                
    def modelflux(self,pars):
        if not isinstance(pars,SALTparameters):
            pars=SALTparameters(self,pars)
        x0=pars.x0
        #Define recalibration factor
        coeffs=pars.spcrcl
        coordinates=jnp.concatenate((jnp.ones(1), pars.coordinates))
        components=pars.components

        recalterm=jnp.dot(self.recaltermderivs,coeffs)
        recalterm=jnp.clip(recalterm,-recalmax,recalmax)
        recalexp=jnp.exp(recalterm)

        colorlaw= sum([fun(c,cl,self.restwavelength) for fun,c,cl in zip(self.colorlawfunction,pars.c, pars.CL)])
        colorexp= 10. ** (  -0.4*colorlaw)

        fluxcoeffs=jnp.dot(coordinates,components)*x0

        return jax.lax.cond( self.n_specrecal==0, lambda : colorexp , lambda: recalexp ) * ( self.pcderivsparse @ (fluxcoeffs))


    def modelfluxvariance(self,pars):
        if not isinstance(pars,SALTparameters):
            pars=SALTparameters(self,pars)
        x0=pars.x0
        #Define recalibration factor
        coeffs=pars.spcrcl
        coordinates=jnp.concatenate((jnp.ones(1), pars.coordinates))
        errs= pars.modelerrs

        recalterm=jnp.dot(self.recaltermderivs,coeffs)
        recalterm=jnp.clip(recalterm,-recalmax,recalmax)
        recalexp=jnp.exp(recalterm)

        #Evaluate model uncertainty
        #errorsurfaces=(coordinates,errs)**2
        errorsurfaces=((coordinates[:len(errs),np.newaxis]*errs)**2 ).sum(axis=0)
        for (i,j),correlation in zip(self.imodelcorrs_coordinds,pars.modelcorrs):
            errorsurfaces= errorsurfaces+2*correlation*coordinates[i]*coordinates[j]* errs[i]*errs[j]
        errorsurfaces=self.errordesignmat @ errorsurfaces
        modelfluxvar=recalexp**2 * self.varianceprefactor**2 * x0**2* errorsurfaces
        return jnp.clip(modelfluxvar ,0,None)


#    @partial(jaxoptions, static_argnums=[3,4],static_argnames= ['fixuncertainties','fixfluxes'],diff_argnum=1)        
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
      
        uncertainty=jax.lax.stop_gradient(jnp.sqrt(variance))
        
        numresids=(~self.ipad).sum() 
        zeropoint= ( -jnp.log(self.fluxerr).sum() - numresids/2)

        return {'residuals':  jnp.nan_to_num(self.spectralsuppression* (modelflux-self.flux)/uncertainty,nan=0),
                'lognorm': (self.spectralsuppression**2 )*(-jnp.log(uncertainty).sum()-zeropoint)}
   
    def determineneededparameters(self,modelobj):
        return []
 
        
class SALTfitcacheSN(SALTtrainingSN):
    """Class to store SN data in addition to cached results useful in speeding up the fitter
    """
    
    __slots__= ['ix0','ix1','ic', 'ixhost','icoordinates','mwextcurve',
                'mwextcurveint','dwave','obswave','obsphase','photdata',
                'specdata','zHelio','snid']
    
    def __init__(self,sndata,residsobj,kcordict,lcpaddingsizes=None,specpaddingsizes=None,n_specrecal=None):
        for attr in sndata.__slots__:
            if attr=='photdata' or  attr=='specdata':
                pass
            else:
                setattr(self,attr, getattr(sndata,attr))
        if isinstance(lcpaddingsizes,int):
            lcpaddingsizes=[lcpaddingsizes]
        if isinstance(specpaddingsizes,int):
            specpaddingsizes=[specpaddingsizes]

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
        self.ic=np.array([np.where(residsobj.parlist==f'c{i}_{self.snid}')[0][0] for i in range(residsobj.ncl)] )

        self.icoordinates=np.array([np.where(residsobj.parlist==f'x{i}_{self.snid}')[0][0] for i in range(1,residsobj.n_components)]
                +([self.ixhost] if residsobj.host_component else []))
        def choosesmallestpadsize(padsizes,datasize):
            if padsizes is None: return 0
            else: 
                padneeded= np.array(padsizes)-datasize
                padneeded= padneeded[padneeded>=0]
                try: return np.min(padneeded)
                except ValueError: raise ValueError(f'Data of length {datasize} is longer than requested zero-padded length of {max(padsizes)}')
                                
        self.photdata={flt: modeledtraininglightcurve(self,sndata.photdata[flt],residsobj,kcordict , choosesmallestpadsize( lcpaddingsizes, len(sndata.photdata[flt]) ) ) for flt in sndata.photdata}
        self.specdata={k: modeledtrainingspectrum(self,sndata.specdata[k],k,residsobj,
        choosesmallestpadsize( specpaddingsizes, len(sndata.specdata[k]) )
         ) for k in sndata.specdata }
    
    def determineneededparameters(self,modelobj):
        paramsneeded=[f'x{i}_{self.snid}' for i in range(modelobj.n_components)]+[f'c_{self.snid}']  
        for k in self.specdata.keys():
            paramsneeded+= self.specdata[k].determineneededparameters(self,modelobj)
        return paramsneeded
    
    
    @partial(jaxoptions, static_argnums=[3,4],static_argnames= ['fixuncertainties','fixfluxes'],diff_argnum=1)              
    def modelloglikelihood(self,*args,**kwargs):
        return sum([lc.modelloglikelihood(*args,**kwargs) for lc in self.photdata.values()])+sum([spec.modelloglikelihood(*args,**kwargs) for spec in self.specdata.values()])

    def dumptostring(self,pars):
        #Dumps a printout of current model and data values to a string
#         if not isinstance(pars,SALTparameters):
#             pars=SALTparameters(self,pars)
        result=''
        for data in self.photdata.values():
            result+='\n'.join(list(data.dumptostring(pars)))+'\n'
        result=result[:-1]
        return result


