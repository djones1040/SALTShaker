from saltshaker.util.readutils import SALTtrainingSN,SALTtraininglightcurve,SALTtrainingspectrum
from saltshaker.util.sparsealgebra import SparseVector,SparseMatrix

from sncosmo.salt2utils import SALT2ColorLaw

from scipy.special import factorial
from scipy.interpolate import splprep,splev,bisplev,bisplrep,interp1d,interp2d,RegularGridInterpolator,RectBivariateSpline

import numpy as np

import autograd as ag
from autograd import numpy as agnp

import extinction
import copy
import warnings
import logging
log=logging.getLogger(__name__)

warnings.simplefilter('ignore',category=FutureWarning)

from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING

_SCALE_FACTOR = 1e-12

def __anyinnonzeroareaforsplinebasis__(phase,wave,phaseknotloc,waveknotloc,bsorder,i):
    phaseindex,waveindex=i//(waveknotloc.size-bsorder-1), i% (waveknotloc.size-bsorder-1)
    return ((phase>=phaseknotloc[phaseindex])&(phase<=phaseknotloc[phaseindex+bsorder+1])).any() and ((wave>=waveknotloc[waveindex])&(wave<=waveknotloc[waveindex+bsorder+1])).any() 


class SALTfitcachelightcurve(SALTtraininglightcurve):

    __slots__ = ['idx','pbspl','denom','lambdaeff','bsplinecoeffshape','colorlawderiv'
    ,'im0','im1','iCL','ix0','ix1','ic','z','splinebasisconvolutions',
    'lambdaeffrest','varianceprefactor',
    'errorgridshape',
    'imodelcorr01','imodelerr0','imodelerr1']
    

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
        
        self.pbspl = pbspl[np.newaxis,:]
        self.denom = denom
        self.lambdaeff = kcordict[sn.survey][lc.filt]['lambdaeff']    
        self.lambdaeffrest= self.lambdaeff/(1+z)    
        self.bsplinecoeffshape=(residsobj.phaseBins[0].size,residsobj.waveBins[0].size)        
        self.z=z

        self.ix1=sn.ix1
        self.icoordinates=[sn.ix1]
        self.icomponents=[residsobj.im0,residsobj.im1]
        if residsobj.host_component:
            self.icoordinates+=[sn.ixhost]
            self.icomponents+=[residsobj.imhost]
        
        self.iCL=residsobj.iCL
        self.ix0=sn.ix0
        self.ic=sn.ic

        clippedphase=np.clip(self.phase,residsobj.phase.min(),residsobj.phase.max())
#########################################################################################
        #Evaluating a bunch of quantities used in the flux model
        #Evaluate derivatives of the color law, here assumed to be linear in coefficients  
        #Either evaluate on fine-grained wavelength grid, or at center of spline basis functions      
        if preintegratebasis:
            colorlawwaveeval=residsobj.waveBinCenters
        else:
            colorlawwaveeval=residsobj.wave
        
        self.colorlawderiv=np.empty((colorlawwaveeval.size,self.iCL.size))
        for i in range(self.iCL.size):
            self.colorlawderiv[:,i]=\
                SALT2ColorLaw(residsobj.colorwaverange, np.arange(self.iCL.size)==i)(colorlawwaveeval)-\
                SALT2ColorLaw(residsobj.colorwaverange, np.zeros(self.iCL.size))(colorlawwaveeval)
    
        #Calculate zero point of flux
        dwave=sn.dwave
        fluxfactor=residsobj.fluxfactor[sn.survey][self.filt]


        #Evaluate the b-spline basis functions for this passband
        #Evaluate parameters only for relevant portions of phase/wavelength space        
        inds=np.array(range(residsobj.im0.size))
        phaseind,waveind=inds//(residsobj.waveknotloc.size-residsobj.bsorder-1),inds%(residsobj.waveknotloc.size-residsobj.bsorder-1)
        inphase=((clippedphase[:,np.newaxis]>= residsobj.phaseknotloc[np.newaxis,phaseind])&(clippedphase[:,np.newaxis]<=residsobj.phaseknotloc[np.newaxis,phaseind+residsobj.bsorder+1])).any(axis=0)
        inwave=~ ((residsobj.wave[self.idx].min() >= residsobj.waveknotloc[waveind]) | (residsobj.wave[self.idx].max() <= residsobj.waveknotloc[waveind+residsobj.bsorder+1]))
        isrelevant=inphase&inwave
    
        #Array output indices match time along 0th axis, wavelength along 1st axis
        derivInterp=np.zeros((clippedphase.size,self.idx.sum(),residsobj.im0.size))        
        for i in np.where(isrelevant)[0]:
            if __anyinnonzeroareaforsplinebasis__(clippedphase,residsobj.wave[self.idx],residsobj.phaseknotloc,residsobj.waveknotloc,residsobj.bsorder,i):
                derivInterp[:,:,i] = bisplev(clippedphase ,residsobj.wave[self.idx],(residsobj.phaseknotloc,residsobj.waveknotloc,np.arange(residsobj.im0.size)==i, residsobj.bsorder,residsobj.bsorder))

        self.splinebasisconvolutions=[]
        #Redden passband transmission by MW extinction, multiply by scalar factors
        reddenedpassband=sn.mwextcurve[self.idx]*self.pbspl[0]*dwave*fluxfactor*_SCALE_FACTOR/(1+self.z)
    
        for pdx in range(len(lc)):
            #For photometry past the edge of the phase grid, extrapolate a linear decline in magnitudes
            if self.phase[pdx]>sn.obsphase.max():
                decayFactor= 10**(-0.4*residsobj.extrapolateDecline*(self.phase[pdx]-sn.obsphase.max()))
            else:
                decayFactor=1
            if preintegratebasis:
                self.splinebasisconvolutions+=[SparseVector(np.dot(derivInterp[pdx,:,:].T,reddenedpassband)*decayFactor)]
            else:
                self.splinebasisconvolutions+=[SparseMatrix((derivInterp[pdx,:,:]*reddenedpassband[:,np.newaxis])*decayFactor)]
        
#         self.fluxdependentparameters=(ag.jacobian(self.modelflux)(residsobj.initparams)!=0).sum(axis=0)>0

#########################################################################################
        
        #Quantities used in computation of model uncertainties
        #Color law derivatives at filter effective wavelength
        self.colorlawderivlambdaeff=np.array([
        SALT2ColorLaw(residsobj.colorwaverange, np.arange(self.iCL.size)==i)(np.array([self.lambdaeffrest]))-\
                SALT2ColorLaw(residsobj.colorwaverange, np.zeros(self.iCL.size))(np.array([self.lambdaeffrest])) for i in range(self.iCL.size)])[:,0]
        #Prefactor for variance
        self.varianceprefactor=fluxfactor*(self.pbspl.sum())*dwave* _SCALE_FACTOR*sn.mwextcurveint(self.lambdaeff) /(1+self.z)
        
        #Identify the relevant error model parameters
        errorwaveind=np.searchsorted(residsobj.errwaveknotloc,self.lambdaeffrest)-1
        errorphaseind=(np.searchsorted(residsobj.errphaseknotloc,clippedphase)-1)
        self.errorgridshape=(residsobj.errphaseknotloc.size-1,residsobj.errwaveknotloc.size-1)
        waveindtemp=np.array([errorwaveind for x in errorphaseind])
        ierrorbin=np.ravel_multi_index((errorphaseind,waveindtemp),self.errorgridshape)
        self.imodelcorr01=residsobj.imodelcorr01[ierrorbin]    
        self.imodelerr0=residsobj.imodelerr0[ierrorbin]    
        self.imodelerr1=residsobj.imodelerr1[ierrorbin]    

             
    def modelflux(self,pars):
        #Define parameters
        x0,c=pars[[self.ix0,self.ic]]
        #Evaluate the coefficients of the spline bases
        
        coordinates=agnp.array([1]+list(pars[self.icoordinates]))
        components=pars[self.icomponents]
        fluxcoeffs=agnp.dot(coordinates,components)*x0
        
        #Evaluate color law at the wavelength basis centers
        colorlaw=agnp.dot(self.colorlawderiv,pars[self.iCL])
        #Exponentiate and multiply by color
        colorexp= 10. ** (  -0.4*colorlaw* c)
        if preintegratebasis:
            #Redden flux coefficients
            fluxcoeffsreddened= (colorexp[np.newaxis,:]*fluxcoeffs.reshape( self.bsplinecoeffshape)).flatten()
            #Multiply spline bases by flux coefficients
            return agnp.array([design.dot(fluxcoeffsreddened) for design in self.splinebasisconvolutions])     
        else:    
            #Integrate basis functions over wavelength and sum over flux coefficients
            return agnp.array([design.multidot(fluxcoeffs,colorexp) for design in self.splinebasisconvolutions])     
    
    def modelfluxvariance(self,pars):
        x0,x1,c=pars[[self.ix0,self.ix1,self.ic]]
        #Evaluate color law at the wavelength basis centers
        colorlaw=agnp.dot(self.colorlawderivlambdaeff,pars[self.iCL])
        #Exponentiate and multiply by color
        colorexp= 10. ** (  -0.4*colorlaw* c)
        
        #Evaluate model uncertainty
        corr01,modelerr0,modelerr1= pars[[ self.imodelcorr01, self.imodelerr0,self.imodelerr1]]
        
        modelfluxvar=colorexp**2 * self.varianceprefactor**2 * x0**2* (modelerr0**2 +2*x1*corr01*modelerr1* modelerr0+  (x1*modelerr1)**2)
        return agnp.clip(modelfluxvar,0,None)
 
 
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
        
        self.ix1=sn.ix1
        self.iCL=residsobj.iCL
        self.ic=sn.ic
        self.ispecx0=np.where(residsobj.parlist=='specx0_{}_{}'.format(sn.snid,k))[0][0]
        self.ispcrcl=np.where(residsobj.parlist=='specrecal_{}_{}'.format(sn.snid,k))[0]
        self.bsplinecoeffshape=(residsobj.phaseBins[0].size,residsobj.waveBins[0].size)        

        self.icoordinates=[sn.ix1]
        self.icomponents=[residsobj.im0,residsobj.im1]
        if residsobj.host_component:
            self.icoordinates+=[sn.ixhost]
            self.icomponents+=[residsobj.imhost]

        self.mwextcurve=sn.mwextcurveint(spectrum.wavelength)
        
        derivInterp=np.zeros((spectrum.wavelength.size,residsobj.im0.size))
        for i in range(residsobj.im0.size):
            if __anyinnonzeroareaforsplinebasis__(spectrum.phase,spectrum.wavelength/(1+z),residsobj.phaseknotloc,residsobj.waveknotloc,residsobj.bsorder,i):
                derivInterp[:,i] = bisplev(spectrum.phase,spectrum.wavelength/(1+z),(residsobj.phaseknotloc,residsobj.waveknotloc,np.arange(residsobj.im0.size)==i, residsobj.bsorder,residsobj.bsorder))
        derivInterp=derivInterp*(_SCALE_FACTOR/(1+self.z)*self.mwextcurve)[:,np.newaxis]
        self.pcderivsparse=SparseMatrix(derivInterp)
        self.spectrumid=k
        

        pow=self.ispcrcl.size-agnp.arange(self.ispcrcl.size)
        recalCoord=(self.wavelength-agnp.mean(self.wavelength))/residsobj.specrange_wavescale_specrecal
        #recalCoord=(residsobj.waveBinCenters-agnp.mean(self.wavelength))/residsobj.specrange_wavescale_specrecal
        self.recaltermderivs=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
        

        self.varianceprefactor= _SCALE_FACTOR*sn.mwextcurveint(self.wavelength) /(1+self.z)
        
        errorwaveind=np.searchsorted(residsobj.errwaveknotloc,self.restwavelength)-1
        errorphaseind=(np.searchsorted(residsobj.errphaseknotloc,self.phase)-1)
        self.errorgridshape=(residsobj.errphaseknotloc.size-1,residsobj.errwaveknotloc.size-1)
        phaseindtemp=np.tile(errorphaseind,errorwaveind.size )
        ierrorbin=np.ravel_multi_index((phaseindtemp,errorwaveind),self.errorgridshape)
        self.imodelcorr01=residsobj.imodelcorr01[ierrorbin]    
        self.imodelerr0=residsobj.imodelerr0[ierrorbin]    
        self.imodelerr1=residsobj.imodelerr1[ierrorbin]    


    def modelflux(self,pars):
        x0=pars[self.ispecx0]
        #Define recalibration factor
        coeffs=pars[self.ispcrcl]
        recalterm=agnp.dot(self.recaltermderivs,coeffs)
        pastbounds=agnp.abs(recalterm)>100
        recalterm=agnp.clip(recalterm,-100,100)
        recalexp=agnp.exp(recalterm)
        
        coordinates=agnp.array([1]+list(pars[self.icoordinates]))
        components=pars[self.icomponents]
        
        fluxcoeffs=agnp.dot(coordinates,components)*x0

        return recalexp*self.pcderivsparse.dot(fluxcoeffs,returnsparse=False)
        #fluxcoeffsreddened= (recalexp[np.newaxis,:]*fluxcoeffs.reshape( self.bsplinecoeffshape)).flatten()
        #return self.pcderivsparse.dot(fluxcoeffsreddened,returnsparse=False)

    def modelfluxvariance(self,pars):
        x0,x1=pars[[self.ispecx0,self.ix1]]
        #Define recalibration factor
        coeffs=pars[self.ispcrcl]
        recalterm=agnp.dot(self.recaltermderivs,coeffs)
        pastbounds=agnp.abs(recalterm)>100
        recalterm=agnp.clip(recalterm,-100,100)
        recalexp=agnp.exp(recalterm)
        
        #Evaluate model uncertainty
        corr01,modelerr0,modelerr1= pars[[ self.imodelcorr01, self.imodelerr0,self.imodelerr1]]
        
        modelfluxvar=recalexp**2 * self.varianceprefactor**2 * x0**2* (modelerr0**2 +2*x1*corr01*modelerr1* modelerr0+  (x1*modelerr1)**2)
        return agnp.clip(modelfluxvar,0,None)


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
        
        self.photdata={flt: SALTfitcachelightcurve(self,sndata.photdata[flt],residsobj,kcordict ) for flt in sndata.photdata}
        self.specdata={k: SALTfitcachespectrum(self,sndata.specdata[k],k,residsobj ) for k in sndata.specdata }
