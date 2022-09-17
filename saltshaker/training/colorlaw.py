import numpy as np
from sncosmo.salt2utils import SALT2ColorLaw
from scipy.interpolate import interp1d

__colorlaws__=dict()
def colorlaw(colorlaw):
    """Decorator to register a given function as a valid prior"""
    __colorlaws__[colorlaw.__name__]=colorlaw
    return colorlaw

@colorlaw
class colorlaw_default:

    def __init__(self,n_colorpars,colorwaverange,wave,interpMethod):

        self.wave = wave
        self.colorwaverange = colorwaverange
        self.n_colorpars = n_colorpars
        self.interpMethod = interpMethod
        
        # color law derivative - user supplied
        self.colorLawDeriv=np.empty((self.wave.size,n_colorpars))
        for i in range(self.n_colorpars):
            self.colorLawDeriv[:,i]=\
                SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)(self.wave)-\
                SALT2ColorLaw(self.colorwaverange, np.zeros(self.n_colorpars))(self.wave)

        ## interpolated color law deriv
        self.colorLawDerivInterp=interp1d(
            self.wave,self.colorLawDeriv,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)

    # color law - user supplied
    # this will be removed
    def colorlaw(self,clpars):
        return SALT2ColorLaw(self.colorwaverange, clpars)(self.wave)

    # reddening - 10**colorlaw for nominal case
    # user supplied
    def reddening(self,c,colorLaw):

        reddening= 10. ** (colorLaw * c)
        
        return reddening

@colorlaw
class colorlaw_intrinsic_plus_dust:

    def colorexp(c,colorlaw):

        pass

    def colorderiv(modulatedFlux,colorlaw):

        pass

    def colorlawderiv(modulatedFlux,colorLawDeriv,varyParams,iCL,c):

        pass

@colorlaw    
class colorlaw_spare:

    def colorexp(c,colorlaw):

        pass

    def colorderiv(modulatedFlux,colorlaw):

        pass

    def colorlawderiv(modulatedFlux,colorLawDeriv,varyParams,iCL,c):

        pass
    
