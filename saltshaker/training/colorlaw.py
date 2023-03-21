import numpy as np
from sncosmo.salt2utils import SALT2ColorLaw
from scipy.interpolate import interp1d
from jax import numpy as jnp
import jax

__colorlaws__=dict()
def colorlaw(colorlaw):
    """Decorator to register a given function as a valid prior"""
    __colorlaws__[colorlaw.__name__]=colorlaw
    return colorlaw

SALT2CL_B = 4302.57
SALT2CL_V = 5428.55
SALT2CL_V_MINUS_B = SALT2CL_V - SALT2CL_B

class SALT2ColorLaw:
    def __init__(self,wave_range,coeffs):
        
        if len(coeffs) > 6:
            raise ValueError("number of coefficients must be equal to or "
                             "less than 6.")

        wave_lo, wave_hi = wave_range
        self.l_lo = (wave_lo - SALT2CL_B) / SALT2CL_V_MINUS_B
        self.l_hi = (wave_hi - SALT2CL_B) / SALT2CL_V_MINUS_B

        self.coeffs = jnp.concatenate((coeffs[::-1],jnp.array([1.0-jnp.sum(coeffs)]),jnp.array([0.])))
        self.ncoeffs = len(self.coeffs)

        # precompute value of
        # P(l) = c[0]*l + c[1]*l^2 + c[2]*l^3 + ...  at l_lo and l_hi
        self.p_lo = jnp.polyval(self.coeffs, self.l_lo)
        self.p_hi = jnp.polyval(self.coeffs, self.l_hi)

        # precompute derivative of P(l) at l_lo and l_hi
        fun = lambda x: jax.grad(jnp.polyval,argnums=1)(self.coeffs,x)
        self.pprime_lo = fun(self.l_lo)
        self.pprime_hi = fun(self.l_hi)
        
    def __call__(self,wave):

        #for i in range(n):
        l = (wave - SALT2CL_B) / SALT2CL_V_MINUS_B

        # Blue side
        return -1* jnp.select(
            [l < self.l_lo,l <= self.l_hi,l > self.l_hi],
            [self.p_lo + self.pprime_lo * (l - self.l_lo),
             jnp.polyval(self.coeffs, l),
             self.p_hi + self.pprime_hi * (l - self.l_hi)])


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
    
