import numpy as np
from scipy.interpolate import interp1d
from jax import numpy as jnp

import jax

__colorlaws__=dict()
def colorlaw(colorlaw):
    """Decorator to register a given function as a valid prior"""
    __colorlaws__[colorlaw.__name__]=colorlaw
    return colorlaw

def getcolorlaw(color_name):
    return __colorlaws__[color_name]

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

        self.coeffs = jnp.concatenate((coeffs[::-1],jnp.array([1.0-jnp.sum(coeffs),0.])))
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
        wave=jnp.atleast_1d(wave)
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

    def __init__(self,n_colorpars,colorwaverange):
        self.n_colorpars=n_colorpars
        self.colorwaverange=colorwaverange

    def __call__(self, color,colorlawparams,wave):
        return color*(SALT2ColorLaw(self.colorwaverange, colorlawparams)(wave))
    
        
@colorlaw
class colorlaw_intrinsic_plus_dust:
    pass

@colorlaw    
class colorlaw_spare:
    pass