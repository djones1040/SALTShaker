import numpy as np
from sncosmo.salt2utils import SALT2ColorLaw
from scipy.interpolate import interp1d
from jax import numpy as jnp

__colorlaws__=dict()
def colorlaw(colorlaw):
    """Decorator to register a given function as a valid prior"""
    __colorlaws__[colorlaw.__name__]=colorlaw
    return colorlaw

def getcolorlaw(color_name):
    return __colorlaws__[color_name]

@colorlaw
class colorlaw_default:

    def __init__(self,n_colorpars,colorwaverange):
        self.n_colorpars=n_colorpars
        self.colorwaverange=colorwaverange

    def __call__(self, color,colorlawparams,wave):
        wave=jnp.atleast_1d(wave)
        colorlawderiv=jnp.empty((wave.size,self.n_colorpars))
        for i in range(self.n_colorpars):
            colorlawderiv[:,i]=\
                SALT2ColorLaw(self.colorwaverange,  np.arange(self.n_colorpars)==i)(wave)-\
                SALT2ColorLaw(self.colorwaverange,  np.zeros(self.n_colorpars))(wave)

        colorlawzero=SALT2ColorLaw(self.colorwaverange, np.zeros(self.n_colorpars))(wave)
        return color*(jnp.dot(colorlawderiv,colorlawparams)+colorlawzero)
    
        
@colorlaw
class colorlaw_intrinsic_plus_dust:
    pass

@colorlaw    
class colorlaw_spare:
    pass