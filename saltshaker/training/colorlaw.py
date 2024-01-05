##import numpy as np
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

def k_Calzetti(wav,RV):
    ''' Calzetti model (Calzetti et al. 2000)
    https://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C/abstract
    inputs:
        wav: wavelength to be evaluated (in Angstroms)
        RV: slope
    output:
        kCal: A numpy array with ratio of attenuation A_lambda/E(B-V)
    '''

    # sanity check
    # DJ removed because jax is annoying
    #Cal_wav_max = 22000
    #Cal_wav_min = 1200
    #if not (wav.min() >= Cal_wav_min) & (wav.max() <= Cal_wav_max):
    #    msg_err = f'wav range = ({wav.min()}, {wav.max()}) outside the defined Calzetti CL range of {Cal_wav_min} < wav < {Cal_wav_max}'
    #    assert False, msg_err
        
    # function is defined for wav in um
    wav_um  = wav/1e4 
    wav_transition_um = 6300/1e4
    
    # evaluate polynomial
    kCal = jnp.select(
        [wav_um >= wav_transition_um/1e4,wav_um < wav_transition_um/1e4],
        [2.659 * (-1.857 + 1.040/wav_um) + RV,
         2.659 * (-2.156 + 1.509/wav_um - 0.198/wav_um**2 + 0.011/wav_um**3) + RV])
    
    return kCal

def k_Salim(wave,RV):
    raise NotImplementedError
    
class GalacticDustLaw:
    def __init__(self,
                 RV=3.1,dust_model='Calzetti'):

        self.RV = RV

        # select galactic model
        gCL_MODELS = {
            'Calzetti':k_Calzetti,
            'Salim':k_Salim
        }
        self.gCL_model = gCL_MODELS[dust_model]

    def __call__(self,wave):
        ''' Designed to be used for SALT3 color law model
        inputs:
            wave: array or scalar of wavelength to be evaluated (in Angstrom)
            RV: R_V (attenuation slope)
        output:
            color law values
        '''

        # handle both scalar and array
        wave_arr = jnp.atleast_1d(wave)

        # evaluate Calzetti CL at wav, B, and V
        k_lambda = self.gCL_model(wave_arr,self.RV)
        k_B = self.gCL_model(jnp.asarray([SALT2CL_B]),self.RV)
        k_V = self.gCL_model(jnp.asarray([SALT2CL_V]),self.RV)

        # rescale so that CL(B)=0, CL(V)=-1
        # convert back to scalar if wav is scalar
        CL = jnp.squeeze((k_lambda - k_B) / (k_B - k_V))

        return CL

@colorlaw
class colorlaw_default:

    def __init__(self,n_colorpars,colorwaverange):
        self.n_colorpars=n_colorpars
        self.colorwaverange=colorwaverange

    def __call__(self, color,colorlawparams,wave):
        return color*(SALT2ColorLaw(self.colorwaverange, colorlawparams)(wave))
    
@colorlaw
class colorlaw_separatecolors:

    def __init__(self,n_colorpars,colorwaverange):
        self.n_colorpars=n_colorpars
        assert(self.n_colorpars%2==0)
        self.n_colorpars=self.n_colorpars//2
        self.colorwaverange=colorwaverange
        self.colorlaws=[colorlaw_default(self.n_colorpars,self.colorwaverange  ),
        colorlaw_default(self.n_colorpars,self.colorwaverange  )]
        

    def __call__(self, color,colorlawparams,wave):
        return jax.lax.cond( color>0, self.colorlaws[0], self.colorlaws[1],
            color,colorlawparams,wave
        )


@colorlaw
class colorlaw_intrinsic_plus_dust:

    def __init__(self,n_colorpars,colorwaverange):
        self.n_colorpars=n_colorpars
        self.colorwaverange=colorwaverange
        self.c_coeffs=[0.0727, 0.57, 1.58]
        
    def __call__(self, color,colorlawparams,wave):

        c_g = self.c_coeffs[0] + self.c_coeffs[1]*color + self.c_coeffs[2]*color**2
        c_i = color - c_g

        # compute colorlaw for each component
        iCL = SALT2ColorLaw(self.colorwaverange, colorlawparams)(wave)
        gCL = GalacticDustLaw()(wave)

        # add two
        return c_i * iCL +  c_g * gCL

@colorlaw
class colorlaw_galactic:

    def __init__(self,n_colorpars,colorwaverange):
        self.n_colorpars=n_colorpars
        self.colorwaverange=colorwaverange
        
    def __call__(self, color,colorlawparams,wave):

        gCL = GalacticDustLaw()(wave)
        # need a minus sign here to match default colorlaw
        return -color*gCL


@colorlaw    
class colorlaw_spare:
    pass
