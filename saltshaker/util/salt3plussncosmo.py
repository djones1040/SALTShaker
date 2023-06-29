import abc
import os
from copy import copy as cp
from math import ceil
from textwrap import dedent

import extinction
import numpy as np
from astropy import (cosmology, units as u)
from astropy.utils.misc import isiterable
from scipy.interpolate import (
    InterpolatedUnivariateSpline as Spline1d,
    RectBivariateSpline as Spline2d
)

from sncosmo._registry import Registry
from sncosmo.bandpasses import Bandpass, get_bandpass
from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
from sncosmo.io import (
    read_griddata_ascii, read_griddata_fits,
    read_multivector_griddata_ascii
)
from sncosmo.magsystems import get_magsystem
from sncosmo.salt2utils import BicubicInterpolator, SALT2ColorLaw
from sncosmo.utils import integration_grid
from sncosmo.models import Source

class SALT3PlusSource(Source):

    _param_names = ['x0', 'x1','x2', 'c']
    param_names_latex = ['x_0', 'x_1','x_2' ,'c']
    _SCALE_FACTOR = 1e-12

    def __init__(self, modeldir=None,
                 m0file='',
                 m1file='salt3_template_1.dat',
                 clfile='salt3_color_correction.dat',
                 cdfile='salt3_color_dispersion.dat',
                 lcrv00file='salt3_lc_variance_0.dat',
                 lcrv11file='salt3_lc_variance_1.dat',
                 lcrv01file='salt3_lc_covariance_01.dat',
                 name=None, version=None):

        self.name = name
        self.version = version
        self._model = {}
        self._parameters = np.array([1.,0, 0., 0.])        
        
        names_or_objs = {
                         'LCRV00': lcrv00file, 'LCRV11': lcrv11file,
                         'LCRV01': lcrv01file,
                         'cdfile': cdfile, 'clfile': clfile}

        # Make filenames into full paths.
        if modeldir is not None:
            for k in names_or_objs:
                v = names_or_objs[k]
                if (v is not None and isinstance(v, str)):
                    names_or_objs[k] = os.path.join(modeldir, v)
            i=0
            components={}
            while os.path.exists(os.path.join(modeldir,f'salt3_template_{i}.dat')):
                components[f'M{i}']=os.path.join(modeldir,f'salt3_template_{i}.dat')
                i+=1
        else:
            i=0
            components={}
            while os.path.exists(f'salt3_template_{i}.dat'):
                components[f'M{i}']=f'salt3_template_{i}.dat'
                i+=1
        self.ncomponents=len(components)
        # model components are interpolated to 2nd order
        for key in components:
            phase, wave, values = read_griddata_ascii(components[key])
            values *= self._SCALE_FACTOR
            self._model[key] = BicubicInterpolator(phase, wave, values)

            # The "native" phases and wavelengths of the model are those
            # of the first model component.
            if key == 'M0':
                self._phase = phase
                self._wave = wave

        # model covariance is interpolated to 1st order
        for key in ['LCRV00', 'LCRV11', 'LCRV01']:
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
            values *= self._SCALE_FACTOR**2.
            self._model[key] = BicubicInterpolator(phase, wave, values)

        # Set the colorlaw based on the "color correction" file.
        self._set_colorlaw_from_file(names_or_objs['clfile'])

        # Set the color dispersion from "color_dispersion" file
        w, val = np.loadtxt(names_or_objs['cdfile'], unpack=True)
        self._colordisp = Spline1d(w, val,  k=1)  # linear interp.

    def _flux(self, phase, wave):
        m0 = self._model['M0'](phase, wave)
        m1 = self._model['M1'](phase, wave)
        m2 = self._model['M2'](phase, wave)
        return np.clip(self._parameters[0] * (m0 + self._parameters[1] * m1 + self._parameters[2] * m2) *
                10. ** (-0.4 * self._colorlaw(wave) * self._parameters[-1]),0,None)

    def _bandflux_rvar_single(self, band, phase):
        """Model relative variance for a single bandpass."""

        # Raise an exception if bandpass is out of model range.
        if (band.minwave() < self._wave[0] or band.maxwave() > self._wave[-1]):
            raise ValueError('bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                             'outside spectral range [{3:.6g}, .., {4:.6g}]'
                             .format(band.name, band.wave[0], band.wave[-1],
                                     self._wave[0], self._wave[-1]))

        x1 = self._parameters[1]

        # integrate m0 and m1 components
        wave, dwave = integration_grid(band.minwave(), band.maxwave(),
                                       MODEL_BANDFLUX_SPACING)
        trans = band(wave)
        m0 = self._model['M0'](phase, wave)
        m1 = self._model['M1'](phase, wave)
        tmp = trans * wave

        # evaluate avg M0 + x1*M1 across a bandpass
        f0 = np.sum(m0 * tmp, axis=1)/tmp.sum()
        m1int = np.sum(m1 * tmp, axis=1)/tmp.sum()
        ftot = f0 + x1 * m1int

        # In the following, the "[:,0]" reduces from a 2-d array of shape
        # (nphase, 1) to a 1-d array.
        lcrv00 = self._model['LCRV00'](phase, band.wave_eff)[:, 0]
        lcrv11 = self._model['LCRV11'](phase, band.wave_eff)[:, 0]
        lcrv01 = self._model['LCRV01'](phase, band.wave_eff)[:, 0]

        # variance in M0 + x1*M1 at the effective wavelength
        # of a bandpass
        v = lcrv00 + 2.0 * x1 * lcrv01 + x1 * x1 * lcrv11

        # v is supposed to be variance but can go negative
        # due to interpolation.  Correct negative values to some small
        # number. (at present, use prescription of snfit : set
        # negatives to 0.0001)
        v[v < 0.0] = 0.0001

        # avoid warnings due to evaluating 0. / 0. in f0 / ftot
        with np.errstate(invalid='ignore'):
            # turn M0+x1*M1 error into a relative error
            result = v/ftot**2.

        # treat cases where ftot is negative the same as snfit
        result[ftot <= 0.0] = 10000.
        return result

    def bandflux_rcov(self, band, phase):
        """Return the *relative* model covariance (or "model error") on
        synthetic photometry generated from the model in the given restframe
        band(s).

        This model covariance represents the scatter of real SNe about
        the model.  The covariance matrix has two components. The
        first component is diagonal (pure variance) and depends on the
        phase :math:`t` and bandpass central wavelength
        :math:`\\lambda_c` of each photometry point:

        .. math::

           (F_{0, \\mathrm{band}}(t) / F_{1, \\mathrm{band}}(t))^2
           S(t, \\lambda_c)^2
           (V_{00}(t, \\lambda_c) + 2 x_1 V_{01}(t, \\lambda_c) +
            x_1^2 V_{11}(t, \\lambda_c))

        where the 2-d functions :math:`S`, :math:`V_{00}`, :math:`V_{01}`,
        and :math:`V_{11}` are given by the files ``errscalefile``,
        ``lcrv00file``, ``lcrv01file``, and ``lcrv11file``
        respectively and :math:`F_0` and :math:`F_1` are given by

        .. math::

           F_{0, \\mathrm{band}}(t) = \\int_\\lambda M_0(t, \\lambda)
                                      T_\\mathrm{band}(\\lambda)
                                      \\frac{\\lambda}{hc} d\\lambda

        .. math::

           F_{1, \\mathrm{band}}(t) = \\int_\\lambda
                                      (M_0(t, \\lambda) + x_1 M_1(t, \\lambda))
                                      T_\\mathrm{band}(\\lambda)
                                      \\frac{\\lambda}{hc} d\\lambda

        As this first component can sometimes be negative due to
        interpolation, there is a floor applied wherein values less than zero
        are set to ``0.01**2``. This is to match the behavior of the
        original SALT2 code, snfit.

        The second component is block diagonal. It has
        constant covariance between all photometry points within a
        bandpass (regardless of phase), and no covariance between
        photometry points in different bandpasses:

        .. math::

           CD(\\lambda_c)^2

        where the 1-d function :math:`CD` is given by the file ``cdfile``.
        Adding these two components gives the *relative* covariance on model
        photometry.

        Parameters
        ----------
        band : `~numpy.ndarray` of `~sncosmo.Bandpass`
            Bandpasses of observations.
        phase : `~numpy.ndarray` (float)
            Phases of observations.


        Returns
        -------
        rcov : `~numpy.ndarray`
            Model relative covariance for given bandpasses and phases.
        """

        # construct covariance array with relative variance on diagonal
        diagonal = np.zeros(phase.shape, dtype=np.float64)
        for b in set(band):
            mask = band == b
            diagonal[mask] = self._bandflux_rvar_single(b, phase[mask])
        result = np.diagflat(diagonal)

        # add kcorr errors
        for b in set(band):
            mask1d = band == b
            mask2d = mask1d * mask1d[:, None]  # mask for result array
            kcorrerr = self._colordisp(b.wave_eff)
            result[mask2d] += kcorrerr**2

        return result

    def _set_colorlaw_from_file(self, name_or_obj):
        """Read color law file and set the internal colorlaw function."""

        # Read file
        if isinstance(name_or_obj, str):
            f = open(name_or_obj, 'r')
        else:
            f = name_or_obj
        words = f.read().split()
        f.close()

        # Get colorlaw coeffecients.
        ncoeffs = int(words[0])
        colorlaw_coeffs = [float(word) for word in words[1: 1 + ncoeffs]]

        # If there are more than 1+ncoeffs words in the file, we expect them to
        # be of the form `keyword value`.
        version = None
        colorlaw_range = [3000., 7000.]
        for i in range(1+ncoeffs, len(words), 2):
            if words[i] == 'Salt2ExtinctionLaw.version':
                version = int(words[i+1])
            elif words[i] == 'Salt2ExtinctionLaw.min_lambda':
                colorlaw_range[0] = float(words[i+1])
            elif words[i] == 'Salt2ExtinctionLaw.max_lambda':
                colorlaw_range[1] = float(words[i+1])
            else:
                raise RuntimeError("Unexpected keyword: {}".format(words[i]))

        # Set extinction function to use.
        if version == 0:
            raise RuntimeError("Salt2ExtinctionLaw.version 0 not supported.")
        elif version == 1:
            self._colorlaw = SALT2ColorLaw(colorlaw_range, colorlaw_coeffs)
        else:
            raise RuntimeError('unrecognized Salt2ExtinctionLaw.version: ' +
                               version)

    def colorlaw(self, wave=None):
        """Return the value of the CL function for the given wavelengths.

        Parameters
        ----------
        wave : float or list_like

        Returns
        -------
        colorlaw : float or `~numpy.ndarray`
            Values of colorlaw function, which can be interpreted as extinction
            in magnitudes.
        """

        if wave is None:
            wave = self._wave
        else:
            wave = np.asarray(wave)
        if wave.ndim == 0:
            return self._colorlaw(np.ravel(wave))[0]
        else:
            return self._colorlaw(wave)

