"""
SALT3 source model for sncosmo.

This module provides the SALT3Source class, an sncosmo-compatible
implementation of the SALT3 Type Ia supernova spectral timeseries model.
Used for light curve fitting and synthetic photometry generation.

Classes
-------
SALT3Source
    SALT3 spectral timeseries model for sncosmo.
SplineColorLaw
    Natural cubic spline color law.

Functions
---------
colorlawfromfile
    Read a color law from a color correction file.
"""
import abc
import os
from collections import OrderedDict as odict
from copy import copy as cp
from textwrap import dedent
from functools import partial
from math import ceil
import itertools

import numpy as np
from scipy.interpolate import (InterpolatedUnivariateSpline as Spline1d,
                               RectBivariateSpline as Spline2d)
from astropy.utils.misc import isiterable
from astropy import (cosmology, units as u, constants as const)
import extinction

from sncosmo.io import (read_griddata_ascii, read_griddata_fits,
                 read_multivector_griddata_ascii)
from sncosmo._registry import Registry
from sncosmo.bandpasses import get_bandpass, Bandpass
from sncosmo.magsystems import get_magsystem
from sncosmo.salt2utils import BicubicInterpolator, SALT2ColorLaw
from sncosmo.utils import integration_grid
from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
from sncosmo.models import Source

SALT2CL_B = 4302.57
SALT2CL_V = 5428.55


class SplineColorLaw:
    """
    Natural cubic spline color law.

    The coefficients are the values of the color law at the knots. Two
    further knots are fixed so that CL(B) = 0 and CL(V) = -1, as for the
    polynomial color law, and the color law is linearly extrapolated outside
    of the knots.

    Parameters
    ----------
    wave_range : tuple
        (min_wave, max_wave) in Angstroms, the first and last knots. Must
        contain the B and V wavelengths.
    n_coeffs : int
        Number of coefficients, at least 2. The knots start out uniform, with
        the two nearest to B and V moved onto those wavelengths.
    """

    def __init__(self, wave_range, n_coeffs):
        wave_lo, wave_hi = wave_range
        if n_coeffs < 2:
            raise ValueError(f"spline color law needs at least 2 coefficients, got {n_coeffs}")
        if not wave_lo < SALT2CL_B < SALT2CL_V < wave_hi:
            raise ValueError(f"spline color law range {wave_range} must contain the B and V wavelengths")

        knots = np.linspace(wave_lo, wave_hi, n_coeffs + 2)
        for anchor in (SALT2CL_B, SALT2CL_V):
            interior = np.arange(1, knots.size - 1)
            interior = interior[~np.isin(knots[interior], (SALT2CL_B, SALT2CL_V))]
            knots[interior[np.argmin(np.abs(knots[interior] - anchor))]] = anchor
        knots.sort()

        self.knots = knots
        self.spacing = np.diff(knots)
        free = ~np.isin(knots, (SALT2CL_B, SALT2CL_V))
        self.fixedvalues = np.where(knots == SALT2CL_V, -1., 0.)
        self.valuebasis = np.eye(knots.size)[:, free]

        # Second derivatives at the knots are linear in the knot values, and
        # vanish at the end knots for a natural spline
        h = self.spacing
        lhs, rhs = np.eye(knots.size), np.zeros((knots.size, knots.size))
        for i in range(1, knots.size - 1):
            lhs[i, i-1:i+2] = h[i-1] / 6, (h[i-1] + h[i]) / 3, h[i] / 6
            rhs[i, i-1:i+2] = 1 / h[i-1], -1 / h[i-1] - 1 / h[i], 1 / h[i]
        self.curvature = np.linalg.solve(lhs, rhs)

    def __call__(self, coeffs, wave, xp=np):
        """
        Evaluate the color law.

        Parameters
        ----------
        coeffs : array_like
            Color law values at the free knots.
        wave : array_like
            Wavelengths in Angstroms.
        xp : module, optional
            Array module to evaluate with, numpy or jax.numpy.

        Returns
        -------
        ndarray
            Color law values at wave.
        """
        wave = xp.atleast_1d(wave)
        knots, spacing = xp.asarray(self.knots), xp.asarray(self.spacing)
        values = self.fixedvalues + xp.dot(self.valuebasis, xp.asarray(coeffs))
        secondderivs = xp.dot(self.curvature, values)

        i = xp.clip(xp.searchsorted(knots, wave, side='right') - 1, 0, knots.size - 2)
        h = spacing[i]
        a = (knots[i+1] - wave) / h
        b = 1 - a
        interior = (a * values[i] + b * values[i+1]
                    + ((a**3 - a) * secondderivs[i]
                       + (b**3 - b) * secondderivs[i+1]) * h**2 / 6)

        slope_lo = ((values[1] - values[0]) / spacing[0]
                    - spacing[0] * secondderivs[1] / 6)
        slope_hi = ((values[-1] - values[-2]) / spacing[-1]
                    + spacing[-1] * secondderivs[-2] / 6)
        return xp.where(wave < knots[0], values[0] + slope_lo * (wave - knots[0]),
                        xp.where(wave > knots[-1], values[-1] + slope_hi * (wave - knots[-1]),
                                 interior))


def colorlawfromfile(name_or_obj):
    """
    Read a color law from a color correction file.

    Parameters
    ----------
    name_or_obj : str or fileobj
        Color correction file. Salt2ExtinctionLaw.version in the file names
        the form of the color law, 1 for the SALT2 polynomial and
        colorlaw_spline for SplineColorLaw.

    Returns
    -------
    callable
        Function of wavelength in Angstroms returning the color law.
    """
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
    version = '0'
    colorlaw_range = [3000., 7000.]
    for i in range(1+ncoeffs, len(words), 2):
        if words[i] == 'Salt2ExtinctionLaw.version':
            version = words[i+1]
        elif words[i] == 'Salt2ExtinctionLaw.min_lambda':
            colorlaw_range[0] = float(words[i+1])
        elif words[i] == 'Salt2ExtinctionLaw.max_lambda':
            colorlaw_range[1] = float(words[i+1])
        else:
            raise RuntimeError("Unexpected keyword: {}".format(words[i]))

    # Set extinction function to use.
    if version == '0':
        raise RuntimeError("Salt2ExtinctionLaw.version 0 not supported.")
    elif version == '1':
        return SALT2ColorLaw(colorlaw_range, colorlaw_coeffs)
    elif version == 'colorlaw_spline':
        return partial(SplineColorLaw(colorlaw_range, ncoeffs), np.array(colorlaw_coeffs))
    raise RuntimeError('unrecognized Salt2ExtinctionLaw.version: ' + version)


class SALT3Source(Source):
    """The SALT3 Type Ia supernova spectral timeseries model.

    The spectral flux density of this model is given by

    .. math::

       F(t, \\lambda) = x_0 (M_0(t, \\lambda) + x_1 M_1(t, \\lambda))
                       \\times 10^{-0.4 CL(\\lambda) c}

    where ``x0``, ``x1`` and ``c`` are the free parameters of the model,
    ``M_0``, ``M_1`` are the zeroth and first components of the model, and
    ``CL`` is the colorlaw, which gives the extinction in magnitudes for
    ``c=1``.

    Parameters
    ----------
    modeldir : str, optional
        Directory path containing model component files. Default is `None`,
        which means that no directory is prepended to filenames when
        determining their path.
    m0file, m1file, clfile : str or fileobj, optional
        Filenames of various model components. Defaults are:

        * m0file = 'salt3_template_0.dat' (2-d grid)
        * m1file = 'salt3_template_1.dat' (2-d grid)
        * clfile = 'salt3_color_correction.dat'

    errscalefile, lcrv00file, lcrv11file, lcrv01file, cdfile : str or fileobj
        (optional) Filenames of various model components for
        model covariance in synthetic photometry. See
        ``bandflux_rcov`` for details.  Defaults are:

        * errscalefile = 'salt3_lc_dispersion_scaling.dat' (2-d grid)
        * lcrv00file = 'salt3_lc_relative_variance_0.dat' (2-d grid)
        * lcrv11file = 'salt3_lc_relative_variance_1.dat' (2-d grid)
        * lcrv01file = 'salt3_lc_relative_covariance_01.dat' (2-d grid)
        * cdfile = 'salt3_color_dispersion.dat' (1-d grid)

    Notes
    -----
    The "2-d grid" files have the format ``<phase> <wavelength>
    <value>`` on each line.

    The phase and wavelength values of the various components don't
    necessarily need to match. (In the most recent salt2 model data,
    they do not all match.) The phase and wavelength values of the
    first model component (in ``m0file``) are taken as the "native"
    sampling of the model, even though these values might require
    interpolation of the other model components.

    """

    # These files are distributed with SALT3 model data but not currently
    # used:
    # v00file = 'salt3_spec_variance_0.dat'              : 2dgrid
    # v11file = 'salt3_spec_variance_1.dat'              : 2dgrid
    # v01file = 'salt3_spec_covariance_01.dat'           : 2dgrid

    _param_names = ['x0', 'x1', 'c']
    param_names_latex = ['x_0', 'x_1', 'c']
    _SCALE_FACTOR = 1e-12

    def __init__(self, modeldir=None,
                 m0file='salt3_template_0.dat',
                 m1file='salt3_template_1.dat',
                 clfile='salt3_color_correction.dat',
                 cdfile='salt3_color_dispersion.dat',
                 errscalefile='salt3_lc_dispersion_scaling.dat',
                 lcrv00file='salt3_lc_variance_0.dat',
                 lcrv11file='salt3_lc_variance_1.dat',
                 lcrv01file='salt3_lc_covariance_01.dat',
                 name=None, version=None):
        self.name = name
        self.version = version
        self._model = {}
        self._parameters = np.array([1., 0., 0.])

        names_or_objs = {'M0': m0file, 'M1': m1file,
                         'LCRV00': lcrv00file, 'LCRV11': lcrv11file,
                         'LCRV01': lcrv01file, 'errscale': errscalefile,
                         'cdfile': cdfile, 'clfile': clfile}

        # Make filenames into full paths.
        if modeldir is not None:
            for k in names_or_objs:
                v = names_or_objs[k]
                if (v is not None and isinstance(v, str)):
                    names_or_objs[k] = os.path.join(modeldir, v)

        # model components are interpolated to 2nd order
        for key in ['M0', 'M1']:
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
            values *= self._SCALE_FACTOR
            self._model[key] = BicubicInterpolator(phase, wave, values)

            # The "native" phases and wavelengths of the model are those
            # of the first model component.
            if key == 'M0':
                self._phase = phase
                self._wave = wave

        # model covariance is interpolated to 1st order
        for key in ['LCRV00', 'LCRV11', 'LCRV01', 'errscale']:
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
            self._model[key] = BicubicInterpolator(phase, wave, values)

        # Set the colorlaw based on the "color correction" file.
        self._set_colorlaw_from_file(names_or_objs['clfile'])

        # Set the color dispersion from "color_dispersion" file
        w, val = np.loadtxt(names_or_objs['cdfile'], unpack=True)
        self._colordisp = Spline1d(w, val,  k=1)  # linear interp.

    def _flux(self, phase, wave):
        m0 = self._model['M0'](phase, wave)
        m1 = self._model['M1'](phase, wave)
        return (self._parameters[0] * (m0 + self._parameters[1] * m1) *
                10. ** (-0.4 * self._colorlaw(wave) * self._parameters[2]))

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
        f0 = np.sum(m0 * tmp, axis=1) * dwave / HC_ERG_AA
        m1int = np.sum(m1 * tmp, axis=1) * dwave / HC_ERG_AA
        ftot = f0 + x1 * m1int

        # In the following, the "[:,0]" reduces from a 2-d array of shape
        # (nphase, 1) to a 1-d array.
        lcrv00 = self._model['LCRV00'](phase, band.wave_eff)[:, 0]
        lcrv11 = self._model['LCRV11'](phase, band.wave_eff)[:, 0]
        lcrv01 = self._model['LCRV01'](phase, band.wave_eff)[:, 0]
        scale = self._model['errscale'](phase, band.wave_eff)[:, 0]

        v = lcrv00 + 2.0 * x1 * lcrv01 + x1 * x1 * lcrv11

        # v is supposed to be variance but can go negative
        # due to interpolation.  Correct negative values to some small
        # number. (at present, use prescription of snfit : set
        # negatives to 0.0001)
        v[v < 0.0] = 0.0001

        # avoid warnings due to evaluating 0. / 0. in f0 / ftot
        with np.errstate(invalid='ignore'):
            #result = v * (f0 / ftot)**2 * scale**2
            result = v/(ftot/(trans*wave*dwave).sum())/HC_ERG_AA/1e12
            
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

        self._colorlaw = colorlawfromfile(name_or_obj)

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

