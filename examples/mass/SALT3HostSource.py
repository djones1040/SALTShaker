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
from sncosmo.models import SALT2Source

__all__ = ['get_source', 'Source', 'TimeSeriesSource', 'StretchSource',
           'SUGARSource', 'SALT2Source', 'SALT3Source', 'MLCS2k2Source',
           'SNEMOSource', 'Model', 'PropagationEffect', 'CCM89Dust',
           'OD94Dust', 'F99Dust']

_SOURCES = Registry()


class SALT3HostSource(SALT2Source):
    """The SALT3 Type Ia supernova spectral timeseries model.
    Kenworthy et al., 2021, ApJ, submitted.  Model definitions
    are the same as SALT2 except for the errors, which are now
    given in flux space.  Unlike SALT2, no file is used for scaling
    the errors.

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

        * m0file = 'salt2_template_0.dat' (2-d grid)
        * m1file = 'salt2_template_1.dat' (2-d grid)
        * clfile = 'salt2_color_correction.dat'

    lcrv00file, lcrv11file, lcrv01file, cdfile : str or fileobj
        (optional) Filenames of various model components for
        model covariance in synthetic photometry. See
        ``bandflux_rcov`` for details.  Defaults are:

        * lcrv00file = 'salt2_lc_relative_variance_0.dat' (2-d grid)
        * lcrv11file = 'salt2_lc_relative_variance_1.dat' (2-d grid)
        * lcrv01file = 'salt2_lc_relative_covariance_01.dat' (2-d grid)
        * cdfile = 'salt2_color_dispersion.dat' (1-d grid)

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

    _param_names = ['x0', 'x1', 'xhost', 'c']
    param_names_latex = ['x_0', 'x_1', 'xhost', 'c']
    _SCALE_FACTOR = 1e-12

    def __init__(self, modeldir=None,
                 m0file='salt3_template_0.dat',
                 m1file='salt3_template_1.dat',
                 mhostfile='salt3_template_host.dat',
                 clfile='salt3_color_correction.dat',
                 cdfile='salt3_color_dispersion.dat',
                 lcrv00file='salt3_lc_variance_0.dat',
                 lcrv11file='salt3_lc_variance_1.dat',
                 lcrv01file='salt3_lc_covariance_01.dat',
                 name=None, version=None):

        self.name = name
        self.version = version
        self._model = {}
        self._parameters = np.array([1., 0., 0., 0.])

        names_or_objs = {'M0': m0file, 'M1': m1file, 'Mhost': mhostfile,
                         'LCRV00': lcrv00file, 'LCRV11': lcrv11file,
                         'LCRV01': lcrv01file,
                         'cdfile': cdfile, 'clfile': clfile}

        # Make filenames into full paths.
        if modeldir is not None:
            for k in names_or_objs:
                v = names_or_objs[k]
                if (v is not None and isinstance(v, str)):
                    names_or_objs[k] = os.path.join(modeldir, v)

        # model components are interpolated to 2nd order
        for key in ['M0', 'M1', 'Mhost']:
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
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

    def _flux(self, phase, wave):
        m0 = self._model['M0'](phase, wave)
        m1 = self._model['M1'](phase, wave)
        mhost = self._model['Mhost'](phase, wave)
        return (self._parameters[0] * (m0 + self._parameters[1] * m1 + self._parameters[2] * mhost) *
                10. ** (-0.4 * self._colorlaw(wave) * self._parameters[3]))
