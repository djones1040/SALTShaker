import io
import pickle
import unittest

import jax
import numpy as np
from jax import numpy as jnp
from scipy.interpolate import CubicSpline
from sncosmo.salt2utils import SALT2ColorLaw

from saltshaker.training import colorlaw
from saltshaker.util.salt3_sncosmo import (SALT2CL_B, SALT2CL_V, SplineColorLaw,
                                           colorlawfromfile)

jax.config.update("jax_enable_x64", True)


class test_colorlaw(unittest.TestCase):
    waverange = (2800., 8000.)
    wave = np.linspace(2000, 11000, 901)
    coeffs = np.array([3.1, 1.2, -1.8, -2.4, -3.3])

    def expected(self, spline, coeffs):
        values = np.insert(coeffs.astype(float), np.searchsorted(spline.knots, SALT2CL_B), 0.)
        values = np.insert(values, np.searchsorted(spline.knots, SALT2CL_V), -1.)
        return CubicSpline(spline.knots, values, bc_type='natural')

    def test_knots(self):
        for n in (2, 3, 5, 9, 20):
            spline = SplineColorLaw(self.waverange, n)
            self.assertEqual(spline.knots.size, n + 2)
            self.assertTrue(np.all(np.diff(spline.knots) > 0))
            self.assertEqual(tuple(spline.knots[[0, -1]]), self.waverange)
            self.assertTrue(np.isin([SALT2CL_B, SALT2CL_V], spline.knots).all())

    def test_invalid(self):
        with self.assertRaises(ValueError):
            SplineColorLaw(self.waverange, 1)
        with self.assertRaises(ValueError):
            SplineColorLaw((5000., 8000.), 5)

    def test_normalization(self):
        spline = SplineColorLaw(self.waverange, 5)
        values = spline(self.coeffs, np.array([SALT2CL_B, SALT2CL_V]))
        self.assertTrue(np.allclose(values, [0, -1], atol=1e-12))

    def test_matches_scipy_natural_spline(self):
        spline = SplineColorLaw(self.waverange, 5)
        inside = (self.wave >= self.waverange[0]) & (self.wave <= self.waverange[1])
        reference = self.expected(spline, self.coeffs)
        self.assertTrue(np.allclose(spline(self.coeffs, self.wave[inside]),
                                    reference(self.wave[inside])))

    def test_linear_extrapolation(self):
        spline = SplineColorLaw(self.waverange, 5)
        reference = self.expected(spline, self.coeffs)
        for edge, offset in ((self.waverange[0], -500.), (self.waverange[1], 1500.)):
            expected = reference(edge) + reference(edge, 1) * offset
            self.assertAlmostEqual(spline(self.coeffs, edge + offset)[0], float(expected))

    def test_numpy_jax_agree(self):
        law = colorlaw.getcolorlaw('colorlaw_spline')(5, self.waverange)
        self.assertTrue(np.allclose(law(1, jnp.array(self.coeffs), self.wave),
                                    law.spline(self.coeffs, self.wave), atol=1e-10))

    def test_jit_and_gradient(self):
        law = colorlaw.getcolorlaw('colorlaw_spline')(5, self.waverange)
        function = jax.jit(lambda c, coeffs, wave: law(c, coeffs, wave).sum())
        args = 0.1, jnp.array(self.coeffs), jnp.array(self.wave)
        self.assertAlmostEqual(float(function(*args)), float(law(*args).sum()), places=8)
        gradient = jax.grad(function, argnums=1)(*args)
        self.assertTrue(np.isfinite(gradient).all() and np.any(gradient != 0))

    def test_pickle(self):
        law = colorlaw.getcolorlaw('colorlaw_spline')(5, self.waverange)
        restored = pickle.loads(pickle.dumps(law))
        self.assertTrue(np.array_equal(restored(1, self.coeffs, self.wave),
                                       law(1, self.coeffs, self.wave)))

    def colorfile(self, version, coeffs):
        lines = [str(len(coeffs))] + [f'{c:.10e}' for c in coeffs] + [
            f'Salt2ExtinctionLaw.version {version}',
            f'Salt2ExtinctionLaw.min_lambda {self.waverange[0]:.0f}',
            f'Salt2ExtinctionLaw.max_lambda {self.waverange[1]:.0f}']
        return io.StringIO('\n'.join(lines))

    def test_file_polynomial(self):
        coeffs = [-0.5, 0.7, -0.4, 0.1, -0.01]
        self.assertTrue(np.array_equal(colorlawfromfile(self.colorfile(1, coeffs))(self.wave),
                                       SALT2ColorLaw(self.waverange, coeffs)(self.wave)))

    def test_file_spline(self):
        fromfile = colorlawfromfile(self.colorfile('colorlaw_spline', self.coeffs))
        spline = SplineColorLaw(self.waverange, 5)
        self.assertTrue(np.allclose(fromfile(self.wave), spline(self.coeffs, self.wave)))

    def test_file_unknown_version(self):
        for version in (0, 3, 'colorlaw_unknown'):
            with self.assertRaises(RuntimeError):
                colorlawfromfile(self.colorfile(version, self.coeffs))

    def test_file_version_names(self):
        self.assertEqual(colorlaw.fileversion('colorlaw_default'), '1')
        self.assertEqual(colorlaw.fileversion('colorlaw_spline'), 'colorlaw_spline')


if __name__ == "__main__":
    unittest.main()
