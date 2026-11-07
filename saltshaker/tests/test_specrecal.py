import unittest

import numpy as np
from scipy.special import factorial

from saltshaker.util.specrecal import specrecalbasis


class test_specrecal(unittest.TestCase):
    wave = np.sort(np.random.default_rng(0).uniform(3500, 9000, 800))

    def test_polynomial(self):
        coord = (self.wave - self.wave.mean()) / 2500
        expected = np.stack([coord**3 / factorial(3), coord**2 / 2, coord], axis=1)
        self.assertTrue(np.allclose(specrecalbasis(self.wave, 3), expected))

    def test_spline_vanishes_at_mean(self):
        # the mean is inserted as a sample so that the basis is evaluated there
        wave = np.sort(np.r_[self.wave, self.wave.mean() - 1e-9, self.wave.mean() + 1e-9])
        for n in (2, 3, 4, 9, 15):
            basis = specrecalbasis(wave, n, "quadraticspline")
            self.assertEqual(basis.shape, (wave.size, n))
            i = np.searchsorted(wave, wave.mean())
            self.assertLess(np.abs(basis[i]).max(), 1e-6)

    def test_spline_is_identifiable(self):
        # full rank, and with no constant for specx0 to be degenerate with
        for n in (2, 3, 4, 9, 15):
            basis = specrecalbasis(self.wave, n, "quadraticspline")
            self.assertEqual(np.linalg.matrix_rank(basis), n)
            full = np.column_stack([basis, np.ones(self.wave.size)])
            self.assertEqual(np.linalg.matrix_rank(full), n + 1)

    def test_spline_is_smooth(self):
        wave = np.linspace(3500, 9000, 5001)
        basis = specrecalbasis(wave, 6, "quadraticspline")
        slope = np.diff(basis, axis=0) / np.diff(wave)[:, np.newaxis]
        self.assertLess(np.abs(np.diff(slope, axis=0)).max(), 1e-5)

    def test_spline_no_parameters(self):
        self.assertEqual(specrecalbasis(self.wave, 0, "quadraticspline").shape, (self.wave.size, 0))

    def test_spline_too_few_parameters(self):
        with self.assertRaises(ValueError):
            specrecalbasis(self.wave, 1, "quadraticspline")

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            specrecalbasis(self.wave, 3, "cubicspline")


if __name__ == "__main__":
    unittest.main()
