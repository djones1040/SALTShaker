import unittest

import numpy as np
from scipy.interpolate import bisplev

from saltshaker.training.init_templates import getsplineknots
from saltshaker.util.bsplinebasis import bsplinebasis, tensorbsplinebasis


def bisplevbasis(x, y, xknots, yknots, order, dx=0, dy=0):
    n = (xknots.size - order - 1) * (yknots.size - order - 1)
    return np.stack([
        np.reshape(bisplev(x, y, (xknots, yknots, np.arange(n) == i, order, order), dx, dy),
                   (len(x), len(y)))
        for i in range(n)], axis=-1)


class test_bsplinebasis(unittest.TestCase):
    knotargs = ([-20, 50], [2000, 11000], 3.0, 69.3, 3)

    def check_matches_bisplev(self, method):
        phaseknots, waveknots = getsplineknots(*self.knotargs, method)
        # Includes both ends of the knot range, an interior knot, and points outside the range
        phase = np.array([-25, -20, -19.5, phaseknots[5], 0, 49.9, 50, 60])
        wave = np.array([1500, 2000, waveknots[7], 5000.3, 10999, 11000, 12000])
        np.testing.assert_allclose(
            tensorbsplinebasis(phase, wave, phaseknots, waveknots, 3),
            bisplevbasis(phase, wave, phaseknots, waveknots, 3), rtol=0, atol=1e-14)
        for dx, dy in [(1, 0), (0, 1), (1, 1)]:
            inphase, inwave = phase[1:-1], wave[1:-1]
            np.testing.assert_allclose(
                tensorbsplinebasis(inphase, inwave, phaseknots, waveknots, 3, dx, dy),
                bisplevbasis(inphase, inwave, phaseknots, waveknots, 3, dx, dy), rtol=0, atol=1e-14)

    def test_matches_bisplev_uniform(self):
        """Every column equals bisplev with a one-hot coefficient vector, on uniform knots."""
        self.check_matches_bisplev("uniform")

    def test_matches_bisplev_snpca(self):
        """Every column equals bisplev with a one-hot coefficient vector, on non-uniform knots."""
        self.check_matches_bisplev("snpca")

    def test_partition_of_unity_at_endpoints(self):
        """The basis sums to one at both ends of the knot range, including the right end."""
        phaseknots, _ = getsplineknots(*self.knotargs, "uniform")
        basis = bsplinebasis([-20, 50], phaseknots, 3)
        np.testing.assert_allclose(basis.sum(axis=1), 1, rtol=0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
