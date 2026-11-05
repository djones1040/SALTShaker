import unittest
from types import SimpleNamespace

import numpy as np

from saltshaker.training.init_templates import getsplineknots


def mkdatadict():
    # dense coverage from 3500-6000 Angstroms, sparse out to 10000
    rng = np.random.default_rng(0)
    wave = np.concatenate([rng.uniform(3500, 6000, 20000), rng.uniform(2500, 10000, 2000)])
    spec = SimpleNamespace(wavelength=wave, flux=np.ones(wave.size), fluxerr=0.05 * np.ones(wave.size))
    return {"sn": SimpleNamespace(zHelio=0.0, specdata={0: spec})}


class test_knots(unittest.TestCase):
    knotargs = ([-20, 50], [2000, 11000], 3.0, 69.3, 3)

    def test_uniform(self):
        phase, wave = getsplineknots(*self.knotargs, "uniform")
        self.assertTrue(np.allclose(np.diff(phase), phase[1] - phase[0]))
        self.assertTrue(np.allclose(np.diff(wave), wave[1] - wave[0]))

    def test_snpca(self):
        phase, wave = getsplineknots(*self.knotargs, "snpca")
        self.assertEqual((phase.size, wave.size), (21, 107))

    def test_adaptive(self):
        phase, wave = getsplineknots(*self.knotargs, "adaptive", mkdatadict())
        uniformphase, uniformwave = getsplineknots(*self.knotargs, "uniform")
        self.assertTrue(np.array_equal(phase, uniformphase))
        self.assertTrue((wave[:4] == 2000).all() and (wave[-4:] == 11000).all())
        self.assertTrue((np.diff(wave[3:-3]) > 0).all())
        self.assertLess(wave.size, uniformwave.size)
        spacing = np.diff(wave[3:-3])
        center = (wave[3:-4] + wave[4:-3]) / 2
        self.assertLess(
            np.median(spacing[(center > 3500) & (center < 6000)]),
            np.median(spacing[center > 6000]),
        )

    def test_adaptive_spacing_reaches_minimum(self):
        wave = getsplineknots(*self.knotargs, "adaptive", mkdatadict())[1]
        spacing = np.diff(wave[3:-3]).min()
        self.assertGreater(spacing, 68.3)
        self.assertLess(spacing, 75)

    def test_adaptive_tuning(self):
        low = getsplineknots(*self.knotargs, "adaptive", mkdatadict(), 0.1)[1]
        high = getsplineknots(*self.knotargs, "adaptive", mkdatadict(), 100.0)[1]
        self.assertLess(np.std(np.diff(high[3:-3])), np.std(np.diff(low[3:-3])))

    def test_adaptive_without_spectra(self):
        with self.assertRaises(ValueError):
            getsplineknots(*self.knotargs, "adaptive")

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            getsplineknots(*self.knotargs, "linear")


if __name__ == "__main__":
    unittest.main()
