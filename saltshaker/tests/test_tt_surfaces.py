"""Tests for Tensor Train surface parameterization."""

import unittest
import numpy as np


class TestTTCoreOperations(unittest.TestCase):
    """Test TT core construction, serialization, and evaluation."""

    def setUp(self):
        self.n_phase = 24
        self.n_wave = 130
        self.rank = 5
        self.rng = np.random.RandomState(42)

    def test_exact_reconstruction_low_rank(self):
        """A rank-r matrix should be exactly reconstructed at rank r."""
        from saltshaker.training.tt_surfaces import bspline_to_tt_cores
        U = self.rng.randn(self.n_phase, self.rank)
        V = self.rng.randn(self.rank, self.n_wave)
        matrix = U @ V
        core_p, core_w = bspline_to_tt_cores(
            matrix.ravel(), self.n_phase, self.n_wave, self.rank)
        recon = core_p @ core_w
        rel_err = np.linalg.norm(matrix - recon) / np.linalg.norm(matrix)
        self.assertLess(rel_err, 1e-12)

    def test_lossy_compression(self):
        """A full-rank matrix compressed to rank 3 should have finite error."""
        from saltshaker.training.tt_surfaces import bspline_to_tt_cores
        matrix = self.rng.randn(self.n_phase, self.n_wave)
        core_p, core_w = bspline_to_tt_cores(
            matrix.ravel(), self.n_phase, self.n_wave, 3)
        recon = core_p @ core_w
        rel_err = np.linalg.norm(matrix - recon) / np.linalg.norm(matrix)
        self.assertGreater(rel_err, 0.01)
        self.assertLess(rel_err, 1.0)

    def test_core_shapes(self):
        """TT cores should have correct shapes."""
        from saltshaker.training.tt_surfaces import bspline_to_tt_cores
        coeffs = self.rng.randn(self.n_phase * self.n_wave)
        core_p, core_w = bspline_to_tt_cores(
            coeffs, self.n_phase, self.n_wave, self.rank)
        self.assertEqual(core_p.shape, (self.n_phase, self.rank))
        self.assertEqual(core_w.shape, (self.rank, self.n_wave))

    def test_flatten_unflatten_2d(self):
        """Flatten and unflatten should be inverse operations (2D)."""
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, tt_cores_to_params, params_to_tt_cores)
        coeffs = self.rng.randn(self.n_phase * self.n_wave)
        core_p, core_w = bspline_to_tt_cores(
            coeffs, self.n_phase, self.n_wave, self.rank)
        params, sizes = tt_cores_to_params(core_p, core_w)
        core_p2, core_w2, core_m2 = params_to_tt_cores(params, sizes)
        np.testing.assert_allclose(core_p, core_p2)
        np.testing.assert_allclose(core_w, core_w2)
        self.assertIsNone(core_m2)

    def test_flatten_unflatten_3d(self):
        """Flatten and unflatten should be inverse operations (3D with mass)."""
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, init_mass_core,
            tt_cores_to_params, params_to_tt_cores)
        coeffs = self.rng.randn(self.n_phase * self.n_wave)
        core_p, core_w = bspline_to_tt_cores(
            coeffs, self.n_phase, self.n_wave, self.rank)
        core_m = init_mass_core(self.rank, 10)
        params, sizes = tt_cores_to_params(core_p, core_w, core_m)
        core_p2, core_w2, core_m2 = params_to_tt_cores(params, sizes)
        np.testing.assert_allclose(core_p, core_p2)
        np.testing.assert_allclose(core_w, core_w2)
        np.testing.assert_allclose(core_m, core_m2)

    def test_parameter_count_reduction(self):
        """TT should use fewer parameters than B-spline for rank < sqrt(min(n))."""
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, tt_cores_to_params)
        coeffs = self.rng.randn(self.n_phase * self.n_wave)
        core_p, core_w = bspline_to_tt_cores(
            coeffs, self.n_phase, self.n_wave, self.rank)
        params, _ = tt_cores_to_params(core_p, core_w)
        n_bspline = self.n_phase * self.n_wave
        n_tt = len(params)
        self.assertLess(n_tt, n_bspline)


class TestMassCore(unittest.TestCase):
    """Test host-mass TT dimension."""

    def setUp(self):
        self.rank = 5
        self.n_mass = 10

    def test_identity_mass_core(self):
        """Identity mass core should have each slice equal to eye(rank)."""
        from saltshaker.training.tt_surfaces import init_mass_core
        core_m = init_mass_core(self.rank, self.n_mass)
        self.assertEqual(core_m.shape, (self.rank, self.n_mass, self.rank))
        for i in range(self.n_mass):
            np.testing.assert_allclose(core_m[:, i, :], np.eye(self.rank))

    def test_identity_preserves_surface(self):
        """With identity mass core, surface should be the same at all masses."""
        from saltshaker.training.tt_surfaces import (
            init_mass_core, make_mass_knots, _interp_mass_core)
        core_m = init_mass_core(self.rank, self.n_mass)
        mass_knots = make_mass_knots(self.n_mass)
        for mass in [7.5, 9.0, 10.0, 11.0, 11.9]:
            mm = _interp_mass_core(core_m, mass, mass_knots)
            np.testing.assert_allclose(mm, np.eye(self.rank), atol=1e-10)

    def test_mass_interpolation_boundary(self):
        """Mass values at grid boundaries should use exact grid values."""
        from saltshaker.training.tt_surfaces import (
            init_mass_core, make_mass_knots, _interp_mass_core)
        core_m = np.random.RandomState(42).randn(self.rank, self.n_mass, self.rank)
        mass_knots = make_mass_knots(self.n_mass)
        # At exact knot location, should get exact slice
        # (JAX float32 arithmetic limits precision to ~1e-5)
        mm = _interp_mass_core(core_m, mass_knots[3], mass_knots)
        np.testing.assert_allclose(mm, core_m[:, 3, :], atol=1e-5)

    def test_mass_interpolation_midpoint(self):
        """Mass at midpoint between two knots should average them."""
        from saltshaker.training.tt_surfaces import (
            make_mass_knots, _interp_mass_core)
        core_m = np.random.RandomState(42).randn(self.rank, self.n_mass, self.rank)
        mass_knots = make_mass_knots(self.n_mass)
        mid = 0.5 * (mass_knots[3] + mass_knots[4])
        mm = _interp_mass_core(core_m, mid, mass_knots)
        expected = 0.5 * (core_m[:, 3, :] + core_m[:, 4, :])
        np.testing.assert_allclose(mm, expected, atol=1e-5)

    def test_perturbed_mass_core_creates_difference(self):
        """A perturbed mass core should produce different surfaces at different masses."""
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, init_mass_core, make_mass_knots,
            tt_cores_to_params, tt_surface_from_params)
        rng = np.random.RandomState(42)
        n_p, n_w, rank = 24, 100, 5
        coeffs = rng.randn(n_p * n_w)
        core_p, core_w = bspline_to_tt_cores(coeffs, n_p, n_w, rank)
        core_m = init_mass_core(rank, 10)
        mass_knots = make_mass_knots(10)
        # Perturb: make low-mass end brighter
        core_m[:, 0, :] *= 1.1
        core_m[:, -1, :] *= 0.9
        params, sizes = tt_cores_to_params(core_p, core_w, core_m)
        surf_lo = tt_surface_from_params(params, sizes, 7.0, mass_knots)
        surf_hi = tt_surface_from_params(params, sizes, 12.0, mass_knots)
        diff = np.linalg.norm(surf_lo - surf_hi) / np.linalg.norm(surf_lo)
        self.assertGreater(diff, 0.01)


class TestTTSurfaceEvaluation(unittest.TestCase):
    """Test end-to-end surface evaluation."""

    def test_tt_surface_from_params_2d(self):
        """tt_surface_from_params should reconstruct the surface."""
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, tt_cores_to_params, tt_surface_from_params)
        rng = np.random.RandomState(42)
        n_p, n_w, rank = 20, 80, 5
        U = rng.randn(n_p, rank)
        V = rng.randn(rank, n_w)
        matrix = U @ V
        core_p, core_w = bspline_to_tt_cores(matrix.ravel(), n_p, n_w, rank)
        params, sizes = tt_cores_to_params(core_p, core_w)
        surf = tt_surface_from_params(params, sizes)
        self.assertEqual(surf.shape, (n_p, n_w))
        np.testing.assert_allclose(surf, matrix, atol=1e-10)

    def test_tt_surface_from_params_3d_identity(self):
        """With identity mass core, 3D evaluation should match 2D."""
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, init_mass_core, make_mass_knots,
            tt_cores_to_params, tt_surface_from_params)
        rng = np.random.RandomState(42)
        n_p, n_w, rank = 20, 80, 5
        coeffs = rng.randn(n_p * n_w)
        core_p, core_w = bspline_to_tt_cores(coeffs, n_p, n_w, rank)
        # 2D surface
        params_2d, sizes_2d = tt_cores_to_params(core_p, core_w)
        surf_2d = tt_surface_from_params(params_2d, sizes_2d)
        # 3D with identity mass core
        core_m = init_mass_core(rank, 10)
        mass_knots = make_mass_knots(10)
        params_3d, sizes_3d = tt_cores_to_params(core_p, core_w, core_m)
        surf_3d = tt_surface_from_params(params_3d, sizes_3d, 10.0, mass_knots)
        np.testing.assert_allclose(surf_2d, surf_3d, atol=1e-5)

    def test_make_mass_knots(self):
        """Mass knots should be uniformly spaced in the given range."""
        from saltshaker.training.tt_surfaces import make_mass_knots
        knots = make_mass_knots(10, (7.0, 12.0))
        self.assertEqual(len(knots), 10)
        self.assertAlmostEqual(knots[0], 7.0)
        self.assertAlmostEqual(knots[-1], 12.0)
        diffs = np.diff(knots)
        np.testing.assert_allclose(diffs, diffs[0], atol=1e-10)


class TestSALTShakerIntegration(unittest.TestCase):
    """Test that TT integrates with SALTShaker config and classes."""

    def test_config_parsing(self):
        """TT config options should parse correctly."""
        from saltshaker.training.TrainSALT import RunTraining, TrainSALT
        salt = TrainSALT()
        rt = RunTraining()
        rt.get_config_options(salt, 'testdata/test.conf', [])
        self.assertEqual(salt.options.surface_type, 'bspline')
        self.assertEqual(salt.options.tt_rank, 5)
        self.assertEqual(salt.options.tt_mass_bins, 0)

    def test_host_logmass_in_slots(self):
        """host_logmass should be in all relevant class slots."""
        from saltshaker.util.readutils import SALTtrainingSN
        from saltshaker.training.datamodels import (
            SALTfitcacheSN, modeledtraininglightcurve)
        self.assertIn('host_logmass', SALTtrainingSN.__slots__)
        self.assertIn('host_logmass', SALTfitcacheSN.__slots__)
        self.assertIn('host_logmass', modeledtraininglightcurve.__slots__)

    def test_tt_attributes_in_lightcurve_slots(self):
        """TT-specific attributes should be available on modeledtraininglightcurve."""
        from saltshaker.training.datamodels import modeledtraininglightcurve
        self.assertIn('surface_type', modeledtraininglightcurve.__slots__)
        self.assertIn('host_logmass', modeledtraininglightcurve.__slots__)
        # tt_core_sizes and tt_mass_knots are class-level (not in __slots__)
        self.assertTrue(hasattr(modeledtraininglightcurve, '_tt_core_sizes'))
        self.assertTrue(hasattr(modeledtraininglightcurve, '_tt_mass_knots'))


if __name__ == '__main__':
    unittest.main()
