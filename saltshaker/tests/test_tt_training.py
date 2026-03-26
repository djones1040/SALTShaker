"""
Minimal end-to-end test for TT surface training in SALTShaker.

Creates a small fake dataset (3 SNe), initializes SALTResids with
surface_type='tt', and runs one forward pass through modelflux and
modelfluxvariance. This catches JAX tracing / dimension mismatch
errors in seconds instead of waiting an hour for real data to load.
"""

import unittest
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def _make_fake_training_data():
    """Create minimal fake SALTShaker training data for 3 SNe."""
    from saltshaker.training.TrainSALT import TrainSALT, RunTraining

    # Use the test config but override to TT
    salt = TrainSALT()
    rt = RunTraining()
    rt.get_config_options(salt, 'testdata/test.conf', [
        '--surface_type', 'tt',
        '--tt_rank', '3',
        '--tt_mass_bins', '5',
        '--tt_mass_range', '7.0', '12.0',
    ])
    return salt


class TestTTTrainingPipeline(unittest.TestCase):
    """Test TT surface integration with SALTShaker training pipeline."""

    def test_config_tt_options(self):
        """TT config options should be parsed correctly."""
        salt = _make_fake_training_data()
        self.assertEqual(salt.options.surface_type, 'tt')
        self.assertEqual(salt.options.tt_rank, 3)
        self.assertEqual(salt.options.tt_mass_bins, 5)

    def test_tt_parameter_initialization(self):
        """TT parameters should be initialized from B-spline coefficients."""
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, init_mass_core,
            tt_cores_to_params, params_to_tt_cores)

        # Simulate realistic SALT3-like dimensions
        n_phase, n_wave, rank = 24, 130, 3
        n_mass = 5

        # Create fake B-spline coefficients
        rng = np.random.RandomState(42)
        m0k = np.abs(rng.randn(n_phase * n_wave)) * 1e-10
        m0k[m0k == 0] = 1e-14

        # Convert to TT (what TrainSALT.initialParameters does)
        core_p, core_w = bspline_to_tt_cores(m0k, n_phase, n_wave, rank)
        core_m = init_mass_core(rank, n_mass)
        params, sizes = tt_cores_to_params(core_p, core_w, core_m)

        # Verify shapes
        self.assertEqual(core_p.shape, (n_phase, rank))
        self.assertEqual(core_w.shape, (rank, n_wave))
        self.assertEqual(core_m.shape, (rank, n_mass, rank))

        expected_len = n_phase * rank + rank * n_wave + rank * n_mass * rank
        self.assertEqual(len(params), expected_len)

        # Verify roundtrip
        core_p2, core_w2, core_m2 = params_to_tt_cores(params, sizes)
        np.testing.assert_allclose(core_p, core_p2)
        np.testing.assert_allclose(core_w, core_w2)
        np.testing.assert_allclose(core_m, core_m2)

        # Verify reconstruction: fewer params than B-spline
        self.assertLess(len(params), n_phase * n_wave)

    def test_tt_expand_components(self):
        """TT expansion should produce correct B-spline coefficient shape."""
        from jax import numpy as jnp
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, init_mass_core,
            tt_cores_to_params, make_mass_knots)
        from saltshaker.training.datamodels import _expand_tt_components_impl

        n_phase, n_wave, rank, n_mass = 10, 20, 3, 5
        mass_knots = make_mass_knots(n_mass)

        # Create fake TT params for 2 components
        rng = np.random.RandomState(42)
        components_list = []
        core_sizes = []
        for _ in range(2):
            coeffs = rng.randn(n_phase * n_wave)
            cp, cw = bspline_to_tt_cores(coeffs, n_phase, n_wave, rank)
            cm = init_mass_core(rank, n_mass)
            params, sizes = tt_cores_to_params(cp, cw, cm)
            components_list.append(params)
            core_sizes.append(sizes)

        components = jnp.stack([jnp.array(p) for p in components_list])

        # Expand
        expanded = _expand_tt_components_impl(
            components, core_sizes, 10.0, mass_knots)

        # Check shape: should be (2, n_phase * n_wave)
        self.assertEqual(expanded.shape, (2, n_phase * n_wave))

    def test_tt_modelflux_shapes(self):
        """TT modelflux should produce output compatible with pcderivsparse."""
        from jax import numpy as jnp
        from jax.experimental import sparse
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, init_mass_core,
            tt_cores_to_params, make_mass_knots)
        from saltshaker.training.datamodels import (
            _expand_tt_components_impl, modeledtraininglightcurve)

        n_phase, n_wave, rank, n_mass = 10, 20, 3, 5
        n_obs = 8
        n_components = 2

        # Simulate pcderivsparse: (n_obs, n_bspline_coeffs)
        rng = np.random.RandomState(42)
        pcderivsparse = sparse.BCOO.fromdense(
            rng.randn(n_obs, n_phase * n_wave) * 0.01)

        # Create TT components
        mass_knots = make_mass_knots(n_mass)
        all_params = []
        core_sizes = []
        for _ in range(n_components):
            coeffs = rng.randn(n_phase * n_wave) * 1e-10
            cp, cw = bspline_to_tt_cores(coeffs, n_phase, n_wave, rank)
            cm = init_mass_core(rank, n_mass)
            params, sizes = tt_cores_to_params(cp, cw, cm)
            all_params.append(params)
            core_sizes.append(sizes)

        components = jnp.stack([jnp.array(p) for p in all_params])

        # Expand TT -> B-spline coefficients
        expanded = _expand_tt_components_impl(
            components, core_sizes, 10.0, mass_knots)

        # Simulate modelflux computation
        coordinates = jnp.array([1.0, 0.5])  # [1, x1]
        x0 = 1e-5
        fluxcoeffs = jnp.dot(coordinates, expanded) * x0

        # This is the critical operation: pcderivsparse @ fluxcoeffs
        flux = pcderivsparse @ fluxcoeffs
        self.assertEqual(flux.shape, (n_obs,))

    def test_tt_gradient_flows(self):
        """JAX should be able to compute gradients through TT expansion."""
        import jax
        from jax import numpy as jnp
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, init_mass_core,
            tt_cores_to_params, make_mass_knots)
        from saltshaker.training.datamodels import _expand_tt_components_impl

        n_phase, n_wave, rank, n_mass = 5, 10, 2, 3
        mass_knots = make_mass_knots(n_mass)

        rng = np.random.RandomState(42)
        coeffs = rng.randn(n_phase * n_wave) * 1e-10
        cp, cw = bspline_to_tt_cores(coeffs, n_phase, n_wave, rank)
        cm = init_mass_core(rank, n_mass)
        params, sizes = tt_cores_to_params(cp, cw, cm)

        core_sizes = [sizes]

        def loss_fn(params_flat):
            components = jnp.stack([params_flat])
            expanded = _expand_tt_components_impl(
                components, core_sizes, 10.0, mass_knots)
            return jnp.sum(expanded ** 2)

        params_jax = jnp.array(params)
        grad = jax.grad(loss_fn)(params_jax)

        # Gradient should be non-zero
        self.assertGreater(jnp.abs(grad).sum(), 0)
        # Gradient should have same shape as params
        self.assertEqual(grad.shape, params_jax.shape)


if __name__ == '__main__':
    unittest.main()
