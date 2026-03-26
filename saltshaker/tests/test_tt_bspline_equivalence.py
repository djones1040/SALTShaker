"""Test that TT and B-spline parameterizations give equivalent model fluxes.

At full rank (rank = min(n_phase, n_wave)), the TT decomposition is lossless,
so the expanded TT coefficients should exactly reproduce the original B-spline
coefficients. This test verifies equivalence at multiple levels:

1. Coefficient-level: TT cores expanded back to B-spline coefficients match
2. Surface-level: bisplev evaluated on both gives the same surface
3. modelflux-level: the full photometric model pipeline gives the same flux
"""

import unittest
import numpy as np
from scipy.interpolate import bisplev


class TestTTBsplineCoeffEquivalence(unittest.TestCase):
    """Test that TT expansion recovers exact B-spline coefficients at full rank."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.n_phase = 12
        self.n_wave = 60
        self.n_coeffs = self.n_phase * self.n_wave
        self.coeffs = self.rng.randn(self.n_coeffs)

    def test_full_rank_roundtrip(self):
        """At full rank, TT cores expanded back should equal original coefficients."""
        from saltshaker.training.tt_surfaces import bspline_to_tt_cores
        rank = min(self.n_phase, self.n_wave)
        core_p, core_w = bspline_to_tt_cores(
            self.coeffs, self.n_phase, self.n_wave, rank)
        reconstructed = (core_p @ core_w).ravel()
        np.testing.assert_allclose(reconstructed, self.coeffs, atol=1e-10)

    def test_full_rank_roundtrip_via_params(self):
        """Full roundtrip through flatten/unflatten/expand should be exact."""
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, tt_cores_to_params,
            params_to_tt_cores)
        rank = min(self.n_phase, self.n_wave)
        core_p, core_w = bspline_to_tt_cores(
            self.coeffs, self.n_phase, self.n_wave, rank)
        params, sizes = tt_cores_to_params(core_p, core_w)
        core_p2, core_w2, _ = params_to_tt_cores(params, sizes)
        reconstructed = (core_p2 @ core_w2).ravel()
        np.testing.assert_allclose(reconstructed, self.coeffs, atol=1e-10)

    def test_full_rank_with_identity_mass_core(self):
        """Identity mass core should not change the expanded coefficients."""
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, init_mass_core, make_mass_knots,
            _interp_mass_core)
        rank = min(self.n_phase, self.n_wave)
        core_p, core_w = bspline_to_tt_cores(
            self.coeffs, self.n_phase, self.n_wave, rank)
        core_m = init_mass_core(rank, 10)
        mass_knots = make_mass_knots(10)
        # At any mass, identity mass core should give same result
        for mass_val in [7.5, 9.0, 10.5, 11.5]:
            mm = _interp_mass_core(core_m, mass_val, mass_knots)
            reconstructed = (core_p @ mm @ core_w).ravel()
            np.testing.assert_allclose(
                reconstructed, self.coeffs, atol=1e-5,
                err_msg=f"Failed at mass={mass_val}")


class TestTTBsplineSurfaceEquivalence(unittest.TestCase):
    """Test that bisplev gives the same surface from B-spline and TT coefficients."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.bsorder = 3
        # Create a realistic knot vector
        self.n_phase_knots = 12
        self.n_wave_knots = 60
        self.n_coeffs = self.n_phase_knots * self.n_wave_knots
        # Knot vectors (with bsorder+1 repeated endpoints)
        phase_interior = np.linspace(-20, 50, self.n_phase_knots - self.bsorder + 1)
        self.phaseknotloc = np.concatenate([
            np.full(self.bsorder, phase_interior[0]),
            phase_interior,
            np.full(self.bsorder, phase_interior[-1])
        ])
        wave_interior = np.linspace(2000, 9200, self.n_wave_knots - self.bsorder + 1)
        self.waveknotloc = np.concatenate([
            np.full(self.bsorder, wave_interior[0]),
            wave_interior,
            np.full(self.bsorder, wave_interior[-1])
        ])
        self.coeffs = self.rng.randn(self.n_coeffs) * 0.01
        # Evaluation grids
        self.eval_phase = np.linspace(-15, 45, 50)
        self.eval_wave = np.linspace(2500, 9000, 100)

    def test_bisplev_equivalence_full_rank(self):
        """bisplev with original and TT-reconstructed coefficients should match."""
        from saltshaker.training.tt_surfaces import bspline_to_tt_cores
        rank = min(self.n_phase_knots, self.n_wave_knots)
        core_p, core_w = bspline_to_tt_cores(
            self.coeffs, self.n_phase_knots, self.n_wave_knots, rank)
        tt_coeffs = (core_p @ core_w).ravel()

        surface_bspline = bisplev(
            self.eval_phase, self.eval_wave,
            (self.phaseknotloc, self.waveknotloc,
             self.coeffs, self.bsorder, self.bsorder))
        surface_tt = bisplev(
            self.eval_phase, self.eval_wave,
            (self.phaseknotloc, self.waveknotloc,
             tt_coeffs, self.bsorder, self.bsorder))

        np.testing.assert_allclose(surface_bspline, surface_tt, atol=1e-8,
                                   err_msg="B-spline and TT surfaces differ")

    def test_bisplev_equivalence_low_rank(self):
        """At low rank, TT surface should be a reasonable approximation."""
        from saltshaker.training.tt_surfaces import bspline_to_tt_cores
        rank = 5
        core_p, core_w = bspline_to_tt_cores(
            self.coeffs, self.n_phase_knots, self.n_wave_knots, rank)
        tt_coeffs = (core_p @ core_w).ravel()

        surface_bspline = bisplev(
            self.eval_phase, self.eval_wave,
            (self.phaseknotloc, self.waveknotloc,
             self.coeffs, self.bsorder, self.bsorder))
        surface_tt = bisplev(
            self.eval_phase, self.eval_wave,
            (self.phaseknotloc, self.waveknotloc,
             tt_coeffs, self.bsorder, self.bsorder))

        # Not exact, but relative error should be bounded
        rel_err = np.linalg.norm(surface_bspline - surface_tt) / np.linalg.norm(surface_bspline)
        self.assertLess(rel_err, 1.0,
                        f"Rank-{rank} TT surface too far from B-spline (rel_err={rel_err:.4f})")


class TestTTBsplineModelFluxEquivalence(unittest.TestCase):
    """Test that the full modelflux pipeline gives the same result for B-spline and TT.

    This is the critical end-to-end test: given the same underlying surface,
    the photometric model flux (pcderivsparse @ coefficients) should be
    identical whether the coefficients come directly from B-splines or from
    TT cores expanded to B-spline coefficients.
    """

    def test_expand_tt_matches_bspline_coeffs(self):
        """_expand_tt_components_impl should return the original B-spline coefficients."""
        import jax.numpy as jnp
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, tt_cores_to_params)
        from saltshaker.training.datamodels import _expand_tt_components_impl

        rng = np.random.RandomState(42)
        n_phase, n_wave = 12, 60
        n_components = 2  # M0, M1
        rank = min(n_phase, n_wave)  # full rank for exact reconstruction

        # Create B-spline coefficients for each component
        bspline_coeffs = []
        tt_params_list = []
        tt_sizes_list = []
        for i in range(n_components):
            coeffs = rng.randn(n_phase * n_wave) * 0.01
            bspline_coeffs.append(coeffs)
            core_p, core_w = bspline_to_tt_cores(coeffs, n_phase, n_wave, rank)
            params, sizes = tt_cores_to_params(core_p, core_w)
            tt_params_list.append(params)
            tt_sizes_list.append(sizes)

        # Stack TT params as components array (like icomponents indexing)
        components = jnp.array(np.stack(tt_params_list))

        # Expand TT back to B-spline coefficients
        expanded = np.array(_expand_tt_components_impl(
            components, tt_sizes_list, 10.0, None))

        # Compare
        for i in range(n_components):
            np.testing.assert_allclose(
                expanded[i], bspline_coeffs[i], atol=1e-8,
                err_msg=f"Component {i}: expanded TT != original B-spline coeffs")

    def test_expand_tt_with_mass_core_identity(self):
        """With identity mass core, expanded TT should match B-spline at any mass."""
        import jax.numpy as jnp
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, init_mass_core, make_mass_knots,
            tt_cores_to_params)
        from saltshaker.training.datamodels import _expand_tt_components_impl

        rng = np.random.RandomState(42)
        n_phase, n_wave = 12, 60
        n_mass = 10
        rank = min(n_phase, n_wave)

        coeffs = rng.randn(n_phase * n_wave) * 0.01
        core_p, core_w = bspline_to_tt_cores(coeffs, n_phase, n_wave, rank)
        core_m = init_mass_core(rank, n_mass)
        mass_knots = make_mass_knots(n_mass)

        params, sizes = tt_cores_to_params(core_p, core_w, core_m)
        components = jnp.array(params[np.newaxis, :])  # 1 component

        for mass_val in [7.0, 9.5, 12.0]:
            expanded = np.array(_expand_tt_components_impl(
                components, [sizes], mass_val, mass_knots))
            np.testing.assert_allclose(
                expanded[0], coeffs, atol=1e-4,
                err_msg=f"Identity mass core changed coefficients at mass={mass_val}")

    def test_pcderivsparse_flux_equivalence(self):
        """Model flux through pcderivsparse should be identical for B-spline and TT.

        Constructs a minimal pcderivsparse (B-spline basis convolved with a
        top-hat filter) and verifies that:
            pcderivsparse @ bspline_coeffs == pcderivsparse @ tt_expanded_coeffs
        """
        import jax.numpy as jnp
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, tt_cores_to_params)
        from saltshaker.training.datamodels import _expand_tt_components_impl

        rng = np.random.RandomState(42)
        bsorder = 3
        n_phase_knots = 12
        n_wave_knots = 60
        n_coeffs = n_phase_knots * n_wave_knots
        rank = min(n_phase_knots, n_wave_knots)

        # Realistic knot vectors
        phase_interior = np.linspace(-20, 50, n_phase_knots - bsorder + 1)
        phaseknotloc = np.concatenate([
            np.full(bsorder, phase_interior[0]),
            phase_interior,
            np.full(bsorder, phase_interior[-1])
        ])
        wave_interior = np.linspace(2000, 9200, n_wave_knots - bsorder + 1)
        waveknotloc = np.concatenate([
            np.full(bsorder, wave_interior[0]),
            wave_interior,
            np.full(bsorder, wave_interior[-1])
        ])

        # B-spline coefficients for M0
        bspline_coeffs = rng.randn(n_coeffs) * 0.01

        # Build a simple pcderivsparse: evaluate B-spline basis at a few
        # phase/wave points (mimics filter convolution)
        eval_phases = np.array([-5.0, 0.0, 5.0, 10.0])
        eval_wave = np.linspace(4000, 7000, 50)
        dwave = np.mean(np.diff(eval_wave))

        # Build basis matrix: (n_obs, n_coeffs)
        n_obs = len(eval_phases)
        pcderivsparse = np.zeros((n_obs, n_coeffs))
        for i in range(n_coeffs):
            basis_i = bisplev(
                eval_phases, eval_wave,
                (phaseknotloc, waveknotloc,
                 np.arange(n_coeffs) == i, bsorder, bsorder))
            # Integrate over wavelength (like filter convolution with top-hat)
            pcderivsparse[:, i] = basis_i.sum(axis=1) * dwave

        # B-spline flux
        flux_bspline = pcderivsparse @ bspline_coeffs

        # TT flux: convert to TT cores, expand back, multiply by same pcderivsparse
        core_p, core_w = bspline_to_tt_cores(
            bspline_coeffs, n_phase_knots, n_wave_knots, rank)
        params, sizes = tt_cores_to_params(core_p, core_w)
        components = jnp.array(params[np.newaxis, :])
        expanded = np.array(_expand_tt_components_impl(
            components, [sizes], 10.0, None))
        flux_tt = pcderivsparse @ expanded[0]

        # Tolerance is 1e-5 because _expand_tt_components_impl uses JAX
        # float32 arithmetic, limiting precision to ~1e-6 relative error
        np.testing.assert_allclose(
            flux_tt, flux_bspline, atol=1e-5, rtol=1e-5,
            err_msg="Model flux differs between B-spline and TT paths")

    def test_pcderivsparse_flux_low_rank_bounded_error(self):
        """At low rank, model flux error should be bounded by coefficient error."""
        import jax.numpy as jnp
        from saltshaker.training.tt_surfaces import (
            bspline_to_tt_cores, tt_cores_to_params)
        from saltshaker.training.datamodels import _expand_tt_components_impl

        rng = np.random.RandomState(42)
        bsorder = 3
        n_phase_knots = 12
        n_wave_knots = 60
        n_coeffs = n_phase_knots * n_wave_knots
        rank = 5

        phase_interior = np.linspace(-20, 50, n_phase_knots - bsorder + 1)
        phaseknotloc = np.concatenate([
            np.full(bsorder, phase_interior[0]),
            phase_interior,
            np.full(bsorder, phase_interior[-1])
        ])
        wave_interior = np.linspace(2000, 9200, n_wave_knots - bsorder + 1)
        waveknotloc = np.concatenate([
            np.full(bsorder, wave_interior[0]),
            wave_interior,
            np.full(bsorder, wave_interior[-1])
        ])

        bspline_coeffs = rng.randn(n_coeffs) * 0.01

        # Build pcderivsparse
        eval_phases = np.array([-5.0, 0.0, 5.0, 10.0])
        eval_wave = np.linspace(4000, 7000, 50)
        dwave = np.mean(np.diff(eval_wave))
        n_obs = len(eval_phases)
        pcderivsparse = np.zeros((n_obs, n_coeffs))
        for i in range(n_coeffs):
            basis_i = bisplev(
                eval_phases, eval_wave,
                (phaseknotloc, waveknotloc,
                 np.arange(n_coeffs) == i, bsorder, bsorder))
            pcderivsparse[:, i] = basis_i.sum(axis=1) * dwave

        flux_bspline = pcderivsparse @ bspline_coeffs

        core_p, core_w = bspline_to_tt_cores(
            bspline_coeffs, n_phase_knots, n_wave_knots, rank)
        params, sizes = tt_cores_to_params(core_p, core_w)
        components = jnp.array(params[np.newaxis, :])
        expanded = np.array(_expand_tt_components_impl(
            components, [sizes], 10.0, None))
        flux_tt = pcderivsparse @ expanded[0]

        # Coefficient-level relative error
        coeff_rel_err = (np.linalg.norm(expanded[0] - bspline_coeffs)
                         / np.linalg.norm(bspline_coeffs))
        # Flux-level relative error should be similar or smaller
        # (pcderivsparse is a smoothing operator)
        flux_rel_err = (np.linalg.norm(flux_tt - flux_bspline)
                        / np.linalg.norm(flux_bspline))

        self.assertLess(flux_rel_err, coeff_rel_err + 0.01,
                        "Flux error exceeds coefficient error by too much")
        # Both should be reasonable
        self.assertLess(flux_rel_err, 1.0,
                        f"Rank-{rank} flux error too large: {flux_rel_err:.4f}")


if __name__ == '__main__':
    unittest.main()
