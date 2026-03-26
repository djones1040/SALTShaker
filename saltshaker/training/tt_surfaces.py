"""
Tensor Train (TT) surface parameterization for SALTShaker.

Provides an alternative to B-spline surfaces for representing M0, M1, and
optionally host-mass-dependent spectral components. When surface_type='tt' is
set in the config, these functions are used instead of bisplev.

The TT decomposition represents a 2D surface (phase x wavelength) as:
    M(p, λ) ≈ Σ_k U_k(p) * V_k(λ)     [rank-r SVD]

With a host-mass dimension, it becomes a 3D tensor:
    M(p, λ, M★) ≈ Σ_{jk} U_j(p) * W_{jk}(M★) * V_k(λ)

where U, W, V are the TT cores.

All functions are JAX-compatible for autodiff in the training loop.
"""

import numpy as np
from scipy.interpolate import bisplev, RectBivariateSpline

import jax
from jax import numpy as jnp


# ---------------------------------------------------------------------------
# Core TT evaluation (JAX)
# ---------------------------------------------------------------------------

def tt_eval_2d(core_phase, core_wave, phase_grid, wave_grid):
    """Evaluate a rank-r TT surface on a phase x wavelength grid.

    This is equivalent to evaluating U @ diag(sigma) @ V^T where
    core_phase = U * sqrt(sigma) and core_wave = V * sqrt(sigma),
    but we store them as TT cores for consistency with 3D.

    Parameters
    ----------
    core_phase : array (n_phase, r)
        Phase TT core. Each row gives the r-dimensional representation
        of that phase point.
    core_wave : array (r, n_wave)
        Wavelength TT core.
    phase_grid : array (n_p,)
        Phase points at which to evaluate (indices into core_phase).
    wave_grid : array (n_w,)
        Wavelength points at which to evaluate (indices into core_wave).

    Returns
    -------
    surface : array (n_p, n_w)
        Evaluated surface.
    """
    return core_phase @ core_wave


def tt_eval_3d(core_phase, core_mass, core_wave, mass_value,
               mass_knots, phase_grid=None, wave_grid=None):
    """Evaluate a rank-r TT surface with host-mass dependence.

    M(p, λ, M★) = core_phase @ core_mass(M★) @ core_wave

    where core_mass(M★) is obtained by linear interpolation of the
    mass core along the mass axis.

    Parameters
    ----------
    core_phase : array (n_phase, r1)
        Phase TT core.
    core_mass : array (r1, n_mass, r2)
        Mass TT core. Middle index is the mass grid.
    core_wave : array (r2, n_wave)
        Wavelength TT core.
    mass_value : float
        Host galaxy log stellar mass for this SN.
    mass_knots : array (n_mass,)
        Mass grid points (e.g., log M★ values).
    phase_grid : ignored (for API compatibility)
    wave_grid : ignored (for API compatibility)

    Returns
    -------
    surface : array (n_phase, n_wave)
        Evaluated surface at the given mass value.
    """
    # Linear interpolation along mass axis
    mass_matrix = _interp_mass_core(core_mass, mass_value, mass_knots)
    # Contract: (n_phase, r1) @ (r1, r2) @ (r2, n_wave)
    return core_phase @ mass_matrix @ core_wave


def _interp_mass_core(core_mass, mass_value, mass_knots):
    """Interpolate the mass core at a given mass value.

    Parameters
    ----------
    core_mass : array (r1, n_mass, r2)
        Mass TT core.
    mass_value : float
        Host galaxy log stellar mass.
    mass_knots : array (n_mass,)
        Mass grid points.

    Returns
    -------
    mass_matrix : array (r1, r2)
        Interpolated mass core matrix.
    """
    # Clip to grid range
    mass_clipped = jnp.clip(mass_value, mass_knots[0], mass_knots[-1])

    # Find interpolation index
    idx = jnp.searchsorted(mass_knots, mass_clipped) - 1
    idx = jnp.clip(idx, 0, len(mass_knots) - 2)

    # Linear interpolation weight
    frac = (mass_clipped - jnp.take(mass_knots, idx)) / (jnp.take(mass_knots, idx + 1) - jnp.take(mass_knots, idx))

    # Interpolate: (r1, r2)
    low = jax.lax.dynamic_slice_in_dim(core_mass, idx, 1, axis=1).squeeze(axis=1)
    high = jax.lax.dynamic_slice_in_dim(core_mass, idx + 1, 1, axis=1).squeeze(axis=1)
    mass_matrix = (1 - frac) * low + frac * high
    return mass_matrix


# ---------------------------------------------------------------------------
# Initialization: B-spline coefficients -> TT cores
# ---------------------------------------------------------------------------

def bspline_to_tt_cores(bspline_coeffs, n_phase_knots, n_wave_knots, rank):
    """Convert B-spline coefficient array to TT cores via truncated SVD.

    Parameters
    ----------
    bspline_coeffs : array (n_phase_knots * n_wave_knots,)
        Flattened B-spline coefficients (SALTShaker format).
    n_phase_knots : int
        Number of phase knot intervals.
    n_wave_knots : int
        Number of wavelength knot intervals.
    rank : int
        TT rank (number of components to keep).

    Returns
    -------
    core_phase : array (n_phase_knots, rank)
    core_wave : array (rank, n_wave_knots)
    """
    # Reshape to 2D matrix
    matrix = bspline_coeffs.reshape(n_phase_knots, n_wave_knots)

    # Truncated SVD
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Truncate to rank
    r = min(rank, len(s))
    core_phase = U[:, :r] * np.sqrt(s[:r])[np.newaxis, :]
    core_wave = np.sqrt(s[:r])[:, np.newaxis] * Vt[:r, :]

    return core_phase, core_wave


def init_mass_core(rank, n_mass_bins):
    """Initialize the mass TT core as identity (no mass dependence).

    Returns core_mass of shape (rank, n_mass_bins, rank) where each
    mass slice is the identity matrix. This means the model starts
    with no mass dependence and learns it during training.

    Parameters
    ----------
    rank : int
        TT rank.
    n_mass_bins : int
        Number of mass grid points.

    Returns
    -------
    core_mass : array (rank, n_mass_bins, rank)
    """
    core_mass = np.zeros((rank, n_mass_bins, rank))
    for i in range(n_mass_bins):
        core_mass[:, i, :] = np.eye(rank)
    return core_mass


def make_mass_knots(n_mass_bins, mass_range=(7.0, 12.0)):
    """Create uniformly spaced mass grid points.

    Parameters
    ----------
    n_mass_bins : int
        Number of grid points.
    mass_range : tuple
        (min, max) of log10(M★/M_sun).

    Returns
    -------
    mass_knots : array (n_mass_bins,)
    """
    return np.linspace(mass_range[0], mass_range[1], n_mass_bins)


# ---------------------------------------------------------------------------
# TT cores <-> flat parameter vector
# ---------------------------------------------------------------------------

def tt_cores_to_params(core_phase, core_wave, core_mass=None):
    """Flatten TT cores into a 1D parameter vector.

    Parameters
    ----------
    core_phase : array (n_phase, r) or (n_phase, r1)
    core_wave : array (r, n_wave) or (r2, n_wave)
    core_mass : array (r1, n_mass, r2) or None

    Returns
    -------
    params : array (n_total,)
        Concatenated flattened cores.
    sizes : dict
        Shapes needed to reconstruct cores.
    """
    parts = [core_phase.ravel(), core_wave.ravel()]
    sizes = {
        'phase_shape': core_phase.shape,
        'wave_shape': core_wave.shape,
    }
    if core_mass is not None:
        parts.append(core_mass.ravel())
        sizes['mass_shape'] = core_mass.shape

    return np.concatenate(parts), sizes


def params_to_tt_cores(params, sizes):
    """Reconstruct TT cores from a flat parameter vector.

    Parameters
    ----------
    params : array (n_total,)
        Flattened TT parameters.
    sizes : dict
        Shapes from tt_cores_to_params.

    Returns
    -------
    core_phase, core_wave, core_mass (or None)
    """
    phase_size = np.prod(sizes['phase_shape'])
    wave_size = np.prod(sizes['wave_shape'])

    core_phase = params[:phase_size].reshape(sizes['phase_shape'])
    core_wave = params[phase_size:phase_size + wave_size].reshape(sizes['wave_shape'])

    core_mass = None
    if 'mass_shape' in sizes:
        mass_start = phase_size + wave_size
        core_mass = params[mass_start:].reshape(sizes['mass_shape'])

    return core_phase, core_wave, core_mass


# ---------------------------------------------------------------------------
# JAX-compatible versions for use in training loop
# ---------------------------------------------------------------------------

def jax_tt_cores_to_params(core_phase, core_wave, core_mass=None):
    """JAX version of tt_cores_to_params."""
    parts = [core_phase.ravel(), core_wave.ravel()]
    if core_mass is not None:
        parts.append(core_mass.ravel())
    return jnp.concatenate(parts)


def jax_params_to_tt_cores(params, sizes):
    """JAX version of params_to_tt_cores."""
    phase_size = int(np.prod(sizes['phase_shape']))
    wave_size = int(np.prod(sizes['wave_shape']))

    core_phase = params[:phase_size].reshape(sizes['phase_shape'])
    core_wave = params[phase_size:phase_size + wave_size].reshape(sizes['wave_shape'])

    core_mass = None
    if 'mass_shape' in sizes:
        mass_start = phase_size + wave_size
        core_mass = params[mass_start:].reshape(sizes['mass_shape'])

    return core_phase, core_wave, core_mass


# ---------------------------------------------------------------------------
# Surface evaluation from flat parameter vector (main entry point)
# ---------------------------------------------------------------------------

def tt_surface_from_params(params, sizes, mass_value=None, mass_knots=None):
    """Evaluate a TT surface from its flat parameter representation.

    This is the main entry point used by SALTModel and modelflux.

    Parameters
    ----------
    params : array
        Flat TT parameter vector for one component (M0 or M1).
    sizes : dict
        Core shapes from tt_cores_to_params.
    mass_value : float or None
        Host galaxy log stellar mass. If None, uses 2D evaluation.
    mass_knots : array or None
        Mass grid points. Required if mass_value is not None.

    Returns
    -------
    surface : array (n_phase, n_wave)
    """
    core_phase, core_wave, core_mass = jax_params_to_tt_cores(params, sizes)

    if core_mass is not None and mass_value is not None:
        return tt_eval_3d(core_phase, core_mass, core_wave,
                          mass_value, mass_knots)
    else:
        return tt_eval_2d(core_phase, core_wave, None, None)


# ---------------------------------------------------------------------------
# Evaluation on interpolated grids (for SALTModel compatibility)
# ---------------------------------------------------------------------------

def tt_surface_on_grid(core_phase, core_wave, phase_knots, wave_knots,
                       eval_phase, eval_wave, bsorder=3,
                       core_mass=None, mass_value=None, mass_knots=None):
    """Evaluate TT surface on arbitrary phase/wave grids using interpolation.

    The TT cores are defined on B-spline knot locations. To evaluate on
    the fine interpolation grid, we:
    1. Reconstruct the surface on the knot grid from TT cores
    2. Use RectBivariateSpline to interpolate to the evaluation grid

    This preserves compatibility with SALTShaker's B-spline interpolation
    while using TT cores as the underlying parameterization.

    Parameters
    ----------
    core_phase : array (n_phase_knots, r)
    core_wave : array (r, n_wave_knots)
    phase_knots : array
        Phase knot locations (not the full B-spline knot vector).
    wave_knots : array
        Wavelength knot locations.
    eval_phase : array
        Phase grid to evaluate on.
    eval_wave : array
        Wavelength grid to evaluate on.
    bsorder : int
        B-spline order for interpolation.
    core_mass : array or None
    mass_value : float or None
    mass_knots : array or None

    Returns
    -------
    surface : array (len(eval_phase), len(eval_wave))
    """
    # Reconstruct on knot grid
    if core_mass is not None and mass_value is not None:
        mass_matrix = _interp_mass_core(core_mass, mass_value, mass_knots)
        knot_surface = core_phase @ mass_matrix @ core_wave
    else:
        knot_surface = core_phase @ core_wave

    # Interpolate to evaluation grid
    # phase_knots and wave_knots are the centers of the B-spline knot intervals
    interp = RectBivariateSpline(phase_knots, wave_knots, knot_surface,
                                  kx=min(bsorder, len(phase_knots) - 1),
                                  ky=min(bsorder, len(wave_knots) - 1))
    return interp(eval_phase, eval_wave)


# ---------------------------------------------------------------------------
# Pre-computation for photometric filter convolutions
# ---------------------------------------------------------------------------

def tt_precompute_filter_basis(core_phase, core_wave, phase_knots, wave_knots,
                               obs_phase, obs_wave_range, passband, bsorder=3):
    """Pre-compute the TT basis convolved with a photometric passband.

    Analogous to the B-spline basis convolution in datamodels.py, but
    for TT cores. Returns a matrix that, when multiplied by the
    TT flux coefficients, gives the synthetic photometry.

    This is computed once per light curve during data setup.

    Parameters
    ----------
    core_phase : array (n_phase_knots, r)
    core_wave : array (r, n_wave_knots)
    phase_knots : array
    wave_knots : array
    obs_phase : array (n_obs,)
        Observed phases for this light curve.
    obs_wave_range : array
        Observer-frame wavelength grid.
    passband : array
        Filter transmission * wavelength / (denom * HC_ERG_AA).
    bsorder : int

    Returns
    -------
    basis_convolution : array (n_obs, n_params)
        Matrix mapping TT parameters to synthetic photometry.
    """
    # For the TT parameterization, the flux coefficients are the
    # TT core parameters themselves. The basis convolution depends
    # on which core we're differentiating with respect to.
    #
    # This is handled differently from B-splines — rather than
    # pre-computing basis function convolutions, we evaluate the
    # full surface and convolve. The Jacobian is computed via JAX autodiff.
    #
    # For now, we use the interpolation approach: evaluate surface
    # on the fine grid, then convolve with passband.
    raise NotImplementedError(
        "TT filter basis pre-computation not yet implemented. "
        "Use preintegrate_photometric_passband=False with TT surfaces."
    )
