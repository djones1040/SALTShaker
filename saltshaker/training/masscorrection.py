"""
Host-galaxy-mass-dependent correction to the model components.

Each component's B-spline coefficient matrix C becomes C + U @ W(m) @ V for a supernova
whose host has log stellar mass m. U (phase) and V (wavelength) are bases shared by all
components, so linear transformations of the components carry over exactly to the
corrections, and W(m) interpolates r x r matrices defined at a set of host-mass knots.

A change of M0 or M1 along M0, M1 or the color law can be undone by changing x0, x1 or c,
so the correction at each knot is restricted to the subspace with zero B-band flux at peak
and no component along M1 or along the color law times M0. These conditions use fixed
reference surfaces and apply to every component, so linear transformations of the
components keep them.

The model is linear in the knot weights, so the component surfaces at the knots, written
to MASSCORRECTIONFILE with the training output, give the model at any host mass.
"""

import os
import re
import shutil
from collections import namedtuple

import numpy as np
from jax import numpy as jnp
from scipy.optimize import minimize
from scipy.stats import norm

from saltshaker.util.bsplinebasis import tensorbsplinebasis

MASSCORRECTIONFILE = "salt3_masscorrection.npz"

# Training output for the correction; population is the (mean, spread) of the sample's log
# masses, and surfaces has shape (n_knots, n_components, n_phase, n_wave)
masscorrectionresult = namedtuple(
    "masscorrectionresult", ["knots", "population", "phasebasis", "wavebasis", "subspace", "surfaces"])

# Files a SALT3 fitter reads from a model directory
_MODELFILE = re.compile(r"^(SALT3\.INFO|salt3_(template_\d+|lc_\w+|color_\w+)\.dat)$")


def massknotweights(logmass, logmasserr, knots):
    """
    Interpolation weights on the mass knots, averaged over a Gaussian in log mass.

    Interpolation is linear between knots and constant beyond the end knots.

    Returns
    -------
    ndarray, shape (len(knots),)
        Weights summing to one; NaN for a NaN mass. A NaN or zero uncertainty gives the
        weights at the mass itself.
    """
    knots = np.asarray(knots, dtype=float)
    if not np.isfinite(logmass):
        return np.full(knots.size, np.nan)
    if not logmasserr > 0:
        return np.array([np.interp(logmass, knots, row) for row in np.eye(knots.size)])

    lo, hi = (knots[:-1] - logmass) / logmasserr, (knots[1:] - logmass) / logmasserr
    # Probability and first moment of the mass within each interval between knots
    prob = norm.cdf(hi) - norm.cdf(lo)
    moment = logmass * prob + logmasserr * (norm.pdf(lo) - norm.pdf(hi))
    spacing = np.diff(knots)
    weights = np.zeros(knots.size)
    weights[1:] += (moment - knots[:-1] * prob) / spacing
    weights[:-1] += (knots[1:] * prob - moment) / spacing
    weights[0] += norm.cdf(lo[0])
    weights[-1] += norm.sf(hi[-1])
    return weights


def populationmassdistribution(logmasses, logmasserrs):
    """
    Mean and intrinsic spread of the sample's log masses.

    The distribution is a Gaussian fit to the measured masses, each broadened by its own
    uncertainty, so imprecise masses carry little weight. NaN uncertainties count as zero.
    """
    logmasses, logmasserrs = np.asarray(logmasses, dtype=float), np.asarray(logmasserrs, dtype=float)
    measured = np.isfinite(logmasses)
    if measured.sum() < 2:
        raise ValueError("the host-mass correction needs host masses for at least two supernovae")
    masses, errs = logmasses[measured], np.nan_to_num(logmasserrs[measured])

    def negloglike(pars):
        mean, logspread = pars
        return -norm.logpdf(masses, mean, np.sqrt(np.exp(2 * logspread) + errs**2)).sum()

    mean, logspread = minimize(negloglike, [masses.mean(), np.log(masses.std() + 1e-3)],
                               method="Nelder-Mead").x
    return mean, np.exp(logspread)


def posteriormass(logmass, logmasserr, popmean, popspread):
    """Mean and spread of a host's log mass given its measurement and the population distribution."""
    if not np.isfinite(logmass):
        return popmean, popspread
    if not logmasserr > 0:
        return logmass, 0.0
    postvar = 1 / (1 / logmasserr**2 + 1 / popspread**2)
    return postvar * (logmass / logmasserr**2 + popmean / popspread**2), np.sqrt(postvar)


def sampleknotweights(logmasses, logmasserrs, knots, population=None):
    """
    Mass-knot weights for each supernova, shape (n_sn, len(knots)).

    Each supernova's weights are averaged over its mass measurement combined with the
    population distribution, which leaves precise masses nearly unchanged and takes
    imprecise or missing masses toward the population. population is (mean, spread),
    fit to the sample when not given.
    """
    if population is None:
        population = populationmassdistribution(logmasses, logmasserrs)
    return np.array([massknotweights(*posteriormass(logmass, logmasserr, *population), knots)
                     for logmass, logmasserr in zip(logmasses, logmasserrs)])


def masscorrectionbases(coefficients, rank):
    """
    Phase and wavelength bases for the correction, shared by all components.

    The bases are the leading singular vectors of the components' coefficient matrices,
    each normalized and stacked along wavelength (phase basis) or phase (wavelength basis).
    The phase basis carries the norm of the first component, so a correction coefficient
    of one is the size of M0.

    Parameters
    ----------
    coefficients : list of ndarray, each shape (n_phase, n_wave)
    rank : int

    Returns
    -------
    phasebasis : ndarray, shape (n_phase, rank)
    wavebasis : ndarray, shape (rank, n_wave)
    """
    normalized = [c / np.linalg.norm(c) for c in coefficients]
    phasebasis = np.linalg.svd(np.hstack(normalized), full_matrices=False)[0][:, :rank]
    wavebasis = np.linalg.svd(np.vstack(normalized), full_matrices=False)[2][:rank]
    return phasebasis * np.linalg.norm(coefficients[0]), wavebasis


def bbandpeakfunctional(kcordict, phaseknotloc, waveknotloc, order, wave):
    """Weights of the B-band flux at peak on a coefficient matrix, up to normalization, shape (n_phase, n_wave)."""
    passband = wave * np.interp(wave, kcordict["default"]["Bwave"], kcordict["default"]["Btp"], left=0, right=0)
    peakbasis = tensorbsplinebasis(np.zeros(1), wave, phaseknotloc, waveknotloc, order)[0]
    return (passband @ peakbasis).reshape(len(phaseknotloc) - order - 1, len(waveknotloc) - order - 1)


def masscorrectionreferences(m0, m1, bbandpeak, colorlaw, waveknotloc, order):
    """
    Surfaces that the correction at each knot is orthogonal to, as coefficient matrices.

    The correction has zero B-band flux at peak, and no component along M1 or along the
    color law times M0, with the color law evaluated at each wavelength coefficient's knot
    average.
    """
    centers = np.array([waveknotloc[j + 1:j + order + 1].mean() for j in range(m0.shape[1])])
    return [bbandpeak, m1, colorlaw(centers)[np.newaxis, :] * m0]


def masscorrectionsubspace(phasebasis, wavebasis, references):
    """
    Orthonormal basis of the r x r matrices W for which U @ W @ V is orthogonal to every reference.

    Returns
    -------
    ndarray, shape (rank**2, rank**2 - len(references))
    """
    rank = phasebasis.shape[1]
    if rank**2 <= len(references):
        raise ValueError(f"masscorrection_rank squared must exceed the {len(references)} conditions on the correction")
    conditions = np.array([(phasebasis.T @ ref @ wavebasis.T).flatten() for ref in references])
    conditions /= np.linalg.norm(conditions, axis=1)[:, np.newaxis]
    singular, rows = np.linalg.svd(conditions)[1:]
    if singular[-1] < 1e-6 * singular[0]:
        raise ValueError("the host-mass correction's reference surfaces are degenerate in its bases")
    return rows[len(references):].T


def masscorrection(weights, masscorr, phasebasis, wavebasis, subspace):
    """
    Coefficient corrections for one supernova, shape (n_components, n_phase * n_wave).

    Parameters
    ----------
    weights : array, shape (n_knots,)
        The supernova's mass-knot weights.
    masscorr : array, shape (n_components, (n_knots - 1) * n_subspace)
        Coordinates in ``subspace`` of the correction matrices at each knot after the
        first, flattened in (knot, coordinate) order. The first knot's coordinates are zero:
        training takes weights relative to the sample average, so only differences between
        knots affect the model.
    phasebasis, wavebasis : arrays from ``masscorrectionbases``
    subspace : array from ``masscorrectionsubspace``
    """
    rank = phasebasis.shape[1]
    perknot = masscorr.reshape(masscorr.shape[0], weights.size - 1, subspace.shape[1])
    atmass = jnp.tensordot(weights[1:], perknot, axes=([0], [1])) @ subspace.T
    return jnp.einsum("pr,crs,sw->cpw", phasebasis, atmass.reshape(-1, rank, rank),
                      wavebasis).reshape(masscorr.shape[0], -1)


def writemasscorrection(outdir, result, phase, wave):
    """Save a masscorrectionresult and the output grid to MASSCORRECTIONFILE in outdir."""
    np.savez(os.path.join(outdir, MASSCORRECTIONFILE), phase=phase, wave=wave,
             **{field: np.asarray(value) for field, value in result._asdict().items()})


def readmasscorrection(modeldir):
    """The masscorrectionresult saved in modeldir, with the output grid as (result, phase, wave)."""
    with np.load(os.path.join(modeldir, MASSCORRECTIONFILE)) as saved:
        result = masscorrectionresult(**{field: saved[field] for field in masscorrectionresult._fields})
        return result, saved["phase"], saved["wave"]


def surfacesatmass(result, logmass, logmasserr=0.0):
    """
    Component surfaces for a host of the given log mass, shape (n_components, n_phase, n_wave).

    The knot weights combine the mass measurement with the training sample's mass
    distribution, as in training; a NaN mass gives the population model.
    """
    weights = massknotweights(*posteriormass(logmass, logmasserr, *result.population), result.knots)
    return np.tensordot(weights, result.surfaces, axes=1)


def writemodelatmass(modeldir, outdir, logmass, logmasserr=0.0):
    """
    Write a standard SALT3 model directory for a host of the given log mass.

    The templates are the trained model at that mass (see surfacesatmass); the other
    model files are copied from modeldir unchanged.
    """
    result, phase, wave = readmasscorrection(modeldir)
    os.makedirs(outdir, exist_ok=True)
    for name in os.listdir(modeldir):
        if _MODELFILE.match(name):
            shutil.copy(os.path.join(modeldir, name), outdir)
    grid = np.reshape(np.meshgrid(phase, wave, indexing="ij"), (2, -1)).T
    for i, surface in enumerate(surfacesatmass(result, logmass, logmasserr)):
        np.savetxt(os.path.join(outdir, f"salt3_template_{i}.dat"),
                   np.column_stack([grid, surface.ravel()]), fmt=["%.1f", "%.2f", "%8.15e"])
