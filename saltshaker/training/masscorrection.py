"""
Host-galaxy-mass-dependent correction to the model components.

Each component's B-spline coefficient matrix C becomes C + U @ W(m) @ V for a supernova
whose host has log stellar mass m. U (phase) and V (wavelength) are bases shared by all
components, so linear transformations of the components carry over exactly to the
corrections, and W(m) interpolates r x r matrices defined at a set of host-mass knots.
"""

import numpy as np
from jax import numpy as jnp
from scipy.optimize import minimize
from scipy.stats import norm


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


def sampleknotweights(logmasses, logmasserrs, knots):
    """
    Mass-knot weights for each supernova, shape (n_sn, len(knots)).

    Each supernova's weights are averaged over its mass measurement combined with the
    population distribution, which leaves precise masses nearly unchanged and takes
    imprecise or missing masses toward the population.
    """
    popmean, popspread = populationmassdistribution(logmasses, logmasserrs)
    weights = []
    for logmass, logmasserr in zip(logmasses, logmasserrs):
        if not np.isfinite(logmass):
            postmean, postspread = popmean, popspread
        elif not logmasserr > 0:
            postmean, postspread = logmass, 0.0
        else:
            postvar = 1 / (1 / logmasserr**2 + 1 / popspread**2)
            postmean = postvar * (logmass / logmasserr**2 + popmean / popspread**2)
            postspread = np.sqrt(postvar)
        weights.append(massknotweights(postmean, postspread, knots))
    return np.array(weights)


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


def masscorrection(weights, masscorr, phasebasis, wavebasis):
    """
    Coefficient corrections for one supernova, shape (n_components, n_phase * n_wave).

    Parameters
    ----------
    weights : array, shape (n_knots,)
        The supernova's mass-knot weights.
    masscorr : array, shape (n_components, n_knots * rank * rank)
        Correction matrices at each knot, flattened in (knot, row, column) order.
    phasebasis, wavebasis : arrays from ``masscorrectionbases``
    """
    rank = phasebasis.shape[1]
    perknot = masscorr.reshape(masscorr.shape[0], weights.size, rank, rank)
    atmass = jnp.tensordot(weights, perknot, axes=([0], [1]))
    return jnp.einsum("pr,crs,sw->cpw", phasebasis, atmass, wavebasis).reshape(masscorr.shape[0], -1)
