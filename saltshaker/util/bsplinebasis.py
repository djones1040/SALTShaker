"""
Vectorized B-spline basis matrices.

Each column reproduces ``scipy.interpolate.bisplev`` called with a one-hot
coefficient vector, so a single call replaces a loop over every coefficient.
"""

import numpy as np
from scipy.interpolate import BSpline


def bsplinebasis(x, knots, order, deriv=0):
    """
    Evaluate every 1D B-spline basis function (or a derivative) at ``x``.

    Points are clamped to ``[knots[order], knots[-order-1]]``, as ``bisplev`` does.

    Returns
    -------
    ndarray, shape (len(x), len(knots) - order - 1)
    """
    knots = np.asarray(knots, dtype=float)
    nbasis = knots.size - order - 1
    x = np.clip(np.atleast_1d(np.asarray(x, dtype=float)), knots[order], knots[nbasis])
    return BSpline(knots, np.eye(nbasis), order, extrapolate=True)(x, nu=deriv)


def tensorbsplinebasis(x, y, xknots, yknots, order, dx=0, dy=0):
    """
    Evaluate every 2D tensor-product basis function on the grid ``x`` by ``y``.

    Column ``i`` is ``bisplev(x, y, (xknots, yknots, np.arange(n) == i, order, order), dx, dy)``,
    i.e. coefficients are ordered with the ``x`` index slowest.

    Returns
    -------
    ndarray, shape (len(x), len(y), nxbasis * nybasis)
    """
    xbasis = bsplinebasis(x, xknots, order, dx)
    ybasis = bsplinebasis(y, yknots, order, dy)
    return np.einsum('ip,jq->ijpq', xbasis, ybasis).reshape(xbasis.shape[0], ybasis.shape[0], -1)
