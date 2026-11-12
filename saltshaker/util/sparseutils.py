"""
Host-side construction of JAX sparse arrays.
"""

import numpy as np
from jax.experimental import sparse


def bcoofromdense(mat):
    """
    Build the same array as ``jax.experimental.sparse.BCOO.fromdense(mat)`` in numpy.

    JAX compiles ``fromdense`` for every distinct shape, which dominates precomputation
    because each light curve and spectrum has its own shape.
    """
    mat = np.asarray(mat)
    nonzero = np.nonzero(mat)
    return bcoofromnonzeros(mat[nonzero], np.stack(nonzero, axis=-1), mat.shape)


def bcoofromnonzeros(values, indices, shape):
    """
    Build a BCOO array from its nonzero values and their indices, given in row-major order.

    The result matches ``bcoofromdense`` of the corresponding dense array.
    """
    return sparse.BCOO((np.asarray(values), np.asarray(indices, dtype=np.int32).reshape(-1, len(shape))),
                       shape=shape, indices_sorted=True, unique_indices=True)
