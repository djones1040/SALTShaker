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
    indices = np.stack(nonzero, axis=-1).astype(np.int32)
    return sparse.BCOO((mat[nonzero], indices), shape=mat.shape,
                       indices_sorted=True, unique_indices=True)
