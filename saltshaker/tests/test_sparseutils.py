import unittest

import numpy as np
from jax.experimental import sparse

from saltshaker.util.sparseutils import bcoofromdense


class test_sparseutils(unittest.TestCase):

    def test_matches_jax_fromdense(self):
        """bcoofromdense returns the same data, indices, dtypes and flags as BCOO.fromdense."""
        rng = np.random.default_rng(0)
        dense3d = rng.normal(size=(4, 5, 6)) * (rng.uniform(size=(4, 5, 6)) < 0.3)
        for mat in [dense3d, dense3d[0, 0], np.zeros((3, 2))]:
            expected, result = sparse.BCOO.fromdense(mat), bcoofromdense(mat)
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.nse, expected.nse)
            self.assertEqual(result.data.dtype, expected.data.dtype)
            self.assertEqual(result.indices.dtype, expected.indices.dtype)
            self.assertEqual((result.indices_sorted, result.unique_indices),
                             (expected.indices_sorted, expected.unique_indices))
            np.testing.assert_array_equal(np.asarray(result.data), np.asarray(expected.data))
            np.testing.assert_array_equal(np.asarray(result.indices), np.asarray(expected.indices))


if __name__ == "__main__":
    unittest.main()
