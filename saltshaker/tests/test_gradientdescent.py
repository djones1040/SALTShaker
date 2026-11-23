import unittest
from types import SimpleNamespace

import numpy as np
from jax import numpy as jnp

from saltshaker.training.optimizers.gradientdescent import rpropwithbacktracking


class shrinkinglinesearch(rpropwithbacktracking):
    """rprop whose line search always cuts the step tenfold, so its learning rates collapse."""

    def twowaybacktracking(self, *args, **kwargs):
        return 0.1

    def initializelearningrates(self, X):
        return np.full(X.size, 30.)

    def lossfunction(self, X, diff=None, **kwargs):
        loss = jnp.sum(X**2)
        return (loss, 2 * X) if diff == 'valueandgrad' else loss


def makeoptimizer(npar):
    optimizer = shrinkinglinesearch.__new__(shrinkinglinesearch)
    optimizer.saltobj = SimpleNamespace(ic=np.array([], dtype=int), icoordinates=np.array([], dtype=int))
    optimizer.ifixedparams = np.zeros(npar, dtype=bool)
    optimizer.Xbounds = np.tile(-np.inf, npar), np.tile(np.inf, npar)
    optimizer.minlearning, optimizer.maxlearning = 0, np.inf
    optimizer.etaminus, optimizer.etaplus = 0.5, 1.2
    optimizer.convergencetolerance = 1e-3
    optimizer.memorydebug = optimizer.write_large_outputs = False
    optimizer.functionevals = 0
    optimizer.losshistory, optimizer.Xhistory = [], []
    return optimizer


class test_rprop_convergence(unittest.TestCase):

    def test_collapsed_rates_not_converged(self):
        """A loss stalled by collapsed learning rates is not reported converged while reinitialized rates still reduce it."""
        npar = 3
        optimizer = makeoptimizer(npar)
        with self.assertLogs('saltshaker.training.optimizers.gradientdescent', 'INFO') as logs:
            X, loss, rates = optimizer.optimizeparams(
                np.ones(npar), np.ones(npar, dtype=bool), np.ones(npar), niter=2000)
        self.assertIn('Convergence achieved', logs.output[-1])
        self.assertLess(loss, 1e-2)


if __name__ == '__main__':
    unittest.main()
