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


class risingloss(rpropwithbacktracking):
    """rprop whose rates collapse within an iteration of each reinitialization, on a loss that rises with each gradient evaluation."""
    gradientevals = 0

    def twowaybacktracking(self, *args, **kwargs):
        return 0.01

    def initializelearningrates(self, X):
        self.trialiterations.append(len(self.losshistory))
        return np.full(X.size, 300.)

    def lossfunction(self, X, diff=None, **kwargs):
        if diff == 'valueandgrad':
            self.gradientevals += 1
        loss = jnp.sum(X**2) + 3e-3 * self.gradientevals
        return (loss, 2 * X) if diff == 'valueandgrad' else loss


class stalledbeforetrial(rpropwithbacktracking):
    """rprop on a slowly rising loss whose line search fails in the 19 iterations before the trial at iteration 139."""
    gradientevals = 0

    def twowaybacktracking(self, *args, **kwargs):
        return 0.01 if 120 <= len(self.losshistory) < 139 else 1.

    def initializelearningrates(self, X):
        return np.ones(X.size)

    def lossfunction(self, X, diff=None, **kwargs):
        if diff == 'valueandgrad':
            self.gradientevals += 1
        loss = jnp.sum(jnp.abs(X)) + 1e-2 * self.gradientevals
        return (loss, jnp.sign(X)) if diff == 'valueandgrad' else loss


def makeoptimizer(cls, npar):
    optimizer = cls.__new__(cls)
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

    def optimize(self, optimizer, start):
        with self.assertLogs('saltshaker.training.optimizers.gradientdescent', 'INFO') as logs:
            X, loss, rates = optimizer.optimizeparams(
                start, np.ones(start.size, dtype=bool), np.ones(start.size), niter=2000)
        self.assertIn('Convergence achieved', logs.output[-1])
        return X, loss

    def test_collapsed_rates_not_converged(self):
        """A loss stalled by collapsed learning rates is not reported converged while reinitialized rates still reduce it."""
        X, loss = self.optimize(makeoptimizer(shrinkinglinesearch, 3), np.ones(3))
        self.assertLess(loss, 1e-2)

    def test_no_improvement_waits_for_trial_step(self):
        """A loss rising under collapsed rates is not reported converged on a trial iteration, before that trial's step is in the loss history."""
        optimizer = makeoptimizer(risingloss, 3)
        optimizer.trialiterations = []
        X, loss = self.optimize(optimizer, np.full(3, 0.7))
        self.assertNotIn(len(optimizer.losshistory) - 1, optimizer.trialiterations)
        np.testing.assert_array_less(X, 0.4)

    def test_no_improvement_waits_for_trial_progress(self):
        """A trial whose first step lowers the loss, but not below the start of the no-improvement window, gets further steps before the fit is reported converged."""
        optimizer = makeoptimizer(stalledbeforetrial, 3)
        optimizer.maxlearning = 0.05
        X, loss = self.optimize(optimizer, np.full(3, 10.))
        np.testing.assert_array_less(np.abs(X), 1)
        losses = optimizer.losshistory
        # Iteration 140 holds the trial's first step; its window starts at iteration 121
        self.assertLess(losses[140], losses[139])
        self.assertGreater(losses[140], losses[121])


if __name__ == '__main__':
    unittest.main()
