#!/usr/bin/env python
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import least_squares

def bazin(time, A, B, t0, tfall, trise):
	X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
	return A * X + B

def estimate_tpk_bazin(time, flux, fluxerr):
	scaled_time = time - time.min()
	t0 = scaled_time[flux.argmax()]
	guess = (0, 0, t0, 40, -5)

	errfunc = lambda params: abs(flux - bazin(scaled_time, *params))/fluxerr

	result = least_squares(errfunc, guess, method='lm')

	return result.x[2]+time.min(),result.message
