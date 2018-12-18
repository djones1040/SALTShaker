#!/usr/bin/env python
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import least_squares

def bazin(time, A, B, t0, tfall, trise):
	X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
	return A * X + B

def estimate_tpk_bazin(time, flux, fluxerr, t0=None, max_nfev=10000000):

	fluxerr = fluxerr*10**(0.4*0.02)
	
	scaled_time = time - time.min()
	if not t0:
		t0 = scaled_time[flux.argmax()]
	else:
		t0 = t0 - time.min()
	guess = (0, 0, t0, 40, -5)

	iTime = (scaled_time > -10) & (scaled_time < 10)
	flux,fluxerr,scaled_time = flux[iTime],fluxerr[iTime],scaled_time[iTime]
	
	errfunc = lambda params: abs(flux - bazin(scaled_time, *params))/fluxerr

	result = least_squares(errfunc, guess, method='lm',max_nfev=max_nfev)

	return result.x[2]+time.min(),result.message
