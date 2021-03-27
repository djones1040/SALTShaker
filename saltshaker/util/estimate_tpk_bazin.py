#!/usr/bin/env python
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import least_squares

import logging
log=logging.getLogger(__name__)

def bazin(time, A, B, t0, tfall, trise):
	X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
	return A * X + B

def estimate_tpk_bazin(time, flux, fluxerr, t0=None, max_nfev=10000000):

	fluxerr = fluxerr*10**(0.4*0.02)
	
	scaled_time = time - time.min()

	if t0:
		iTime = ((scaled_time > t0-time.min()-10) &
				 (scaled_time < t0-time.min()+20))
	else:
		iTime = (scaled_time > 0) & (scaled_time < 30)
	flux,fluxerr,scaled_time = flux[iTime],fluxerr[iTime],scaled_time[iTime]
	if len(scaled_time) < 3:
		return -99,'failed'
	
	if not t0:
		t0 = scaled_time[flux.argmax()]
		bounds = ((-np.inf,-np.inf,-np.inf,-np.inf,-np.inf),
				  (np.inf,np.inf,np.inf,np.inf,np.inf))
	else:
		t0 = t0 - time.min()
		bounds = ((-np.inf,-np.inf,np.min(scaled_time),-np.inf,-np.inf),
				  (np.inf,np.inf,np.max(scaled_time),np.inf,np.inf))
	guess = (flux.max(), 0, t0, 40, -5)
	
	errfunc = lambda params: abs(flux - bazin(scaled_time, *params))#/fluxerr
	result = least_squares(errfunc, guess,#method='trf',
						   max_nfev=max_nfev,bounds=bounds)
	scaled_time_bettersamp = np.arange(
		np.min(scaled_time),np.max(scaled_time),0.1)
	tpk = scaled_time_bettersamp[bazin(
		scaled_time_bettersamp, *result.x).argmax()]

	return tpk+time.min(),f"{'success' if status>0 else ''}:{result.message}"
