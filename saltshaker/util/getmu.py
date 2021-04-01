#!/usr/bin/env python
"""very basic utilities for dealing w/ fitres objects"""
import numpy as np
from saltshaker.util import cosmo

def getmu(inp,salt2alpha=0.147,salt2beta=3.13,sigint=0.1,deltam=0,peczerr=0.00083):

	inp.mu,inp.muerr = salt2mu(x1=inp.x1,x1err=inp.x1ERR,c=inp.c,cerr=inp.cERR,mb=inp.mB,mberr=inp.mBERR,
							   cov_x1_c=inp.COV_x1_c,cov_x1_x0=inp.COV_x1_x0,cov_c_x0=inp.COV_c_x0,
							   hostmass=inp.HOST_LOGMASS,
							   alpha=salt2alpha,
							   beta=salt2beta,
							   x0=inp.x0,z=inp.zHD,sigint=sigint,deltam=deltam,peczerr=peczerr)
	inp.mures = inp.mu - cosmo.mu(inp.zHD)

	return(inp)

def getmu_h0(inp,salt2alpha=0.147,salt2beta=3.13,sigint=0.1,deltam=0):

	inp.mu,inp.muerr = salt2mu_h0(x1=inp.x1,x1err=inp.x1ERR,c=inp.c,cerr=inp.cERR,mb=inp.mB,mberr=inp.mBERR,
								  cov_x1_c=inp.COV_x1_c,cov_x1_x0=inp.COV_x1_x0,cov_c_x0=inp.COV_c_x0,
								  alpha=salt2alpha,
								  beta=salt2beta,
								  hostmass=11,
								  x0=inp.x0,z=inp.zHD,sigint=sigint,deltam=deltam)	 
	inp.mures = inp.mu - cosmo.mu(inp.zHD)

	return(inp)


def mkcuts(fr,salt2alpha=0.147,salt2beta=3.13,zmin=None,zmax=None,fitprobmin=0.001,trestmax=5):

	if not zmin: zmin = np.min(fr.zHD)
	if not zmax: zmax = np.max(fr.zHD)

	sf = -2.5/(fr.x0*np.log(10.0))
	invvars = 1./(fr.mBERR**2.+ salt2alpha**2. * fr.x1ERR**2. + \
					  salt2beta**2. * fr.cERR**2. +	 2.0 * salt2alpha * (fr.COV_x1_x0*sf) - \
					  2.0 * salt2beta * (fr.COV_c_x0*sf) - \
					  2.0 * salt2alpha*salt2beta * (fr.COV_x1_c) )

	try:
		cols = np.where((fr.x1 > -3.0) & (fr.x1 < 3.0) &
						(fr.c > -0.3) & (fr.c < 0.3) &
						(fr.x1ERR < 1) & (fr.PKMJDERR < 2*(1+fr.zHD)) &
						(fr.FITPROB >= fitprobmin) &
						(invvars > 0) & (fr.zHD >= zmin) & 
						(fr.zHD <= zmax) & (fr.TrestMAX > trestmax))
	except:
		print('Warning : Keyword TrestMAX not found!!!')
		cols = np.where((fr.x1 > -3.0) & (fr.x1 < 3.0) &
						(fr.c > -0.3) & (fr.c < 0.3) &
						(fr.x1ERR < 1) & (fr.PKMJDERR < 2*(1+fr.zHD)) &
						(fr.FITPROB >= fitprobmin) &
						(invvars > 0) & (fr.zHD >= zmin) & 
						(fr.zHD <= zmax))

	for k in fr.__dict__.keys():
		fr.__dict__[k] = fr.__dict__[k][cols]

	return(fr)

def mkfoundcuts(fr,salt2alpha=0.147,salt2beta=3.13,zmin=None,zmax=None,fitprobmin=0.001):

	if not zmin: zmin = np.min(fr.zHD)
	if not zmax: zmax = np.max(fr.zHD)

	sf = -2.5/(fr.x0*np.log(10.0))
	invvars = 1./(fr.mBERR**2.+ salt2alpha**2. * fr.x1ERR**2. + \
					  salt2beta**2. * fr.cERR**2. +	 2.0 * salt2alpha * (fr.COV_x1_x0*sf) - \
					  2.0 * salt2beta * (fr.COV_c_x0*sf) - \
					  2.0 * salt2alpha*salt2beta * (fr.COV_x1_c) )

	try:
		cols = np.where((fr.x1 > -3.0) & (fr.x1 < 3.0) &
						(fr.c > -0.3) & (fr.c < 0.3) &
						(fr.x1ERR < 1) & (fr.PKMJDERR < 1*(1+fr.zHD)) &
						(fr.FITPROB >= fitprobmin) &
						(invvars > 0) & (fr.zHD >= zmin) & 
						(fr.zHD <= zmax) & (fr.TrestMAX > 5))
	except:
		print('Warning : Keyword TrestMAX not found!!!')
		cols = np.where((fr.x1 > -3.0) & (fr.x1 < 3.0) &
						(fr.c > -0.3) & (fr.c < 0.3) &
						(fr.x1ERR < 1) & (fr.PKMJDERR < 1*(1+fr.zHD)) &
						(fr.FITPROB >= fitprobmin) &
						(invvars > 0) & (fr.zHD >= zmin) & 
						(fr.zHD <= zmax))

	for k in fr.__dict__.keys():
		fr.__dict__[k] = fr.__dict__[k][cols]

	return(fr)


def salt2mu(x1=None,x1err=None,
			c=None,cerr=None,
			mb=None,mberr=None,
			cov_x1_c=None,cov_x1_x0=None,cov_c_x0=None,
			alpha=None,beta=None,hostmass=None,
			M=None,x0=None,sigint=None,z=None,peczerr=0.00083,deltam=None):

	sf = -2.5/(x0*np.log(10.0))
	cov_mb_c = cov_c_x0*sf
	cov_mb_x1 = cov_x1_x0*sf
	mu_out = mb + x1*alpha - beta*c + 19.36
	invvars = 1.0 / (mberr**2.+ alpha**2. * x1err**2. + beta**2. * cerr**2. + \
						 2.0 * alpha * (cov_x1_x0*sf) - 2.0 * beta * (cov_c_x0*sf) - \
						 2.0 * alpha*beta * (cov_x1_c) )

	if deltam:
		if len(np.where(hostmass > 10)[0]):
			mu_out[hostmass > 10] += deltam/2.
		if len(np.where(hostmass < 10)[0]):
			mu_out[hostmass < 10] -= deltam/2.

	zerr = peczerr*5.0/np.log(10)*(1.0+z)/(z*(1.0+z/2.0))
	muerr_out = np.sqrt(1/invvars + zerr**2. + 0.055**2.*z**2.)
	if sigint: muerr_out = np.sqrt(muerr_out**2. + sigint**2.)
	return(mu_out,muerr_out)

def salt2mu_h0(x1=None,x1err=None,
			   c=None,cerr=None,
			   mb=None,mberr=None,
			   cov_x1_c=None,cov_x1_x0=None,cov_c_x0=None,
			   alpha=None,beta=None,hostmass=None,
			   M=None,x0=None,sigint=None,z=None,peczerr=0.00083,deltam=0):

	sf = -2.5/(x0*np.log(10.0))
	cov_mb_c = cov_c_x0*sf
	cov_mb_x1 = cov_x1_x0*sf
	mu_out = mb + x1*alpha - beta*c + 19.21
#19.233
	invvars = 1.0 / (mberr**2.+ alpha**2. * x1err**2. + beta**2. * cerr**2. + \
						 2.0 * alpha * (cov_x1_x0*sf) - 2.0 * beta * (cov_c_x0*sf) - \
						 2.0 * alpha*beta * (cov_x1_c) )

	if deltam:
		if len(np.where(hostmass > 10)[0]):
			mu_out[hostmass > 10] += deltam/2.
		if len(np.where(hostmass < 10)[0]):
			mu_out[hostmass < 10] -= deltam/2.
   
	zerr = 0
	h0err = 0.050
	#muerr_out = np.sqrt(1/invvars + zerr**2. + 0.055**2.*z**2. + h0err**2.)
	muerr_out = np.sqrt(1/invvars + h0err**2.)
	if sigint: muerr_out = np.sqrt(muerr_out**2. + sigint**2.)
	return(mu_out,muerr_out)
