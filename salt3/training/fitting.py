#!/usr/bin/env python

from scipy.optimize import minimize, least_squares, differential_evolution
import numpy as np

class fitting:
	def __init__(self,n_components,n_colorpars,n_phaseknots,n_waveknots,datadict):

		self.n_phaseknots = n_phaseknots
		self.n_waveknots = n_waveknots
		self.n_components = n_components
		self.n_colorpars = n_colorpars
		self.datadict = datadict
		
	def mcmc(self,saltfitter,guess,SNpars,SNparlist,
			 n_processes,n_mcmc_steps,n_burnin_mcmc,
			 init=False):

		if init: saltfitter.onlySNpars = True
		saltfitter.SNpars = SNpars
		saltfitter.SNparlist = SNparlist
		saltfitter.mcmc = True
		saltfitter.debug = False
		saltfitter.fitstrategy = 'mcmc'
		
		x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams = \
			saltfitter.mcmcfit(guess,n_mcmc_steps,n_burnin_mcmc)
		return x,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,clscaterr,SNParams,'simple MCMC was successful'
