#!/usr/bin/env python

from scipy.optimize import minimize, least_squares, differential_evolution
import numpy as np

class fitting:
	def __init__(self,n_components,n_colorpars,n_phaseknots,n_waveknots,datadict,x_init,initparlist,parlist):

		self.n_phaseknots = n_phaseknots
		self.n_waveknots = n_waveknots
		self.n_components = n_components
		self.n_colorpars = n_colorpars
		self.datadict = datadict
		self.x_init = x_init
		self.initparlist = initparlist
		self.parlist = parlist
		
	def least_squares(self,saltfitter,guess,SNpars,SNparlist,n_processes,fitmethod):

		bounds_lower,bounds_upper = [],[]
		for i in range(len(guess)):
			if guess[i] != 0:
				bounds_lower += [guess[i]*10**(-0.4*0.5)]
				bounds_upper += [guess[i]*10**(0.4*0.5)]
			else:
				bounds_lower += [0]
				bounds_upper += [np.max(guess)*1e-20]
		#import pdb; pdb.set_trace()
				
		#bounds_lower = [0]*(self.n_components*self.n_phaseknots*self.n_waveknots + self.n_colorpars)
		#for k in self.datadict.keys():
		#	bounds_lower += [self.x_init[self.initparlist == 'x0_%s'%k]*1e-1,-5,-1,-5]
		#bounds_upper = [np.inf]*(self.n_components*self.n_phaseknots*self.n_waveknots+self.n_colorpars)
		#for k in self.datadict.keys():
		#	bounds_upper += [self.x_init[self.initparlist == 'x0_%s'%k]*1e1,5,1,5]

		bounds = (bounds_lower,bounds_upper)

		if n_processes>1:
			with multiprocessing.Pool(n_processes) as pool:
				md = least_squares(saltfitter.chi2fit,guess,method=fitmethod,bounds=bounds,
								   args=(None,None,pool,SNpars,SNparlist,False,False))
		else:
			md = least_squares(saltfitter.chi2fit,guess,method=fitmethod,bounds=bounds,
							   ftol=1e-16,gtol=1e-16,xtol=1e-16,
							   args=(None,None,None,SNpars,SNparlist,False,False),
							   verbose=2,tr_solver='exact')#,max_nfev=1)
			
		for k in self.datadict.keys():
			md.x[self.parlist == 'x0_%s'%k],md.x[self.parlist == 'x1_%s'%k],\
				md.x[self.parlist == 'c_%s'%k],md.x[self.parlist == 'tpkoff_%s'%k] = \
				self.x_init[self.initparlist == 'x0_%s'%k],self.x_init[self.initparlist == 'x1_%s'%k],\
				self.x_init[self.initparlist == 'c_%s'%k],self.x_init[self.initparlist == 'tpkoff_%s'%k]
			
		phase,wave,M0,M1,clpars,SNParams = \
			saltfitter.getPars(md.x)
			
		return md.x,phase,wave,M0,M1,clpars,SNParams,md.message

	def minimize(self,saltfitter,guess,SNpars,SNparlist,n_processes,fitmethod):

		# 1 mag bounds on M0
		# this won't work for M1/color obviously, needs to be tweaked
		#bounds = ()
		#for i in range(len(guess)):
		#	if guess[i] != 0:
		#		bounds += ((guess[i]*10**(-0.4*1e-5),guess[i]*10**(0.4*1e-5)),)
		#	else:
		#		bounds += ((0,1e-22),)#np.max(guess)),)

		bounds_lower,bounds_upper = [],[]
		for i in range(len(guess)):
			if guess[i] != 0:
				bounds_lower += [guess[i]*10**(-0.4*3)]
				bounds_upper += [guess[i]*10**(0.4*3)]
			else:
				bounds_lower += [0]
				bounds_upper += [np.max(guess)*1e-20]
				
		#bounds_lower = [0]*(self.n_components*self.n_phaseknots*self.n_waveknots + self.n_colorpars)
		#for k in self.datadict.keys():
		#	bounds_lower += [self.x_init[self.initparlist == 'x0_%s'%k]*1e-1,-5,-1,-5]
		#bounds_upper = [np.inf]*(self.n_components*self.n_phaseknots*self.n_waveknots+self.n_colorpars)
		#for k in self.datadict.keys():
		#	bounds_upper += [self.x_init[self.initparlist == 'x0_%s'%k]*1e1,5,1,5]

		bounds = (bounds_lower,bounds_upper)
		
		#bounds = ((-np.inf,np.inf),)*self.n_phaseknots*self.n_waveknots + ((-np.inf,np.inf),)*\
		#		 ((self.n_components-1)*self.n_phaseknots*self.n_waveknots + self.n_colorpars)
		#for k in self.datadict.keys():
		#	bounds += ((self.x_init[self.initparlist == 'x0_%s'%k]*1e-1,self.x_init[self.initparlist == 'x0_%s'%k]*1e1),
		#				(-np.inf,np.inf),(-np.inf,np.inf),(-5,5))
		# import pdb; pdb.set_trace()
		
		if n_processes > 1:
			md = minimize(saltfitter.chi2fit,guess,
						  bounds=bounds,
						  method=fitmethod,
						  args=(None,None,Pool,False,False),
						  options={'maxiter':100000,'maxfev':100000,'maxfun':100000})
		else:
			# Powell?
			#import pdb; pdb.set_trace()
			md = minimize(saltfitter.chi2fit,guess,
						  bounds=bounds,
		 				  method=fitmethod,
		  				  args=(None,None,None,SNpars,SNparlist,False,False),
						  options={'maxiter':1000})

		for k in self.datadict.keys():
			md.x[self.parlist == 'x0_%s'%k],md.x[self.parlist == 'x1_%s'%k],\
				md.x[self.parlist == 'c_%s'%k],md.x[self.parlist == 'tpkoff_%s'%k] = \
				self.x_init[self.initparlist == 'x0_%s'%k],self.x_init[self.initparlist == 'x1_%s'%k],\
				self.x_init[self.initparlist == 'c_%s'%k],self.x_init[self.initparlist == 'tpkoff_%s'%k]
			
		phase,wave,M0,M1,clpars,SNParams = \
			saltfitter.getPars(md.x)
			
		return md.x,phase,wave,M0,M1,clpars,SNParams,md.message

	def emcee(self,saltfitter,guess,SNpars,SNparlist,n_processes):

		saltfitter.mcmc = True
		
		import emcee
		ndim, nwalkers = len(self.parlist), 2*len(self.parlist)
		pos = [guess + 1e-21*np.random.randn(ndim) for i in range(nwalkers)]
		sampler = emcee.EnsembleSampler(nwalkers, ndim, saltfitter.chi2fit,
										args=(None,None,None,SNpars,SNparlist,False,False),threads=n_processes)
		sampler.run_mcmc(pos, 200)
		samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

		phase,wave,M0,M1,clpars,SNParams = \
			saltfitter.getEmceePars(md.x)

		message = 'emcee finished successfully'
		
		return phase,wave,M0,M1,clpars,SNParams,message

	def pymultinest(self,saltfitter,guess,SNpars,SNparlist,n_processes):

		saltfitter.mcmc = True
		
		import pymultinest
		saltfitter.m0min = np.min(np.where(saltfitter.parlist == 'm0')[0])
		saltfitter.m0max = np.max(np.where(saltfitter.parlist == 'm0')[0])
		saltfitter.SNpars = SNpars
		saltfitter.SNparlist = SNparlist
		saltfitter.mcmc = True
		saltfitter.debug = False
		#pymultinest.run(saltfitter.chi2fit, saltfitter.prior, len(guess),
		#				outputfiles_basename='output/pmntest_1_',
		#				resume = False, verbose = True)
		
		result = pymultinest.solve(#LogLikelihood=myloglike, Prior=myprior,
								   LogLikelihood=saltfitter.chi2fit, Prior=saltfitter.prior,
								   n_dims=len(guess),
								   outputfiles_basename='output/pmntest_1_',
								   verbose=True,resume=False,n_live_points=150,#max_iter=300,
								   multimodal=False,importance_nested_sampling=True,
								   #n_iter_before_update=50,
								   evidence_tolerance=5,
								   sampling_efficiency=1)
		import pdb; pdb.set_trace()
		phase,wave,M0,M1,clpars,SNParams = \
			saltfitter.getParsMCMC(result['samples'].transpose())
		
		message = 'pymultinest finished successfully'
		
		return phase,wave,M0,M1,clpars,SNParams,message
		
	def diffevol(self,saltfitter,guess,SNpars,SNparlist,n_processes,fitmethod):

		bounds = ((0,np.max(self.x_init[self.initparlist == 'x0_%s'%(list(self.datadict.keys())[0])])),)*\
				 (self.n_components*self.n_phaseknots*self.n_waveknots + self.n_colorpars)
		for k in self.datadict.keys():
			bounds += ((self.x_init[self.initparlist == 'x0_%s'%k]*1e-1,self.x_init[self.initparlist == 'x0_%s'%k]*1e1),
						(-5,5),(-1,1),(-5,5))
		
		md = differential_evolution(saltfitter.chi2fit,
									bounds=bounds,
									strategy=fitmethod,
		  							args=(None,None,None,SNpars,SNparlist,False,False))

		for k in self.datadict.keys():
			md.x[self.parlist == 'x0_%s'%k],md.x[self.parlist == 'x1_%s'%k],\
				md.x[self.parlist == 'c_%s'%k],md.x[self.parlist == 'tpkoff_%s'%k] = \
				self.x_init[self.initparlist == 'x0_%s'%k],self.x_init[self.initparlist == 'x1_%s'%k],\
				self.x_init[self.initparlist == 'c_%s'%k],self.x_init[self.initparlist == 'tpkoff_%s'%k]
		
		phase,wave,M0,M1,clpars,SNParams = \
			saltfitter.getPars(md.x)
			
		return md.x,phase,wave,M0,M1,clpars,SNParams,md.message

	def hyperopt(self,saltfitter,guess,m0knots,SNpars,SNparlist,n_processes):

		from hyperopt import hp, tpe, fmin

		space = []
		m0count,m1count,clcount = 0,0,0
		for i in self.parlist:
			if i.startswith('x0'):
				x0guess = guess[self.parlist == i]
				#10**(-0.4*(-2.5*np.log10(np.max(self.datadict[i.split('_')[1]]['photdata']['fluxcal'])*1e12)+27.5 + 19.36))
				space += [hp.uniform(i,x0guess*0.99,x0guess*1.01)]
			elif i.startswith('x1'):
				space += [hp.uniform(i,-3,3)]
			elif i.startswith('c'):
				space += [hp.uniform(i,-0.3,0.3)]
			elif i.startswith('tpkoff'):
				space += [hp.uniform(i,-5,5)]
			elif i.startswith('m0'):
				space += [hp.uniform('m0_%i'%m0count,0,m0knots[m0count]*1e2+0.1)]
				m0count += 1
			else: 
				import pdb; pdb.set_trace()
				
		best = fmin(saltfitter.chi2fit,
					space = space, #hp.normal('x', 4.9, 0.5),
					algo=tpe.suggest, 
					max_evals = 2000)


	def bobyqa(self,saltfitter,guess,SNpars,SNparlist,n_processes,fitmethod):
		import pybobyqa
		import dfols
		
		# 1 mag bounds on M0
		bounds_lower,bounds_upper = [],[]
		for i in range(len(guess)):
			if guess[i] != 0:
				bounds_lower += [guess[i]*10**(-0.4*3)]
				bounds_upper += [guess[i]*10**(0.4*3)]
			else:
				bounds_lower += [0]
				bounds_upper += [np.max(guess)*1e-20]
		bounds = (np.array(bounds_lower),np.array(bounds_upper))
		
		if n_processes > 1:
			md = pybobyqa.solve(saltfitter.chi2fit,guess,
								bounds=bounds,
		  						args=(None,None,None,SNpars,SNparlist,False,False),
								maxfun=10000,scaling_within_bounds=True)
		else:
			# Powell?
			md = dfols.solve(saltfitter.chi2fit,guess,
							 bounds=bounds,
		  					 args=(None,None,None,SNpars,SNparlist,False,False),
							 maxfun=10000,scaling_within_bounds=True)
			import pdb; pdb.set_trace()
			
		for k in self.datadict.keys():
			md.x[self.parlist == 'x0_%s'%k],md.x[self.parlist == 'x1_%s'%k],\
				md.x[self.parlist == 'c_%s'%k],md.x[self.parlist == 'tpkoff_%s'%k] = \
				self.x_init[self.initparlist == 'x0_%s'%k],self.x_init[self.initparlist == 'x1_%s'%k],\
				self.x_init[self.initparlist == 'c_%s'%k],self.x_init[self.initparlist == 'tpkoff_%s'%k]
			
		phase,wave,M0,M1,clpars,SNParams = \
			saltfitter.getPars(md.x)
			
		return md.x,phase,wave,M0,M1,clpars,SNParams,md.message
		
	def mcmc(self,saltfitter,guess,SNpars,SNparlist,n_processes):

		saltfitter.SNpars = SNpars
		saltfitter.SNparlist = SNparlist
		saltfitter.mcmc = True
		saltfitter.debug = False
		saltfitter.fitstrategy = 'mcmc'
		
		phase,wave,M0,M1,clpars,SNParams = \
			saltfitter.mcmcfit(guess,10000)
		
		return phase,wave,M0,M1,clpars,SNParams,'simple MCMC was successful'
