import numpy as np
from salt3.training import saltresids
from inspect import signature
from functools import partial
from scipy.interpolate import splprep,splev,bisplev,bisplrep,interp1d,interp2d,RegularGridInterpolator,RectBivariateSpline

__priors__=dict()
def prior(prior):
	"""Decorator to register a given function as a valid prior"""
	#Check that the method accepts 4 inputs: a SALTResids object, a width, parameter vector, model components
	assert(len(signature(prior).parameters)==4 or len(signature(prior).parameters)==5)
	__priors__[prior.__name__]=prior
	return prior

__boundedpriors__=dict()
def boundedprior(boundedprior):
	"""Decorator to register a given function as a valid prior"""
	#Check that the method accepts 4 inputs: a SALTResids object, a width, parameter vector, model components
	assert(len(signature(boundedprior).parameters)==5 or len(signature(boundedprior).parameters)==6)
	__boundedpriors__[boundedprior.__name__]=boundedprior
	return boundedprior


# TODO
# x0 > 0, M0 > 0
# -5 < tpk < 5

class SALTPriors:

	def __init__(self,SALTResidsObj):
		for k in SALTResidsObj.__dict__.keys():
			self.__dict__[k] = SALTResidsObj.__dict__[k]
		self.SALTModel = SALTResidsObj.SALTModel

		self.priors={ key: partial(__priors__[key],self) for key in __priors__}
		self.boundedPriors={ key: partial(__boundedpriors__[key],self) for key in __boundedpriors__}
		for prior in self.priors:
			result=self.priors[prior](1,self.guess,self.SALTModel(self.guess))
			try:
				self.priors[prior].numResids=result[0].size
			except:
				self.priors[prior].numResids=1
		self.numPriorResids=sum([self.priors[x].numResids for x in self.priors])
		self.numBoundResids=0
		for boundedparam in self.BoundedParams:
			#width,bound,x,par
			result=self.boundedPriors[boundedprior](0.1,(0,1),self.guess,boundedparam)
			self.numPriorResids += result[0].size
			self.numBoundResids += result[0].size
			
		#self.numPriorResids += sum([self.boundedpriors[x].numResids for x in self.boundedpriors])

	#@prior
	def peakprior(self,width,x,components):
		wave=self.wave[self.bbandoverlap]
		lightcurve=np.sum(self.bbandpbspl[np.newaxis,:]*components[0][:,self.bbandoverlap],axis=1)
		# from D'Arcy - disabled for now!!	(barfs if model goes crazy w/o regularization)
		#maxPhase=np.argmax(lightcurve)
		#finePhase=np.arange(self.phase[maxPhase-1],self.phase[maxPhase+1],0.1)
		finePhase=np.arange(self.phase[self.maxPhase-1],self.phase[self.maxPhase+1],0.1)
		fineGrid=self.SALTModel(x,evaluatePhase=finePhase,evaluateWave=wave)[0]
		lightcurve=np.sum(self.bbandpbspl[np.newaxis,:]*fineGrid,axis=1)

		value=finePhase[np.argmax(lightcurve)]	
		#Need to write the derivative with respect to parameters
		return value/width,value,np.zeros(self.npar)
	
	@prior
	def m0prior(self,width,x,components):
		"""Prior on the magnitude of the M0 component at t=0"""
		int1d = interp1d(self.phase,components[0],axis=0,assume_sorted=True)
		m0Bflux = np.sum(self.kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
			self.kcordict['default']['Bdwave']*self.kcordict['default']['fluxfactor']
		m0B = -2.5*np.log10(m0Bflux) + 27.5
		residual = (m0B-self.m0guess) / width
		#This derivative is constant, and never needs to be recalculated, so I store it in a hidden attribute
		try:
			fluxDeriv= self.__m0priorfluxderiv__.copy()
		except:
			fluxDeriv= np.zeros(self.npar)
			passbandColorExp = self.kcordict['default']['Bpbspl']
			intmult = (self.wave[1] - self.wave[0])*self.kcordict['default']['fluxfactor']
			for i in range(self.im0.size):
				waverange=self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
				phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-self.bsorder-1),i//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
				#Check if this filter is inside values affected by changes in knot i
				if waverange[0] > self.kcordict['default']['maxlam'] or waverange[1] < self.kcordict['default']['minlam']:
					pass
				if (0>=phaserange[0] ) & (0<=phaserange[1]):
					#Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
					#Integrate only over wavelengths within the relevant range
					inbounds=(self.wave>waverange[0]) & (self.wave<waverange[1])
					derivInterp = interp1d(self.phase,self.spline_derivs[:,inbounds,i],axis=0,kind=self.interpMethod,bounds_error=False,fill_value="extrapolate",assume_sorted=True)
					fluxDeriv[self.im0[i]] = np.sum( passbandColorExp[inbounds] * derivInterp(0))*intmult 
			self.__m0priorfluxderiv__=fluxDeriv.copy()
		
		jacobian=-fluxDeriv* (2.5 / (np.log(10) *m0Bflux * width))
		return residual,m0B,jacobian
		
	@prior
	def x1mean(self,width,x,components):
		"""Prior such that the mean of the x1 population is 0"""
		x1mean=np.mean(x[self.ix1])
		residual = x1mean/width
		jacobian=np.zeros(self.npar)
		jacobian[self.ix1] = 1/len(self.datadict.keys())/width
		return residual,x1mean,jacobian
		
	@prior
	def x1std(self,width,x,components):
		"""Prior such that the standard deviation of the x1 population is 1"""
		x1s=x[self.ix1]
		x1mean=np.mean(x1s)
		x1std=np.std(x1s)
		residual = (x1std-1)/width
		jacobian=np.zeros(self.npar)
		if x1std!=0:
			jacobian[self.ix1] = (x1s-x1mean)/(len(self.datadict.keys())*x1std*width)
		return residual,x1std,jacobian

	@prior
	def m0endprior_alllam(self,width,x,components):
		"""Prior such that at early times there is no flux"""
		upper,lower=components[0].shape[1],0
		value=components[0][0,lower:upper]
		residual = value/width
		jacobian=np.zeros((upper-lower,self.npar))
		for i in range((self.waveknotloc.size-self.bsorder-1)):
			jacobian[lower:upper,self.im0[i]] = self.spline_derivs[0,lower:upper,i]
		jacobian/=width
		return residual,value,jacobian

	@prior
	def m1endprior_alllam(self,width,x,components):
		"""Prior such that at early times there is no flux"""
		upper,lower=components[0].shape[1],0
		value=components[1][0,lower:upper]
		residual = value/width
		jacobian=np.zeros((upper-lower,self.npar))
		for i in range((self.waveknotloc.size-self.bsorder-1)):
			jacobian[lower:upper,self.im1[i]] = self.spline_derivs[0,lower:upper,i]
		jacobian/=width
		return residual,value,jacobian	

	@boundedprior
	def boundedprior(self,width,bound,x,par):
		"""Flexible prior that sets Gaussian bounds on parameters"""

		lbound,ubound = bound
		
		iPar = self.__dict__['i%s'%par]
		
		iOut = (x[iPar] < lbound) | (x[iPar] > ubound)
		iLow = (x[iPar] < lbound)
		iHigh = (x[iPar] > ubound)
		residual = np.zeros(iPar.size)
		residual[iLow] = (x[iPar][iLow]-lbound)**2./(2*width**2.)
		residual[iHigh] = (x[iPar][iHigh]-ubound)**2./(2*width**2.)

		jacobian = np.zeros(len(x[iPar][iOut]),iPar.size)
		jacobian[iLow,:] = (x[iPar][iLow]-lbound)/(width**2.)
		jacobian[iHigh,:] = (x[iPar][iHigh]-ubound)/(width**2.)
		
		return residual,x[iPar],jacobian	

		
	def priorResids(self,priors,widths,x):
		"""Given a list of names of priors and widths returns a residuals vector, list of prior values, and Jacobian """

		alllam_vals = range(0,self.im0.size)
		components = self.SALTModel(x)
		residuals=np.zeros(self.numPriorResids)
		jacobian=np.zeros((self.numPriorResids,self.npar))
		values=np.zeros(self.numPriorResids)
		idx=0
		for prior,width in zip(priors,widths):
			try:
				priorFunction=self.priors[prior]
			except:
				raise ValueError('Invalid prior supplied: {}'.format(prior)) 
			
			residuals[idx:idx+priorFunction.numResids],values[idx:idx+priorFunction.numResids],\
				jacobian[idx:idx+priorFunction.numResids,:]=priorFunction(width,x,components)
			idx+=priorFunction.numResids
		return residuals,values,jacobian


	def BoundedPriorResids(self,bounds,boundparams,x):
		"""Given a list of names of priors and widths returns a residuals vector, list of prior values, and Jacobian """

		alllam_vals = range(0,self.im0.size)
		components = self.SALTModel(x)
		residuals=np.zeros(self.numBoundResids)
		jacobian=np.zeros((self.numBoundResids,self.npar))
		values=np.zeros(self.numBoundResids)
		idx=0
		for bound,par in zip(bounds,boundparams):
			try:
				boundFunction=self.boundedprior[bound]
			except:
				raise ValueError('Invalid bound supplied: {}'.format(bound))

			residuals[idx:idx+priorFunction.numResids],values[idx:idx+boundFunction.numResids],\
				jacobian[idx:idx+priorFunction.numResids,:]=boundFunction(bound[-1],(bound[0],bound[1]),x,par)
			idx+=boundFunction.numResids
		return residuals,values,jacobian
