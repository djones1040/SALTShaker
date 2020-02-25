import numpy as np
from salt3.training import saltresids
from inspect import signature
from functools import partial
from scipy.interpolate import splprep,splev,bisplev,bisplrep,interp1d,interp2d,RegularGridInterpolator,RectBivariateSpline
import logging
log=logging.getLogger(__name__)

__priors__=dict()
def prior(prior):
	"""Decorator to register a given function as a valid prior"""
	#Check that the method accepts 4 inputs: a SALTResids object, a width, parameter vector, model components
	assert(len(signature(prior).parameters)==4 or len(signature(prior).parameters)==5)
	__priors__[prior.__name__]=prior
	return prior

# TODO
# x0 > 0, M0 > 0
# -5 < tpk < 5

class SALTPriors:

	def __init__(self,SALTResidsObj):
		for k in SALTResidsObj.__dict__.keys():
			self.__dict__[k] = SALTResidsObj.__dict__[k]
		self.SALTModel = SALTResidsObj.SALTModel
		self.SALTModelDeriv = SALTResidsObj.SALTModelDeriv
		self.priors={ key: partial(__priors__[key],self) for key in __priors__}
		for prior in self.usePriors:
			result=self.priors[prior](1,self.guess,self.SALTModel(self.guess))
		self.numBoundResids=0
		for boundedparam in self.boundedParams:
			#width,bound,x,par
			result=self.boundedprior(0.1,(0,1),self.guess,boundedparam)
			self.numBoundResids += result[0].size
	@prior
	def m0m1prior(self,width,x,components):
		"""M1 should have no outer product with M0"""
		phase=self.phaseRegularizationPoints
		wave=self.waveRegularizationPoints
		components=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		
		m0sqr=(components[0]**2).sum()
		m1sqr=(components[1]**2).sum()
		m0m1=(components[0]*components[1]).sum()
		
		corr=np.array([m0m1/np.sqrt(m1sqr*m0sqr)])
		#Derivative with respect to m0
		jacobian=np.zeros((1,self.npar))
		jacobian[:,self.im0]= ((components[1]* m1sqr*m0sqr - m0m1*m1sqr*components[0] )[:,:,np.newaxis]*self.regularizationDerivs[0]).sum(axis=(0,1))/np.sqrt((m0sqr*m1sqr)**3)
		jacobian[:,self.im1]= ((components[0]* m1sqr*m0sqr - m0m1*m0sqr*components[1] )[:,:,np.newaxis]*self.regularizationDerivs[0]).sum(axis=(0,1))/np.sqrt((m0sqr*m1sqr)**3)
		residual=corr/width
		
		return corr/width,corr,jacobian/width	


	@prior
	def peakprior(self,width,x,components):
		""" At t=0, minimize time derivative of B-band lightcurve"""
		M0deriv=self.SALTModelDeriv(x,1,0,self.phase,self.wave)[0]
		int1d = interp1d(self.phase,M0deriv,axis=0,assume_sorted=True)
		m0Bderiv = np.sum(self.kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
			(self.wave[1]-self.wave[0])*self.fluxfactor['default']['B']
		bStdFlux=(10**((self.m0guess-27.5)/-2.5) )
		#This derivative is constant, and never needs to be recalculated, so I store it in a hidden attribute
		try:
			m0Bderivjac = self.__peakpriorderiv__.copy()
		except:
			m0Bderivjac= np.zeros(self.npar)
			passbandColorExp = self.kcordict['default']['Bpbspl']
			intmult = (self.wave[1]-self.wave[0])*self.fluxfactor['default']['B']
			for i in range(self.im0.size):
				waverange=self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
				phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-self.bsorder-1),i//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
				#Check if this filter is inside values affected by changes in knot i
				minlam=np.min(self.kcordict['default']['Bwave'][self.kcordict['default']['Btp'] > 0.01])
				maxlam=np.max(self.kcordict['default']['Bwave'][self.kcordict['default']['Btp'] > 0.01])

				if waverange[0] > maxlam or waverange[1] < minlam:
					pass
				if (0>=phaserange[0] ) & (0<=phaserange[1]):
					#Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
					#Integrate only over wavelengths within the relevant range
					inbounds=(self.wave>waverange[0]) & (self.wave<waverange[1])
					derivInterp = interp1d(self.phase,bisplev(self.phase,self.wave[inbounds],(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder),dx=1),axis=0,kind=self.interpMethod,bounds_error=False,fill_value="extrapolate",assume_sorted=True)
					m0Bderivjac[self.im0[i]] = np.sum( passbandColorExp[inbounds] * derivInterp(0))*intmult 
			self.__peakpriorderiv__=m0Bderivjac.copy()
		#import pdb;pdb.set_trace()
		value=m0Bderiv/bStdFlux
		return value/width,value, m0Bderivjac/(bStdFlux*width)
	
	def satisfyDefinitions(self,X,components):
		"""Ensures that the definitions of M1,M0,x0,x1 are satisfied"""
		X=X.copy()

		int1d = interp1d(self.phase,components[0],axis=0,assume_sorted=True)
		m0Bflux = np.sum(self.kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
			(self.wave[1]-self.wave[0])*self.fluxfactor['default']['B']
			
		int1d = interp1d(self.phase,components[1],axis=0,assume_sorted=True)
		m1Bflux = np.sum(self.kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
			(self.wave[1]-self.wave[0])*self.fluxfactor['default']['B']
		ratio=m1Bflux/m0Bflux
		#Define M1 to have no effect on B band at t=0
		X[self.ix0]*=1+ratio*X[self.ix1]
		X[self.ix1]/=1+ratio*X[self.ix1]
		X[self.im1]-=ratio*X[self.im0]
		
		#Define x1 to have mean 0
		#m0 at peak is not modified, since m1B at peak is defined as 0
		#Thus does not need to be recalculated for the last definition
		X[self.im0]+= np.mean(X[self.ix1])*X[self.im1]
		X[self.ix1]-=np.mean(X[self.ix1])
		
		#Define x1 to have std deviation 1
		x1std = np.std(X[self.ix1])
		if x1std == x1std and x1std != 0.0:
			X[self.im1]*= x1std
			X[self.ix1]/= x1std
			
		#Define m0 to have a standard B-band magnitude at peak
		bStdFlux=(10**((self.m0guess-27.5)/-2.5) )
		X[self.im0]*=bStdFlux/m0Bflux 
		X[self.ix0]*=m0Bflux /bStdFlux
		return X
		
	@prior
	def m0prior(self,width,x,components):
		"""Prior on the magnitude of the M0 component at t=0"""
		int1d = interp1d(self.phase,components[0],axis=0,assume_sorted=True)
		m0Bflux = np.sum(self.kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
			(self.wave[1]-self.wave[0])*self.fluxfactor['default']['B']
		#m0Bflux=np.clip(m0Bflux,(10**((self.m0guess-27.5)/-2.5) ),None)
		m0B= -2.5*np.log10(m0Bflux)+27.5
		bStdFlux=(10**((self.m0guess-27.5)/-2.5) )
		residual = (m0Bflux-bStdFlux) / (width*bStdFlux)
		#This derivative is constant, and never needs to be recalculated, so I store it in a hidden attribute
		try:
			fluxDeriv= self.__m0priorfluxderiv__.copy()
		except:
			fluxDeriv= np.zeros(self.npar)
			passbandColorExp = self.kcordict['default']['Bpbspl']
			intmult = (self.wave[1] - self.wave[0])*self.fluxfactor['default']['B']
			for i in range(self.im0.size):
				waverange=self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
				phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-self.bsorder-1),i//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
				#Check if this filter is inside values affected by changes in knot i
				minlam=np.min(self.kcordict['default']['Bwave'][self.kcordict['default']['Btp'] > 0.01])
				maxlam=np.max(self.kcordict['default']['Bwave'][self.kcordict['default']['Btp'] > 0.01])
				if waverange[0] > maxlam or waverange[1] < minlam:
					pass
				if (0>=phaserange[0] ) & (0<=phaserange[1]):
					#Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
					#Integrate only over wavelengths within the relevant range
					inbounds=(self.wave>waverange[0]) & (self.wave<waverange[1])
					derivInterp = interp1d(self.phase,self.spline_derivs[:,inbounds,i],axis=0,kind=self.interpMethod,bounds_error=False,fill_value="extrapolate",assume_sorted=True)
					fluxDeriv[self.im0[i]] = np.sum( passbandColorExp[inbounds] * derivInterp(0))*intmult 
			self.__m0priorfluxderiv__=fluxDeriv.copy()
		
		jacobian=fluxDeriv/ (bStdFlux* width)
		return residual,m0B,jacobian



		
	@prior
	def m1prior(self,width,x,components):
		"""M1 should have zero flux at t=0 in the B band"""
		pbspl=(self.kcordict['default']['Bpbspl']*(self.wave[1]-self.wave[0])*self.fluxfactor['default']['B'])[np.newaxis,:]
		int1d = interp1d(self.phase,components[1],axis=0,assume_sorted=True)
		m1flux = np.sum(pbspl*int1d([0]), axis=1)
		bStdFlux=(10**((self.m0guess-27.5)/-2.5) )
		residual = (m1flux) / (width*bStdFlux)
		#This derivative is constant, and never needs to be recalculated, so I store it in a hidden attribute
		try:
			fluxDeriv= self.__m1priorfluxderiv__.copy()
		except:
			fluxDeriv= np.zeros((pbspl.shape[0],self.npar))
			for i in range(self.im1.size):
				waverange=self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
				phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-self.bsorder-1),i//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
				#Check if this filter is inside values affected by changes in knot i
				minlam=min([np.min(self.kcordict['default'][filt+'wave'][self.kcordict['default'][filt+'tp'] > 0.01]) for filt in ['B']]) 
				maxlam=max([np.max(self.kcordict['default'][filt+'wave'][self.kcordict['default'][filt+'tp'] > 0.01]) for filt in ['B']]) 
				if waverange[0] > maxlam or waverange[1] < minlam:
					pass
				if (0>=phaserange[0] ) & (0<=phaserange[1]):
					#Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
					#Integrate only over wavelengths within the relevant range
					inbounds=(self.wave>waverange[0]) & (self.wave<waverange[1])
					derivInterp = interp1d(self.phase,self.spline_derivs[:,inbounds,i],axis=0,kind=self.interpMethod,bounds_error=False,fill_value="extrapolate",assume_sorted=True)
					fluxDeriv[:,self.im1[i]] = np.sum( pbspl[:,inbounds]* derivInterp([0]),axis=1) 
			self.__m1priorfluxderiv__=fluxDeriv.copy()
		
		jacobian=fluxDeriv/ (bStdFlux* width)
		
		return residual,m1flux/bStdFlux,jacobian
	
	@prior
	def colormean(self,width,x,components):
		"""Prior such that the mean of the color population is 0"""
		mean=np.mean(x[self.ic])
		residual = mean/width
		jacobian=np.zeros(self.npar)
		jacobian[self.ic] = 1/len(self.datadict.keys())/width
		return residual,mean,jacobian

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
			jacobian[self.ix1] = (x1s-x1mean)/(x1s.size*x1std*width)
		return residual,x1std,jacobian

	@prior
	def m0endalllam(self,width,x,components):
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
	def m1endalllam(self,width,x,components):
		"""Prior such that at early times there is no flux"""
		upper,lower=components[0].shape[1],0
		value=components[1][0,lower:upper]
		residual = value/width
		jacobian=np.zeros((upper-lower,self.npar))
		for i in range((self.waveknotloc.size-self.bsorder-1)):
			jacobian[lower:upper,self.im1[i]] = self.spline_derivs[0,lower:upper,i]
		jacobian/=width
		return residual,value,jacobian	

	def boundedprior(self,width,bound,x,par):
		"""Flexible prior that sets Gaussian bounds on parameters"""
		#import pdb;pdb.set_trace()

		lbound,ubound = bound
		
		iPar = self.__dict__['i%s'%par]
		
		iOut = (x[iPar] < lbound) | (x[iPar] > ubound)
		iLow = (x[iPar] < lbound)
		iHigh = (x[iPar] > ubound)
		residual = np.zeros(iPar.size)
		residual[iLow] = (x[iPar][iLow]-lbound)**2./(2*width**2.)
		residual[iHigh] = (x[iPar][iHigh]-ubound)**2./(2*width**2.)

		jacobian = np.zeros((x[iPar].size,self.npar))
		jacobian[iLow,iPar[iLow]] = (x[iPar][iLow]-lbound)/(width**2.)
		jacobian[iHigh,iPar[iHigh]] = (x[iPar][iHigh]-ubound)/(width**2.)
		
		return residual,x[iPar],jacobian	

	def getBounds(self,bounds,boundparams):
		lower=np.ones(self.npar)*-np.inf
		upper=np.ones(self.npar)*np.inf
		for bound,par in zip(bounds,boundparams):
			lbound,ubound,width = bound
			iPar = self.__dict__['i%s'%par]
			lower[iPar]=lbound
			upper[iPar]=ubound
		return lower,upper
		
	def priorResids(self,priors,widths,x):
		"""Given a list of names of priors and widths returns a residuals vector, list of prior values, and Jacobian """
		components = self.SALTModel(x)
		results=[]
		debugstring='Prior values are '
		for prior,width in zip(priors,widths):
			try:
				priorFunction=self.priors[prior]
			except:
				raise ValueError('Invalid prior supplied: {}'.format(prior)) 
			results+=[priorFunction(width,x,components)]
			if results[-1][0].size==1:
				debugstring+='{}: {:.2e},'.format(prior,float(results[-1][1]))
				#debugstring+=f'{prior}: '+' '.join(['{:.2e}'.format(val) for val in results[-1][1]])+','
		log.debug(debugstring)
		residuals,values,jacobian=zip(*results)
		return np.concatenate([np.array([x]) if x.shape==() else x for x in residuals]),np.concatenate([np.array([x]) if x.shape==() else x for x in values]),np.concatenate([x if len(x.shape)==2 else x[np.newaxis,:] for x in jacobian])


	def BoundedPriorResids(self,bounds,boundparams,x):
		"""Given a list of names of priors and widths returns a residuals vector, list of prior values, and Jacobian """

		components = self.SALTModel(x)
		residuals=np.zeros(self.numBoundResids)
		jacobian=np.zeros((self.numBoundResids,self.npar))
		values=np.zeros(self.numBoundResids)
		idx=0
		debugstring='Values outside bounds: '
		for bound,par in zip(bounds,boundparams):
			result=self.boundedprior(bound[-1],(bound[0],bound[1]),x,par)
			numResids=result[0].size
			residuals[idx:idx+numResids],values[idx:idx+numResids],\
				jacobian[idx:idx+numResids,:]=result
			if result[0].any():
				debugstring+='{} {}, '.format(par,np.nonzero(result[0])[0].size)
			idx+=numResids
		if residuals.any():
			log.debug(debugstring[:-1])
		return residuals,values,jacobian
