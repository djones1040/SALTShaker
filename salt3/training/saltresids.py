from salt3.util.synphot import synphot
from salt3.training import init_hsiao

from sncosmo.models import StretchSource
from sncosmo.salt2utils import SALT2ColorLaw
from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
from sncosmo.utils import integration_grid

import scipy.stats as ss
from scipy.optimize import minimize, least_squares
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from scipy.special import factorial
from scipy.interpolate import splprep,splev,bisplev,bisplrep,interp1d,interp2d,RegularGridInterpolator,RectBivariateSpline
from scipy.integrate import trapz
from scipy import linalg 

import numpy as np
from numpy.random import standard_normal
from numpy.linalg import slogdet

from astropy.cosmology import Planck15 as cosmo
from multiprocessing import Pool, get_context
from inspect import signature
from functools import partial
from itertools import starmap

import time
import pylab as plt
import extinction
import copy
import warnings
import pyParz

_SCALE_FACTOR = 1e-12
_B_LAMBDA_EFF = np.array([4302.57])	 # B-band-ish wavelength
_V_LAMBDA_EFF = np.array([5428.55])	 # V-band-ish wavelength
warnings.simplefilter('ignore',category=FutureWarning)


__priors__=dict()
def prior(prior):
	"""Decorator to register a given function as a valid prior"""
	#Check that the method accepts 4 inputs: a SALTResids object, a width, parameter vector, model components
	assert(len(signature(prior).parameters)==4 or len(signature(prior).parameters)==5)
	__priors__[prior.__name__]=prior
	return prior
	
class SALTResids:
	def __init__(self,guess,datadict,parlist,**kwargs):
		
		assert type(parlist) == np.ndarray
		self.blah = False
		self.debug = False
		self.nstep = 0
		self.parlist = parlist
		self.npar = len(parlist)
		self.datadict = datadict
		#ndata = 0
		#for sn in self.datadict.keys():
		#	photdata = self.datadict[sn]['photdata']
		#	for flt in np.unique(photdata['filt']):
		#		ndata += len(photdata['filt'][photdata['filt'] == flt])
		#self.n_data = ndata
		
		self.bsorder=3
		self.guess = guess
		self.nsn = len(self.datadict.keys())
		
		for key, value in kwargs.items(): 
			self.__dict__[key] = value

		if self.usePriors:
			self.usePriors=self.usePriors.split(',')
			self.priorWidths=[float(x) for x in self.priorWidths.split(',')]
		else:
			self.usePriors = []
			self.priorWidths = []
			
		# pre-set some indices
		self.m0min = np.min(np.where(self.parlist == 'm0')[0])
		self.m0max = np.max(np.where(self.parlist == 'm0')[0])
		self.errmin = tuple([np.min(np.where(self.parlist == 'modelerr_{}'.format(i))[0]) for i in range(self.n_components)]) 
		self.errmax = tuple([np.max(np.where(self.parlist == 'modelerr_{}'.format(i))[0]) for i in range(self.n_components)]) 
		self.corrcombinations=sum([[(i,j) for j in range(i+1,self.n_components)]for i in range(self.n_components)] ,[])
		self.corrmin = tuple([np.min(np.where(self.parlist == 'modelcorr_{}{}'.format(i,j))[0]) for i,j in self.corrcombinations]) 
		self.corrmax = tuple([np.max(np.where(self.parlist == 'modelcorr_{}{}'.format(i,j))[0]) for i,j in self.corrcombinations]) 
		self.ix1 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x1')])
		self.ix0 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x0')])
		self.ic	 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('c_')])
		self.itpk = np.array([i for i, si in enumerate(self.parlist) if si.startswith('tpkoff')])
		self.im0 = np.where(self.parlist == 'm0')[0]
		self.im1 = np.where(self.parlist == 'm1')[0]
		self.iCL = np.where(self.parlist == 'cl')[0]
		self.ispcrcl = np.array([i for i, si in enumerate(self.parlist) if si.startswith('specrecal')])
		self.imodelerr = np.array([i for i, si in enumerate(self.parlist) if si.startswith('modelerr')])
		self.imodelcorr = np.array([i for i, si in enumerate(self.parlist) if si.startswith('modelcorr')])
		self.iclscat = np.where(self.parlist=='clscat')[0]
		self.ispcrcl_norm,self.ispcrcl_poly = np.array([],dtype='int'),np.array([],dtype='int')
		for i,parname in enumerate(np.unique(self.parlist[self.ispcrcl])):
			self.ispcrcl_norm = np.append(self.ispcrcl_norm,np.where(self.parlist == parname)[0][-1])
			self.ispcrcl_poly = np.append(self.ispcrcl_poly,np.where(self.parlist == parname)[0][:-1])

		# set some phase/wavelength arrays
		self.phase = np.linspace(self.phaserange[0],self.phaserange[1],
								 int((self.phaserange[1]-self.phaserange[0])/self.phaseoutres)+1,True)

		self.interpMethod='nearest'
		
		nwaveout=int((self.waverange[1]-self.waverange[0])/self.waveoutres)
		self.wave = np.linspace(self.waverange[0],self.waverange[1],
								nwaveout+1,True)
		self.maxPhase=np.where(abs(self.phase) == np.min(abs(self.phase)))[0]
				
		self.neff=0

		# initialize the model
		self.components = self.SALTModel(guess)
		self.salterr = self.ErrModel(guess)
		
		self.m0guess = -19.49 #10**(-0.4*(-19.49-27.5))
		self.extrapolateDecline=0.015
		# set up the filters
		self.stdmag = {}
		self.fluxfactor = {}
		for survey in self.kcordict.keys():
			if survey == 'default': 
				self.stdmag[survey] = {}
				self.bbandoverlap = (self.wave>=self.kcordict['default']['Bwave'].min())&(self.wave<=self.kcordict['default']['Bwave'].max())
				self.bbandpbspl = np.interp(self.wave[self.bbandoverlap],self.kcordict['default']['Bwave'],self.kcordict['default']['Bwave'])
				self.bbandpbspl *= self.wave[self.bbandoverlap]
				self.bbandpbspl /= np.trapz(self.bbandpbspl,self.wave[self.bbandoverlap])*HC_ERG_AA
				self.stdmag[survey]['B']=synphot(
					self.kcordict[survey]['primarywave'],self.kcordict[survey]['AB'],
					filtwave=self.kcordict['default']['Bwave'],filttp=self.kcordict[survey]['Btp'],
					zpoff=0)
				self.kcordict['default']['minlam'] = np.min(self.kcordict['default']['Bwave'][self.kcordict['default']['Btp'] > 0.01])
				self.kcordict['default']['maxlam'] = np.max(self.kcordict['default']['Bwave'][self.kcordict['default']['Btp'] > 0.01])
				self.kcordict['default']['fluxfactor'] = 10**(0.4*(self.stdmag[survey]['B']+27.5))
				continue

			self.stdmag[survey] = {}
			self.fluxfactor[survey] = {}
			primarywave = self.kcordict[survey]['primarywave']
			for flt in self.kcordict[survey].keys():
				if flt == 'filtwave' or flt == 'primarywave' or flt == 'snflux' or flt == 'AB' or flt == 'BD17' or flt == 'Vega': continue
				if self.kcordict[survey][flt]['magsys'] == 'AB': primarykey = 'AB'
				elif self.kcordict[survey][flt]['magsys'].upper() == 'VEGA': primarykey = 'Vega'
				elif self.kcordict[survey][flt]['magsys'] == 'BD17': primarykey = 'BD17'
				self.stdmag[survey][flt] = synphot(
					primarywave,self.kcordict[survey][primarykey],
					filtwave=self.kcordict[survey]['filtwave'],
					filttp=self.kcordict[survey][flt]['filttrans'],
					zpoff=0) - self.kcordict[survey][flt]['primarymag']
				self.fluxfactor[survey][flt] = 10**(0.4*(self.stdmag[survey][flt]+27.5))
				self.kcordict[survey][flt]['minlam'] = np.min(self.kcordict[survey]['filtwave'][self.kcordict[survey][flt]['filttrans'] > 0.01])
				self.kcordict[survey][flt]['maxlam'] = np.max(self.kcordict[survey]['filtwave'][self.kcordict[survey][flt]['filttrans'] > 0.01])
				
		#Count number of photometric and spectroscopic points
		
		self.num_spec=0
		self.num_phot=0
		for sn in self.datadict.keys():
			photdata = self.datadict[sn]['photdata']
			specdata = self.datadict[sn]['specdata']
			survey = self.datadict[sn]['survey']
			filtwave = self.kcordict[survey]['filtwave']
			z = self.datadict[sn]['zHelio']
			self.num_spec += sum([specdata[key]['flux'].size for key in specdata])

			for flt in np.unique(photdata['filt']):
				self.num_phot+=(photdata['filt']==flt).sum()
			#While we're at it, calculate the extinction curve for the milky way
			self.datadict[sn]['mwextcurve']   = 10**(-0.4*extinction.fitzpatrick99(self.wave*(1+z),self.datadict[sn]['MWEBV']*3.1))
			self.datadict[sn]['mwextcurveint'] = interp1d(self.wave*(1+z),self.datadict[sn]['mwextcurve'] ,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)

		self.fixedUncertainties={}
		starttime=time.time()
		#Store derivatives of a spline with fixed knot locations with respect to each knot value
		self.spline_derivs = np.zeros([len(self.phase),len(self.wave),self.im0.size])
		for i in range(self.im0.size):
			self.spline_derivs[:,:,i]=bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder))
		nonzero=np.nonzero(self.spline_derivs)
		self.spline_deriv_interp= RegularGridInterpolator((self.phase,self.wave),self.spline_derivs,self.interpMethod,False,0)
		
		#Repeat for the error model parameters
		self.errorspline_deriv= np.zeros([len(self.phase),len(self.wave),self.imodelerr.size//self.n_components])
		for i in range(self.imodelerr.size//self.n_components):
			self.errorspline_deriv[:,:,i]=bisplev(self.phase, self.wave ,(self.errphaseknotloc,self.errwaveknotloc,np.arange(self.imodelerr.size//self.n_components)==i,self.bsorder,self.bsorder))
		self.errorspline_deriv_interp= RegularGridInterpolator((self.phase,self.wave),self.errorspline_deriv,self.interpMethod,False,0)
		
		#Store the lower and upper edges of the phase/wavelength basis functions
		self.phaseBins=self.phaseknotloc[:-(self.bsorder+1)],self.phaseknotloc[(self.bsorder+1):]
		self.waveBins=self.waveknotloc[:-(self.bsorder+1)],self.waveknotloc[(self.bsorder+1):]
		
		#Find the iqr of the phase/wavelength basis functions
		self.phaseRegularizationPoints=np.zeros(self.phaseBins[0].size*2)
		self.waveRegularizationPoints=np.zeros(self.waveBins[0].size*2)
		binRange=1/3,2/3
		phaseCumulative=(np.cumsum(self.spline_derivs[:,:,::self.waveBins[0].size],axis=0).sum(axis=1)/np.sum(self.spline_derivs[:,:,::self.waveBins[0].size],axis=(0,1)))
		self.phaseRegularizationPoints[::2]=self.phase[ np.abs(phaseCumulative-binRange[0]).argmin(axis=0)]
		self.phaseRegularizationPoints[1::2]=self.phase[ np.abs(phaseCumulative-binRange[1]).argmin(axis=0)]
		
		waveCumulative=(np.cumsum(self.spline_derivs[:,:,:self.waveBins[0].size],axis=1).sum(axis=0)/np.sum(self.spline_derivs[:,:,:self.waveBins[0].size],axis=(0,1)))
		self.waveRegularizationPoints[::2]=self.wave[ np.abs(waveCumulative-binRange[0]).argmin(axis=0)]
		self.waveRegularizationPoints[1::2]=self.wave[ np.abs(waveCumulative-binRange[1]).argmin(axis=0)]

		self.phaseBinCenters=np.array([(self.phase[:,np.newaxis]* self.spline_derivs[:,:,i*(self.waveBins[0].size)]).sum()/self.spline_derivs[:,:,i*(self.waveBins[0].size)].sum() for i in range(self.phaseBins[0].size) ])
		self.waveBinCenters=np.array([(self.wave[np.newaxis,:]* self.spline_derivs[:,:,i]).sum()/self.spline_derivs[:,:,i].sum() for i in range(self.waveBins[0].size)])

		#Find the basis functions evaluated at the centers of the basis functions for use in the regularization derivatives
		self.regularizationDerivs=[np.zeros((self.phaseRegularizationPoints.size,self.waveRegularizationPoints.size,self.im0.size)) for i in range(4)]
		for i in range(len(self.im0)):
			for j,derivs in enumerate([(0,0),(1,0),(0,1),(1,1)]):
				self.regularizationDerivs[j][:,:,i]=bisplev(self.phaseRegularizationPoints,self.waveRegularizationPoints,(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder),dx=derivs[0],dy=derivs[1])
		
		phase=self.phaseRegularizationPoints
		wave=self.waveRegularizationPoints
		fluxes=self.SALTModel(guess,evaluatePhase=self.phaseRegularizationPoints,evaluateWave=self.waveRegularizationPoints)
		self.guessScale=[np.sqrt(np.mean(f**2)) for f in fluxes]
		
		print('Time to calculate spline_derivs: %.2f'%(time.time()-starttime))
		
		
		self.getobswave()
		if self.regularize:
			self.updateEffectivePoints(guess)

		self.priors={ key: partial(__priors__[key],self) for key in __priors__}
		for prior in self.priors: 
			result=self.priors[prior](1,self.guess,self.SALTModel(self.guess))
			try:
				self.priors[prior].numResids=result[0].size
			except:
				self.priors[prior].numResids=1
		self.numPriorResids=sum([self.priors[x].numResids for x in self.priors])		
		self.__specFixedUncertainty__={}
		self.__photFixedUncertainty__={}
		
	def getobswave(self):
		"for each filter, setting up some things needed for synthetic photometry"
		
		for sn in self.datadict.keys():
			z = self.datadict[sn]['zHelio']
			survey = self.datadict[sn]['survey']
			filtwave = self.kcordict[survey]['filtwave']
			obswave=self.wave*(1+z)
			self.datadict[sn]['obswave'] = obswave
			
			self.datadict[sn]['obsphase'] = self.phase*(1+z)
			self.datadict[sn]['pbspl'] = {}
			self.datadict[sn]['denom'] = {}
			self.datadict[sn]['idx'] = {}
			self.datadict[sn]['lambdaeff']={}
			self.datadict[sn]['dwave'] = self.wave[1]*(1+z) - self.wave[0]*(1+z)
			for flt in np.unique(self.datadict[sn]['photdata']['filt']):

				filttrans = self.kcordict[survey][flt]['filttrans']

				g = (obswave>= self.kcordict[survey][flt]['minlam']) & (obswave<= self.kcordict[survey][flt]['maxlam'])	# overlap range
				
				self.datadict[sn]['idx'][flt] = g

				pbspl = np.interp(self.datadict[sn]['obswave'][g],filtwave,filttrans)
				pbspl *= self.datadict[sn]['obswave'][g]
				denom = np.trapz(pbspl,self.datadict[sn]['obswave'][g])
				pbspl /= denom*HC_ERG_AA

				self.datadict[sn]['pbspl'][flt] = pbspl[np.newaxis,:]
				self.datadict[sn]['denom'][flt] = denom
				self.datadict[sn]['lambdaeff'][flt] = self.kcordict[survey][flt]['lambdaeff']
		# rest-frame B
		filttrans = self.kcordict['default']['Btp']
		filtwave = self.kcordict['default']['Bwave']
			
		pbspl = np.interp(self.wave,filtwave,filttrans)
		pbspl *= self.wave
		denom = np.trapz(pbspl,self.wave)
		pbspl /= denom*HC_ERG_AA
		self.kcordict['default']['Bpbspl'] = pbspl
		self.kcordict['default']['Bdwave'] = self.wave[1] - self.wave[0]

	def maxlikefit(self,x,pool=None,debug=False,timeit=False,computeDerivatives=False,computePCDerivs=False,SpecErrScale=1.0):
		"""
		Calculates the likelihood of given SALT model to photometric and spectroscopic data given during initialization
		
		Parameters
		----------
		x : array
			SALT model parameters
			
		pool :	multiprocessing.pool.Pool, optional
			Optional worker pool to be used for calculating chi2 values for each SN. If not provided, all work is done in root process
		
		debug : boolean, optional
			Debug flag
		
		Returns
		-------
		
		chi2: float
			Goodness of fit of model to training data	
		"""
		
		#Set up SALT model
		# HACK
		components = self.SALTModel(x)
		if computeDerivatives:
			componentderivs = self.SALTModelDeriv(x,1,0,self.phase,self.wave)
		else:
			componentderivs = None
			
		salterr = self.ErrModel(x)
		saltcorr= self.CorrelationModel(x)
		if self.n_colorpars:
			colorLaw = SALT2ColorLaw(self.colorwaverange, x[self.parlist == 'cl'])
		else: colorLaw = None
		if self.n_colorscatpars:
			colorScat = True
		else: colorScat = None

		
		# timing stuff
		if timeit:
			self.tdelt0,self.tdelt1,self.tdelt2,self.tdelt3,self.tdelt4,self.tdelt5 = 0,0,0,0,0,0
		
		chi2 = 0
		#Construct arguments for maxlikeforSN method
		#If worker pool available, use it to calculate chi2 for each SN; otherwise, do it in this process
		args=[(None,sn,x,components,componentderivs,salterr,saltcorr,colorLaw,colorScat,debug,timeit,computeDerivatives,computePCDerivs,SpecErrScale) for sn in self.datadict.keys()]
		args2 = (x,components,componentderivs,salterr,saltcorr,colorLaw,colorScat,debug,timeit,computeDerivatives,computePCDerivs,SpecErrScale)
		mapFun=pool.map if pool else starmap
		if computeDerivatives:
			#result = list(pyParz.foreach(self.datadict.keys(),self.loglikeforSN,args2))
			result=list(mapFun(self.loglikeforSN,args))
			loglike=sum([r[0] for r in result])
			grad=sum([r[1] for r in result])
		else:
			#import pdb; pdb.set_trace()
			#result = list(pyParz.foreach(list(self.datadict.keys())[:4],self.loglikeforSN_multiprocess,args2))
			#import pdb; pdb.set_trace()
			loglike=sum(mapFun(self.loglikeforSN,args))

		logp = loglike
		if len(self.usePriors):
			priorResids,priorVals,priorJac=self.priorResids(self.usePriors,self.priorWidths,x)	
			logp -=(priorResids**2).sum()/2
			if computeDerivatives:
				grad-= (priorResids [:,np.newaxis] * priorJac).sum(axis=0)
		
		if self.regularize:
			for regularization, weight in [(self.phaseGradientRegularization, self.regulargradientphase),(self.waveGradientRegularization,self.regulargradientwave ),(self.dyadicRegularization,self.regulardyad)]:
				if weight ==0:
					continue
				regResids,regJac=regularization(x,components,computeDerivatives)
				logp-= sum([(res**2).sum()*weight/2 for res in regResids])
				if computeDerivatives:
					for idx,res,jac in zip([self.im0,self.im1],regResids,regJac):
						grad[idx] -= (res[:,np.newaxis]*jac ).sum(axis=0)
		self.nstep += 1
		print(logp.sum()*-2)

		if timeit:
			print('%.3f %.3f %.3f %.3f %.3f'%(self.tdelt0,self.tdelt1,self.tdelt2,self.tdelt3,self.tdelt4))
		if computeDerivatives:
			return logp,grad
		else:
			return logp
	
	#def getCholesky(self,photmodel)
	def getFixedUncertainties(self,x):		
		colorLaw = SALT2ColorLaw(self.colorwaverange, x[self.parlist == 'cl'])
		components=self.SALTModel(x)
		componentderivs=self.SALTModelDeriv(x,1,0,self.phase,self.wave)
		saltErr=self.ErrModel(x)
		saltCorr=self.CorrelationModel(x)

		fixedUncertainties={}
		for sn in self.datadict:

			photmodel,specmodel=self.modelvalsforSN(x,sn,components,componentderivs,colorLaw,saltErr,saltCorr,False,False)
			variance=photmodel['fluxvariance']+photmodel['modelvariance']
			Ls,colorvar=[],[]
			#Add color scatter
			for selectFilter,clscat,dclscat in photmodel['colorvariance']:
				flux=clscat*photmodel['modelflux'][selectFilter]
				#Find cholesky matrix as sqrt of diagonal uncertainties, then perform rank one update to incorporate color scatter
				L= np.diag(np.sqrt(variance[selectFilter]))
				for k in range(flux.size):
					r = np.sqrt(L[k,k]**2 + flux[k]**2)
					c = r/L[k,k]
					s = flux[k]/L[k,k]
					L[k,k] = r
					L[k+1:,k] = (L[k+1:,k] + s*flux[k+1:])/c
					flux[k+1:]= c*flux[k+1:] - s*L[k+1:,k]
				Ls+=[L]
				colorvar+=[(selectFilter,clscat,dclscat)]
			fixedUncertainties['phot_{}'.format(sn)]=Ls,colorvar
			fixedUncertainties['spec_{}'.format(sn)]=specmodel['fluxvariance'] + specmodel['modelvariance']
		return fixedUncertainties


				
	def ResidsForSN(self,x,sn,components,componentderivs,colorLaw,saltErr,saltCorr,
					computeDerivatives,computePCDerivs=False,fixedUncertainties=None,
					SpecErrScale=1.0,returnSpecModel=False):
		""" This method should be the only one required for any fitter to process the supernova data. 
		Find the residuals of a set of parameters to the photometric and spectroscopic data of a given supernova. 
		Photometric residuals are first decorrelated to diagonalize color scatter"""
		fixUncertainty= saltErr is None and saltCorr is None
		photmodel,specmodel=self.modelvalsforSN(x,sn,components,componentderivs,colorLaw,saltErr,saltCorr,computeDerivatives,computePCDerivs)

		#Separate code for photometry and spectra now, since photometry has to handle a covariance matrix with off-diagonal elements
		if fixUncertainty:
			Ls,colorvar=fixedUncertainties['phot_{}'.format(sn)]
			
		else:
			#Calculate cholesky matrix for each set of photometric measurements in each filter
			variance=photmodel['fluxvariance']+photmodel['modelvariance']
			Ls,colorvar=[],[]
			#Add color scatter
			for selectFilter,clscat,dclscat in photmodel['colorvariance']:
				flux=clscat*photmodel['modelflux'][selectFilter]
				#Find cholesky matrix as sqrt of diagonal uncertainties, then perform rank one update to incorporate color scatter
				L= np.diag(np.sqrt(variance[selectFilter]))
				for k in range(flux.size):
					r = np.sqrt(L[k,k]**2 + flux[k]**2)
					c = r/L[k,k]
					s = flux[k]/L[k,k]
					L[k,k] = r
					L[k+1:,k] = (L[k+1:,k] + s*flux[k+1:])/c
					flux[k+1:]= c*flux[k+1:] - s*L[k+1:,k]
				Ls+=[L]
				colorvar+=[(selectFilter,clscat,dclscat)]

		photresids={'resid':np.zeros(photmodel['modelflux'].size)}
		if computeDerivatives:
			photresids['resid_jacobian']=np.zeros((photmodel['modelflux'].size,self.npar))
		if not fixUncertainty: 
			photresids['lognorm']=0
			if computeDerivatives:
				photresids['lognorm_grad']=np.zeros(self.npar)
		
		for L,(selectFilter,clscat,dclscat) in zip(Ls,colorvar):
			#More stable to solve by backsubstitution than to invert and multiply
			fluxdiff=photmodel['modelflux'][selectFilter]-photmodel['dataflux'][selectFilter]
			photresids['resid'][selectFilter]=linalg.solve_triangular(L, fluxdiff,lower=True)
			if not fixUncertainty:
				photresids['lognorm']-= (np.log(np.diag(L)).sum())
		
			if computeDerivatives:
				photresids['resid_jacobian'][selectFilter]=linalg.solve_triangular(L,photmodel['modelflux_jacobian'][selectFilter],lower=True)
				if not fixUncertainty:
					#Cut out zeroed jacobian entries to save time
					nonzero=(~((photmodel['modelvariance_jacobian'][selectFilter]==0) & (photmodel['modelflux_jacobian'][selectFilter]==0)).all(axis=0)) | (self.parlist=='clscat')
					reducedJac=photmodel['modelvariance_jacobian'][selectFilter][:,nonzero]
					#Calculate L^-1 (necessary for the diagonal derivative)
					invL=linalg.solve_triangular(L,np.diag(np.ones(fluxdiff.size)),lower=True)
					# Calculate the fractional derivative of L w.r.t model parameters
					# L^-1 dL/dx = {L^-1 x d Sigma / dx x L^-T, with the upper triangular part zeroed and the diagonal halved}
					
					#First calculating diagonal part
					fractionalLDeriv=np.dot(invL,np.swapaxes(reducedJac[:,np.newaxis,:]* invL.T[:,:,np.newaxis],0,1))
					
					fluxPrime= linalg.solve_triangular(L,photmodel['modelflux'][selectFilter])
					jacPrime = linalg.solve_triangular(L,photmodel['modelflux_jacobian'][selectFilter][:,nonzero])
					#Derivative w.r.t  color scatter 
					fractionalLDeriv[:,:,self.parlist[nonzero]=='clscat']+= 2*clscat*dclscat[np.newaxis,np.newaxis,:]* np.outer(fluxPrime,fluxPrime)[:,:,np.newaxis]
					#Derivative w.r.t model flux
					dfluxSquared = jacPrime[:,np.newaxis,:]*fluxPrime[np.newaxis,:,np.newaxis]
					#Symmetrize
					dfluxSquared = np.swapaxes(dfluxSquared,0,1)+dfluxSquared
					fractionalLDeriv +=clscat**2 * dfluxSquared
					
					fractionalLDeriv[np.triu(np.ones((fluxdiff.size,fluxdiff.size),dtype=bool),1),:]=0
					fractionalLDeriv[np.diag(np.ones((fluxdiff.size),dtype=bool)),:]/=2
				
					#Multiply by size of transformed residuals and apply to resiudal jacobian
					
					photresids['resid_jacobian'][np.outer(selectFilter,nonzero)]-=np.dot(np.swapaxes(fractionalLDeriv,1,2),photresids['resid'][selectFilter]).flatten()
				
					#Trace of fractional derivative gives the gradient of the lognorm term, since determinant is product of diagonal
					photresids['lognorm_grad'][nonzero]-= np.trace(fractionalLDeriv)
				
		#Handle spectra
		
		if fixUncertainty:
			variance=fixedUncertainties['spec_{}'.format(sn)]
		else:
			variance=specmodel['fluxvariance'] + specmodel['modelvariance']
		uncertainty=np.sqrt(variance)*SpecErrScale
		# HACK
		#uncertainty *= 0.01
		#Suppress the effect of the spectra by multiplying chi^2 by number of photometric points over number of spectral points
		spectralSuppression=np.sqrt(self.num_phot/self.num_spec	)			
		
		specresids={'resid': spectralSuppression * (specmodel['modelflux']-specmodel['dataflux'])/uncertainty}
		specresids['uncertainty'] = uncertainty
		if not fixUncertainty:
			specresids['lognorm']=-np.log((np.sqrt(2*np.pi)*uncertainty)).sum()
		
		if computeDerivatives:
			specresids['resid_jacobian']=spectralSuppression * specmodel['modelflux_jacobian']/(uncertainty[:,np.newaxis])
			if not fixUncertainty:
				#Calculate jacobian of total spectral uncertainty including reported uncertainties
				uncertainty_jac=  specmodel['modelvariance_jacobian'] / (2*uncertainty[:,np.newaxis])
				specresids['lognorm_grad']= - (uncertainty_jac/uncertainty[:,np.newaxis]).sum(axis=0)
				specresids['resid_jacobian']-=   uncertainty_jac*(specresids['resid'] /uncertainty)[:,np.newaxis]

		if returnSpecModel: return photresids,specresids,specmodel
		else: return photresids,specresids
		
	def specValsForSN(self,x,sn,componentsModInterp,componentsDerivInterp,colorlaw,colorexp,computeDerivatives,computePCDerivs):
		z = self.datadict[sn]['zHelio']
		survey = self.datadict[sn]['survey']
		filtwave = self.kcordict[survey]['filtwave']
		obswave = self.datadict[sn]['obswave'] #self.wave*(1+z)
		obsphase = self.datadict[sn]['obsphase'] #self.phase*(1+z)
		specdata = self.datadict[sn]['specdata']
		pbspl = self.datadict[sn]['pbspl']
		dwave = self.datadict[sn]['dwave']
		idx = self.datadict[sn]['idx']
		x0,x1,c,tpkoff = x[self.parlist == 'x0_%s'%sn],x[self.parlist == 'x1_%s'%sn],\
						 x[self.parlist == 'c_%s'%sn],x[self.parlist == 'tpkoff_%s'%sn]

		if computeDerivatives:
			int1dM0,int1dM1=componentsModInterp
			int1dM0phasederiv,int1dM1phasederiv=componentsDerivInterp
		else:
			int1d=componentsModInterp

		nspecdata = sum([specdata[key]['flux'].size for key in specdata])
		specresultsdict={}
		specresultsdict['modelflux'] = np.zeros(nspecdata)
		specresultsdict['dataflux'] = np.zeros(nspecdata)
		if computeDerivatives:
			specresultsdict['modelflux_jacobian'] = np.zeros((nspecdata,self.npar))
			if computePCDerivs:
				self.__dict__['dmodelflux_dM0_spec_%s'%sn]=np.zeros([nspecdata,self.im0.size])
		iSpecStart = 0
		for k in specdata.keys():
			SpecLen = specdata[k]['flux'].size
			phase=specdata[k]['tobs']+tpkoff
			clippedPhase=np.clip(phase,obsphase.min(),obsphase.max())
			if computeDerivatives:
				M0interp = int1dM0(clippedPhase)
				M1interp = int1dM1(clippedPhase)
				M0phasederivinterp = int1dM0phasederiv(clippedPhase)
				M1phasederivinterp = int1dM1phasederiv(clippedPhase)
			else:
				modinterp = int1d(clippedPhase)
				
			#Check spectrum is inside proper phase range, extrapolate decline if necessary
			if phase < obsphase.min():
				pass
			elif phase > obsphase.max():
				pass
			#	saltfluxinterp*=10**(-0.4* self.extrapolateDecline* (phase-obsphase.max()))

			#Define recalibration factor
			coeffs=x[self.parlist=='specrecal_{}_{}'.format(sn,k)]
			pow=coeffs.size-1-np.arange(coeffs.size)
			coeffs/=factorial(pow)
			recalexp = np.exp(np.poly1d(coeffs)((specdata[k]['wavelength']-np.mean(specdata[k]['wavelength']))/self.specrange_wavescale_specrecal))

			if computeDerivatives:
				M0int = interp1d(obswave,M0interp[0],kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				M0interp = M0int(specdata[k]['wavelength'])*recalexp
				M1int = interp1d(obswave,M1interp[0],kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				M1interp = M1int(specdata[k]['wavelength'])*recalexp
				M0phasederivint = interp1d(obswave,M0phasederivinterp[0],kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				M0phasederivinterp = M0phasederivint(specdata[k]['wavelength'])*recalexp
				M1phasederivint = interp1d(obswave,M1phasederivinterp[0],kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				M1phasederivinterp = M1phasederivint(specdata[k]['wavelength'])*recalexp
				
				colorexpint = interp1d(obswave,colorexp,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				colorexpinterp = colorexpint(specdata[k]['wavelength'])
				colorlawint = interp1d(obswave,colorlaw,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				colorlawinterp = colorlawint(specdata[k]['wavelength'])

				modulatedFlux = x0*(M0interp + x1*M1interp)
				modulatedFluxDeriv = x0*(M0phasederivinterp + x1*M1phasederivinterp)
				specresultsdict['modelflux'][iSpecStart:iSpecStart+SpecLen] = modulatedFlux
			else:
				modint = interp1d(obswave,modinterp[0],kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				modinterp = modint(specdata[k]['wavelength'])*recalexp
				specresultsdict['modelflux'][iSpecStart:iSpecStart+SpecLen] = modinterp

			specresultsdict['dataflux'][iSpecStart:iSpecStart+SpecLen] = specdata[k]['flux']
			# derivatives....
			if computeDerivatives:
				
				intmult = _SCALE_FACTOR/(1+z)*x0*recalexp*colorexpinterp*self.datadict[sn]['mwextcurveint'](specdata[k]['wavelength'])
				intmultnox = _SCALE_FACTOR/(1+z)*recalexp*colorexpinterp*self.datadict[sn]['mwextcurveint'](specdata[k]['wavelength'])

				specresultsdict['modelflux_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist == 'c_{}'.format(sn))[0][0]] = modulatedFlux *np.log(10)*colorlawinterp
				specresultsdict['modelflux_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist == 'x0_{}'.format(sn))[0][0]] = (M0interp + x1*M1interp)
				specresultsdict['modelflux_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist == 'x1_{}'.format(sn))[0][0]] = x0*M1interp
				specresultsdict['modelflux_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist == 'tpkoff_{}'.format(sn))[0][0]] = modulatedFluxDeriv

				if self.specrecal :
					drecaltermdrecal=(((specdata[k]['wavelength']-np.mean(specdata[k]['wavelength']))/self.specrange_wavescale_specrecal)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
					specresultsdict['modelflux_jacobian'][iSpecStart:iSpecStart+SpecLen,self.parlist == 'specrecal_{}_{}'.format(sn,k)]  = modulatedFlux[:,np.newaxis] * drecaltermdrecal
					
				# color law
				for i in range(self.n_colorpars):
					dcolorlaw_dcli = interp1d(obswave,SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)(self.wave)-SALT2ColorLaw(self.colorwaverange, np.zeros(self.n_colorpars))(self.wave),kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)(specdata[k]['wavelength'])

					#dcolorlaw_dcli = SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)(specdata[k]['wavelength']/(1+z))-SALT2ColorLaw(self.colorwaverange, np.zeros(self.n_colorpars))(specdata[k]['wavelength']/(1+z))
					specresultsdict['modelflux_jacobian'][iSpecStart:iSpecStart+SpecLen,self.iCL[i]] = modulatedFlux*-0.4*np.log(10)*c*dcolorlaw_dcli
				
				# M0, M1
				if computePCDerivs:

					derivInterp = self.spline_deriv_interp((clippedPhase[0]/(1+z),specdata[k]['wavelength']/(1+z)),method=self.interpMethod)
					specresultsdict['modelflux_jacobian'][iSpecStart:iSpecStart+SpecLen,self.im0]  = derivInterp*intmult[:,np.newaxis]
					specresultsdict['modelflux_jacobian'][iSpecStart:iSpecStart+SpecLen,self.im1] =  derivInterp*intmult[:,np.newaxis]*x1
					self.__dict__['dmodelflux_dM0_spec_%s'%sn][iSpecStart:iSpecStart+SpecLen,:] = derivInterp*intmultnox[:,np.newaxis]
					
							
					if ( (phase>obsphase.max())).any():
						if phase > obsphase.max():
							#if computePCDerivs != 2:
							specresultsdict['modelflux_jacobian'][iSpecStart:iSpecStart+SpecLen,self.im0] *= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))
							specresultsdict['modelflux_jacobian'][iSpecStart:iSpecStart+SpecLen,self.im1] *= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))

							self.__dict__['dmodelflux_dM0_spec_%s'%sn][iSpecStart:iSpecStart+SpecLen,:] *= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))
							#if computePCDerivs != 1:

			iSpecStart += SpecLen
			
		if not computePCDerivs and computeDerivatives and 'dmodelflux_dM0_spec_%s'%sn in self.__dict__ :
			specresultsdict['modelflux_jacobian'][:,self.im0] = self.__dict__['dmodelflux_dM0_spec_%s'%sn]*x0
			specresultsdict['modelflux_jacobian'][:,self.im1] = self.__dict__['dmodelflux_dM0_spec_%s'%sn]*x0*x1

		return specresultsdict
		
	def specVarianceForSN(self,x,sn,interr1d,intcorr1d,colorlaw,colorexp,computeDerivatives):
		z = self.datadict[sn]['zHelio']
		survey = self.datadict[sn]['survey']
		filtwave = self.kcordict[survey]['filtwave']
		obswave = self.datadict[sn]['obswave'] #self.wave*(1+z)
		obsphase = self.datadict[sn]['obsphase'] #self.phase*(1+z)
		specdata = self.datadict[sn]['specdata']
		pbspl = self.datadict[sn]['pbspl']
		dwave = self.datadict[sn]['dwave']
		idx = self.datadict[sn]['idx']
		x0,x1,c,tpkoff = x[self.parlist == 'x0_%s'%sn],x[self.parlist == 'x1_%s'%sn],\
						 x[self.parlist == 'c_%s'%sn],x[self.parlist == 'tpkoff_%s'%sn]

		nspecdata = sum([specdata[key]['flux'].size for key in specdata])
		specresultsdict={}
		specresultsdict['fluxvariance'] =  np.zeros(nspecdata)
		specresultsdict['modelvariance'] =  np.zeros(nspecdata)
		if computeDerivatives:
			specresultsdict['modelvariance_jacobian']=np.zeros([nspecdata,self.npar])
		iSpecStart = 0
		for k in specdata.keys():
			SpecLen = specdata[k]['flux'].size
			phase=specdata[k]['tobs']+tpkoff
			clippedPhase=np.clip(phase,obsphase.min(),obsphase.max())

			#Define recalibration factor
			coeffs=x[self.parlist=='specrecal_{}_{}'.format(sn,k)]
			pow=coeffs.size-1-np.arange(coeffs.size)
			coeffs/=factorial(pow)
			recalexp = np.exp(np.poly1d(coeffs)((specdata[k]['wavelength']-np.mean(specdata[k]['wavelength']))/self.specrange_wavescale_specrecal))
			modelErrInt = [ interp1d( obswave, interr(clippedPhase)[0],kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True) for interr in interr1d]
			modelCorrInt= [ interp1d( obswave, intcorr(clippedPhase)[0],kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True) for intcorr in intcorr1d]
			corr=  [intcorr(specdata[k]['wavelength']) for intcorr in modelCorrInt]

			if computeDerivatives:
				colorexpint = interp1d(obswave,colorexp,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				colorexpinterp = colorexpint(specdata[k]['wavelength'])
				colorlawint = interp1d(obswave,colorlaw,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				colorlawinterp = colorlawint(specdata[k]['wavelength'])
					
				modelerrnox = [  interr( specdata[k]['wavelength']) *recalexp for interr in (modelErrInt)]
				modelUncertainty=  modelerrnox[0]**2  + 2*x1* corr[0]*modelerrnox[0]*modelerrnox[1] + x1**2 *modelerrnox[1]**2
			else:
				modelErrInt = [  interr( specdata[k]['wavelength'])  for interr in (modelErrInt)]
				modelUncertainty=recalexp**2 *  (modelErrInt[0]**2  + 2*x1* corr[0]*modelErrInt[0]*modelErrInt[1] + x1**2*modelErrInt[1]**2)
			
			specresultsdict['fluxvariance'][iSpecStart:iSpecStart+SpecLen] = specdata[k]['fluxerr']**2
			specresultsdict['modelvariance'][iSpecStart:iSpecStart+SpecLen] = x0**2 * modelUncertainty

			#
		
			# derivatives....
			if computeDerivatives:

				specresultsdict['modelvariance_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist == 'c_{}'.format(sn))[0][0]] = modelUncertainty * 2*x0**2 *np.log(10)*colorlawinterp
				specresultsdict['modelvariance_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist == 'x0_{}'.format(sn))[0][0]] = modelUncertainty*2*x0
				specresultsdict['modelvariance_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist == 'x1_{}'.format(sn))[0][0]] = x0**2 * 2*(modelerrnox[0]*modelerrnox[1]*corr[0]+ x1* modelerrnox[1]**2)

				if self.specrecal : 
					drecaltermdrecal=(((specdata[k]['wavelength']-np.mean(specdata[k]['wavelength']))/self.specrange_wavescale_specrecal)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
					specresultsdict['modelvariance_jacobian'][iSpecStart:iSpecStart+SpecLen,self.parlist == 'specrecal_{}_{}'.format(sn,k)]  = x0**2 * modelUncertainty[:,np.newaxis] * drecaltermdrecal * 2
			
				# color law
				for i in range(self.n_colorpars):
					dcolorlaw_dcli = interp1d(obswave,SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)(self.wave)-SALT2ColorLaw(self.colorwaverange, np.zeros(self.n_colorpars))(self.wave),kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)(specdata[k]['wavelength'])
					specresultsdict['modelvariance_jacobian'][iSpecStart:iSpecStart+SpecLen,self.iCL[i]] = x0**2 *2* (-0.4 *np.log(10)*c)*modelUncertainty*dcolorlaw_dcli
				
				interpresult=  self.errorspline_deriv_interp((clippedPhase[0]/(1+z),specdata[k]['wavelength']/(1+z)),method=self.interpMethod) 
				extinctionexp=(recalexp*colorexpinterp* _SCALE_FACTOR/(1+z)*self.datadict[sn]['mwextcurveint'](specdata[k]['wavelength']))
				specresultsdict['modelvariance_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist=='modelerr_0')[0]]   = 2* x0**2  * (extinctionexp *( modelerrnox[0] + corr[0]*modelerrnox[1]*x1))[:,np.newaxis] * interpresult
				specresultsdict['modelvariance_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist=='modelerr_1')[0]]   = 2* x0**2  * (extinctionexp *(modelerrnox[1]*x1**2 + corr[0]*modelerrnox[0]*x1))[:,np.newaxis] * interpresult
				specresultsdict['modelvariance_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist=='modelcorr_01')[0]] = 2* x0**2  * (modelerrnox[1]*modelerrnox[0]*x1)[:,np.newaxis]  * interpresult

			iSpecStart += SpecLen
		
		return specresultsdict

	def photVarianceForSN(self,x,sn,interr1d,intcorr1d,colorlaw,colorexp,computeDerivatives):
		"""Currently calculated only at the effective wavelength of the filter, not integrated over."""
		z = self.datadict[sn]['zHelio']
		survey = self.datadict[sn]['survey']
		filtwave = self.kcordict[survey]['filtwave']
		obswave = self.datadict[sn]['obswave'] #self.wave*(1+z)
		obsphase = self.datadict[sn]['obsphase'] #self.phase*(1+z)
		photdata = self.datadict[sn]['photdata']
		pbspl = self.datadict[sn]['pbspl']
		dwave = self.datadict[sn]['dwave']
		idx = self.datadict[sn]['idx']
		x0,x1,c,tpkoff = x[self.parlist == 'x0_%s'%sn],x[self.parlist == 'x1_%s'%sn],\
						 x[self.parlist == 'c_%s'%sn],x[self.parlist == 'tpkoff_%s'%sn]
		
		photresultsdict={}
		photresultsdict['fluxvariance'] =  np.zeros(len(photdata['filt']))
		photresultsdict['modelvariance'] =  np.zeros(len(photdata['filt']))
		photresultsdict['colorvariance'] = []
		if computeDerivatives:
			photresultsdict['modelvariance_jacobian']=np.zeros([photdata['filt'].size,self.npar])
		for flt in np.unique(photdata['filt']):
			lameff= self.datadict[sn]['lambdaeff'][flt]
			#Select data from the appropriate filter filter
			selectFilter=(photdata['filt']==flt)
			phase=photdata['tobs']+tpkoff
			phase=phase[selectFilter]
			clippedPhase=np.clip(phase,obsphase.min(),obsphase.max())
			nphase = len(phase)
			
			#Calculate color scatter

			if self.iclscat.size:
				coeffs=x[self.parlist=='clscat']
				pow=coeffs.size-1-np.arange(coeffs.size)
				coeffs/=factorial(pow)
				lameffPrime=lameff/(1+z)/1000
				colorscat=np.exp(np.poly1d(coeffs)(lameffPrime))
				
				if computeDerivatives:
					dcolorscatdx= colorscat*((lameffPrime) ** (pow) )/ factorial(pow)

				else :
					dcolorscatdx=0
			else:
				colorscat=0
				dcolorscatdx=np.array([])

			photresultsdict['colorvariance']+= [(selectFilter,colorscat,dcolorscatdx)]
			
			#Calculate model uncertainty
			modelErrInt = [ interp1d( obswave, interr(clippedPhase),axis=1,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True) for interr in interr1d]
			modelCorrInt= [ interp1d( obswave, intcorr(clippedPhase),axis=1,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True) for intcorr in intcorr1d]
			
			corr=  [intcorr(lameff) for intcorr in modelCorrInt]
			
			modelerrnox = [  interr( lameff) for interr in (modelErrInt)]
			modelUncertainty=  modelerrnox[0]**2  + 2*x1* corr[0]*modelerrnox[0]*modelerrnox[1] + x1**2 *modelerrnox[1]**2
			
			photresultsdict['fluxvariance'][selectFilter] = photdata['fluxcalerr'][selectFilter]**2
			fluxfactor=(self.fluxfactor[survey][flt]*(pbspl[flt].sum())*dwave)**2
			photresultsdict['modelvariance'][selectFilter]= x0**2 *fluxfactor  * modelUncertainty

		
			# derivatives....
			if computeDerivatives:

				colorexpint = interp1d(obswave,colorexp,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				colorexpinterp = colorexpint(lameff)
				colorlawint = interp1d(obswave,colorlaw,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				colorlawinterp = colorlawint(lameff)

				photresultsdict['modelvariance_jacobian'][selectFilter,np.where(self.parlist == 'c_{}'.format(sn))[0][0]] = fluxfactor * 2*x0**2 *np.log(10) * colorlawinterp* modelUncertainty 
				photresultsdict['modelvariance_jacobian'][selectFilter,np.where(self.parlist == 'x0_{}'.format(sn))[0][0]] = fluxfactor *2*x0* modelUncertainty
				photresultsdict['modelvariance_jacobian'][selectFilter,np.where(self.parlist == 'x1_{}'.format(sn))[0][0]] = x0**2 *fluxfactor* 2*(modelerrnox[0]*modelerrnox[1]*corr[0]+ x1* modelerrnox[1]**2)

				# color law
				for i in range(self.n_colorpars):
					dcolorlaw_dcli = interp1d(obswave,SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)(self.wave)-SALT2ColorLaw(self.colorwaverange, np.zeros(self.n_colorpars))(self.wave),kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)(lameff)
					photresultsdict['modelvariance_jacobian'][selectFilter,self.iCL[i]] =fluxfactor* x0**2 *2* (-0.4 *np.log(10)*c)*modelUncertainty*dcolorlaw_dcli

				interpresult=  self.errorspline_deriv_interp((clippedPhase/(1+z),lameff/(1+z)),method=self.interpMethod) 
				extinctionexp=(colorexpinterp* _SCALE_FACTOR/(1+z)*self.datadict[sn]['mwextcurveint'](lameff))

				photresultsdict['modelvariance_jacobian'][selectFilter[:,np.newaxis] & (self.parlist=='modelerr_0')[np.newaxis,:]]   = (fluxfactor* 2* x0**2  * extinctionexp * (( modelerrnox[0] + corr[0]*modelerrnox[1]*x1))[:,np.newaxis] * interpresult).flatten()
				photresultsdict['modelvariance_jacobian'][selectFilter[:,np.newaxis] & (self.parlist=='modelerr_1')[np.newaxis,:]]   = (fluxfactor* 2* x0**2  * extinctionexp  * ((modelerrnox[1]*x1**2 + corr[0]*modelerrnox[0]*x1))[:,np.newaxis] * interpresult).flatten()
				photresultsdict['modelvariance_jacobian'][selectFilter[:,np.newaxis] & (self.parlist=='modelcorr_01')[np.newaxis,:]] = (fluxfactor* 2* x0**2 * (modelerrnox[1]*modelerrnox[0]*x1)[:,np.newaxis]  * interpresult).flatten()
						
					
		return photresultsdict
		
	def photValsForSN(self,x,sn,componentsModInterp,componentsDerivInterp,colorlaw,colorexp,computeDerivatives,computePCDerivs):

		z = self.datadict[sn]['zHelio']
		survey = self.datadict[sn]['survey']
		filtwave = self.kcordict[survey]['filtwave']
		obswave = self.datadict[sn]['obswave'] #self.wave*(1+z)
		obsphase = self.datadict[sn]['obsphase'] #self.phase*(1+z)
		photdata = self.datadict[sn]['photdata']
		pbspl = self.datadict[sn]['pbspl']
		dwave = self.datadict[sn]['dwave']
		idx = self.datadict[sn]['idx']
		x0,x1,c,tpkoff = x[self.parlist == 'x0_%s'%sn],x[self.parlist == 'x1_%s'%sn],\
						 x[self.parlist == 'c_%s'%sn],x[self.parlist == 'tpkoff_%s'%sn]
		if computeDerivatives:
			int1dM0,int1dM1=componentsModInterp
			int1dM0phasederiv,int1dM1phasederiv=componentsDerivInterp
		else:
			int1d=componentsModInterp
		photresultsdict={}
		photresultsdict['modelflux'] = np.zeros(len(photdata['filt']))
		photresultsdict['dataflux'] = photdata['fluxcal']
		if computeDerivatives:
			photresultsdict['modelflux_jacobian'] = np.zeros((photdata['filt'].size,self.npar))
			photresultsdict['modelvariance_jacobian']=np.zeros([photdata['filt'].size,self.npar])
			if computePCDerivs:
				self.__dict__['dmodelflux_dM0_phot_%s'%sn] = np.zeros([photdata['filt'].size,len(self.im0)])#*1e-6 #+ 1e-5
		for flt in np.unique(photdata['filt']):
			#Select data from the appropriate filter filter
			selectFilter=(photdata['filt']==flt)
			phase=photdata['tobs']+tpkoff
			phase=phase[selectFilter]
			clippedPhase=np.clip(phase,obsphase.min(),obsphase.max())
			nphase = len(phase)
			
			#Array output indices match time along 0th axis, wavelength along 1st axis
			if computeDerivatives:
				M0interp = int1dM0(clippedPhase)
				M1interp = int1dM1(clippedPhase)
				M0phasederivinterp = int1dM0phasederiv(clippedPhase)
				M1phasederivinterp = int1dM1phasederiv(clippedPhase)

				modulatedM0= pbspl[flt]*M0interp[:,idx[flt]]
				modulatedM1=pbspl[flt]*M1interp[:,idx[flt]]
				modulatedphasederivM0= pbspl[flt]*M0phasederivinterp[:,idx[flt]]
				modulatedphasederivM1=pbspl[flt]*M1phasederivinterp[:,idx[flt]]

				if ( (phase>obsphase.max())).any():
					decayFactor=10**(-0.4*self.extrapolateDecline*(phase[phase>obsphase.max()]-obsphase.max()))
					modulatedM0[np.where(phase>obsphase.max())[0]] *= decayFactor
					modulatedM1[np.where(phase>obsphase.max())[0]] *= decayFactor
					modulatedphasederivM0[np.where(phase>obsphase.max())[0]] *= decayFactor
					modulatedphasederivM1[np.where(phase>obsphase.max())[0]] *= decayFactor
						
				modelsynM0flux=np.sum(modulatedM0, axis=1)*dwave*self.fluxfactor[survey][flt]
				modelsynM1flux=np.sum(modulatedM1, axis=1)*dwave*self.fluxfactor[survey][flt]
				modelsynphasederivM0flux=np.sum(modulatedphasederivM0, axis=1)*dwave*self.fluxfactor[survey][flt]
				modelsynphasederivM1flux=np.sum(modulatedphasederivM1, axis=1)*dwave*self.fluxfactor[survey][flt]
				
				modulatedFlux= x0*(modulatedM0 +modulatedM1*x1)
				modelflux = x0* (modelsynM0flux+ x1*modelsynM1flux)
				#Need to figure out how to handle derivatives wrt time when dealing with nearest neighbor interpolation; maybe require linear?
				for p in np.where(phase>obsphase.max())[0]:
				
					photresultsdict['modelflux_jacobian'][np.where(selectFilter)[0][p],self.parlist=='tpkoff_{}'.format(sn)]=-0.4*np.log(10)*self.extrapolateDecline*modelflux[p]
				
				photresultsdict['modelflux_jacobian'][selectFilter,self.parlist == 'x0_{}'.format(sn)] = modelsynM0flux+ x1*modelsynM1flux
				photresultsdict['modelflux_jacobian'][selectFilter,self.parlist == 'x1_{}'.format(sn)] = modelsynM1flux*x0
				photresultsdict['modelflux_jacobian'][selectFilter,self.parlist == 'tpkoff_{}'.format(sn)] = x0*(modelsynphasederivM0flux+ x1*modelsynphasederivM1flux)
				
				#d model / dc is total flux (M0 and M1 components (already modulated with passband)) times the color law and a factor of ln(10)
				photresultsdict['modelflux_jacobian'][selectFilter,self.parlist == 'c_{}'.format(sn)]=np.sum((modulatedFlux)*np.log(10)*colorlaw[np.newaxis,idx[flt]], axis=1)*dwave*self.fluxfactor[survey][flt]
				for i in range(self.n_colorpars):
					#Color law is linear wrt to the color law parameters; therefore derivative of the color law
					# with respect to color law parameter i is the color law with all other values zeroed minus the color law with all values zeroed
					dcolorlaw_dcli = SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)(self.wave[idx[flt]])-SALT2ColorLaw(self.colorwaverange, np.zeros(self.n_colorpars))(self.wave[idx[flt]])
					#Multiply M0 and M1 components (already modulated with passband) by c* d colorlaw / d cl_i, with associated normalizations
					photresultsdict['modelflux_jacobian'][selectFilter,self.iCL[i]] =  np.sum((modulatedFlux)*-0.4*np.log(10)*c*dcolorlaw_dcli[np.newaxis,:], axis=1)*dwave*self.fluxfactor[survey][flt]
				
				if computePCDerivs:
					passbandColorExp=pbspl[flt]*colorexp[idx[flt]]*self.datadict[sn]['mwextcurve'][idx[flt]]
					intmult = dwave*self.fluxfactor[survey][flt]*_SCALE_FACTOR/(1+z)*x0
					intmultnox = dwave*self.fluxfactor[survey][flt]*_SCALE_FACTOR/(1+z)
					for pdx,p in enumerate(np.where(selectFilter)[0]):
						derivInterp = self.spline_deriv_interp(
							(clippedPhase[pdx]/(1+z),self.wave[idx[flt]]),
							method=self.interpMethod)
					
						summation = np.sum( passbandColorExp.T * derivInterp, axis=0)
						photresultsdict['modelflux_jacobian'][p,self.im0]=  summation*intmult
						photresultsdict['modelflux_jacobian'][p,self.im1] = summation*intmult*x1
						self.__dict__['dmodelflux_dM0_phot_%s'%sn][p,:] = summation*intmultnox

			else:
				modinterp = int1d(clippedPhase)
				modelflux = np.sum(pbspl[flt]*modinterp[:,idx[flt]], axis=1)*dwave*self.fluxfactor[survey][flt]
				if ( (phase>obsphase.max())).any():
					modelflux[(phase>obsphase.max())]*= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))[(phase>obsphase.max())]
			photresultsdict['modelflux'][selectFilter]=modelflux
			
		phase=photdata['tobs']+tpkoff
		if computeDerivatives:

			if computePCDerivs:
				if ( (phase>obsphase.max())).any():
					for idx in np.where(phase>obsphase.max())[0]:
						
						decayFactor= 10**(-0.4*self.extrapolateDecline*(phase[idx]-obsphase.max()))
						photresultsdict['modelflux_jacobian'][idx,self.im0]*=decayFactor
						photresultsdict['modelflux_jacobian'][idx,self.im1]*=decayFactor
						self.__dict__['dmodelflux_dM0_phot_%s'%sn][idx,:] *= decayFactor
			elif 'dmodelflux_dM0_phot_%s'%sn  in self.__dict__:
					photresultsdict['modelflux_jacobian'][:,self.im0] = self.__dict__['dmodelflux_dM0_phot_%s'%sn]*x0
					photresultsdict['modelflux_jacobian'][:,self.im1]= self.__dict__['dmodelflux_dM0_phot_%s'%sn]*x0*x1
		return photresultsdict
	
	
	def modelvalsforSN(self,x,sn,components,componentderivs,colorLaw,saltErr,saltCorr,computeDerivatives,computePCDerivs):

		# model pars, initialization
		M0,M1 = copy.deepcopy(components)
		if componentderivs is not None:
			M0phasederiv,M1phasederiv = copy.deepcopy(componentderivs)
		
		z = self.datadict[sn]['zHelio']
		obsphase = self.datadict[sn]['obsphase'] #self.phase*(1+z)
		x0,x1,c,tpkoff = x[self.parlist == 'x0_%s'%sn],x[self.parlist == 'x1_%s'%sn],\
						 x[self.parlist == 'c_%s'%sn],x[self.parlist == 'tpkoff_%s'%sn]

		
		#Apply MW extinction
		M0 *= self.datadict[sn]['mwextcurve'][np.newaxis,:]
		M1 *= self.datadict[sn]['mwextcurve'][np.newaxis,:]
		if computeDerivatives:
			M0phasederiv *= self.datadict[sn]['mwextcurve'][np.newaxis,:]
			M1phasederiv *= self.datadict[sn]['mwextcurve'][np.newaxis,:]
		
		if colorLaw:
			colorlaw = -0.4 * colorLaw(self.wave)
			colorexp = 10. ** (colorlaw * c)
		else:
			colorexp=1

		if computeDerivatives:
			M0 *= colorexp; M1 *= colorexp
			M0phasederiv *= colorexp; M1phasederiv *= colorexp
		else: mod = x0*(M0 + x1*M1)*colorexp
				
		if computeDerivatives:
			M0 *= _SCALE_FACTOR/(1+z); M1 *= _SCALE_FACTOR/(1+z)
			int1dM0 = interp1d(obsphase,M0,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
			int1dM1 = interp1d(obsphase,M1,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
			M0phasederiv *= _SCALE_FACTOR/(1+z); M1phasederiv *= _SCALE_FACTOR/(1+z)
			int1dM0phasederiv = interp1d(obsphase,M0phasederiv,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
			int1dM1phasederiv = interp1d(obsphase,M1phasederiv,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
		else:
			mod *= _SCALE_FACTOR/(1+z)
			int1d = interp1d(obsphase,mod,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)

		if not ( saltErr is None and saltCorr is None):
			interr1d = [interp1d(obsphase,err * (self.datadict[sn]['mwextcurve'] *colorexp*  _SCALE_FACTOR/(1+z)) ,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True) for err in saltErr]
			intcorr1d= [interp1d(obsphase,corr ,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True) for corr in saltCorr ]
		returndicts=[]
		for valfun,uncertaintyfun,name in [(self.photValsForSN,self.photVarianceForSN,'phot'),(self.specValsForSN,self.specVarianceForSN, 'spec')]:
			valdict=valfun(x,sn,(int1dM0,int1dM1) if computeDerivatives else int1d,
                     (int1dM0phasederiv,int1dM1phasederiv) if computeDerivatives else None,
                     colorlaw,colorexp,computeDerivatives,computePCDerivs)

			if not ( saltErr is None and saltCorr is None): 
				uncertaintydict=uncertaintyfun(x,sn,interr1d,intcorr1d,colorlaw,colorexp,computeDerivatives)
				valdict.update(uncertaintydict)
			returndicts+=[valdict]

		return returndicts
		

	@prior
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
			
			residuals[idx:idx+priorFunction.numResids],values[idx:idx+priorFunction.numResids],jacobian[idx:idx+priorFunction.numResids,:]=priorFunction(width,x,components)
			idx+=priorFunction.numResids
		return residuals,values,jacobian

	def loglikeforSN_multiprocess(self,args):
		
		"""
		Calculates the likelihood of given SALT model to photometric and spectroscopic observations of a single SN 

		Parameters
		----------
		args : tuple or None
			Placeholder that contains all the other variables for multiprocessing quirks

		sn : str
			Name of supernova to compare to model
			
		x : array
			SALT model parameters
			
		components: array_like, optional
			SALT model components, if not provided will be derived from SALT model parameters passed in \'x\'
		
		colorLaw: function, optional
			SALT color law which takes wavelength as an argument

		debug : boolean, optional
			Debug flag
		
		Returns
		-------
		chi2: float
			Model chi2 relative to training data	
		"""

		#if timeit: tstart = time.time()

		sn,x,components,componentderivs,salterr,saltcorr,colorLaw,colorScat,debug,timeit,computeDerivatives,computePCDerivs,SpecErrScale = args[:]
		x = np.array(x)
		
		#Set up SALT model
		if components is None:
			components = self.SALTModel(x)
		if componentderivs is None and computeDerivatives:
			componentderivs = self.SALTModelDeriv(x,1,0,self.phase,self.wave)			
		if salterr is None:
			salterr = self.ErrModel(x)


		photResidsDict,specResidsDict = self.ResidsForSN(
			x,sn,components,componentderivs,colorLaw,salterr,
			saltcorr,computeDerivatives,computePCDerivs,
			SpecErrScale=SpecErrScale)

		
		loglike= - (photResidsDict['resid']**2).sum() / 2.   -(specResidsDict['resid']**2).sum()/2.+photResidsDict['lognorm']+specResidsDict['lognorm']
		if computeDerivatives: 
			grad_loglike=  - (photResidsDict['resid'][:,np.newaxis] * photResidsDict['resid_jacobian']).sum(axis=0) - (specResidsDict['resid'][:,np.newaxis] * specResidsDict['resid_jacobian']).sum(axis=0) + photResidsDict['lognorm_grad'] +specResidsDict['lognorm_grad']
			return loglike,grad_loglike
		else:
			return loglike

	
	def loglikeforSN(self,args,sn=None,x=None,components=None,componentderivs=None,salterr=None,saltcorr=None,
					 colorLaw=None,colorScat=None,
					 debug=False,timeit=False,computeDerivatives=False,computePCDerivs=False,SpecErrScale=1.0):
		
		"""
		Calculates the likelihood of given SALT model to photometric and spectroscopic observations of a single SN 

		Parameters
		----------
		args : tuple or None
			Placeholder that contains all the other variables for multiprocessing quirks

		sn : str
			Name of supernova to compare to model
			
		x : array
			SALT model parameters
			
		components: array_like, optional
			SALT model components, if not provided will be derived from SALT model parameters passed in \'x\'
		
		colorLaw: function, optional
			SALT color law which takes wavelength as an argument

		debug : boolean, optional
			Debug flag
		
		Returns
		-------
		chi2: float
			Model chi2 relative to training data	
		"""

		if timeit: tstart = time.time()

		if args: empty,sn,x,components,salterr,colorLaw,colorScat,debug = args[:]
		x = np.array(x)
		
		#Set up SALT model
		if components is None:
			components = self.SALTModel(x)
		if componentderivs is None and computeDerivatives:
			componentderivs = self.SALTModelDeriv(x,1,0,self.phase,self.wave)			
		if salterr is None:
			salterr = self.ErrModel(x)


		photResidsDict,specResidsDict = self.ResidsForSN(
			x,sn,components,componentderivs,colorLaw,salterr,
			saltcorr,computeDerivatives,computePCDerivs,
			SpecErrScale=SpecErrScale)

		
		loglike= - (photResidsDict['resid']**2).sum() / 2.   -(specResidsDict['resid']**2).sum()/2.+photResidsDict['lognorm']+specResidsDict['lognorm']
		if computeDerivatives: 
			grad_loglike=  - (photResidsDict['resid'][:,np.newaxis] * photResidsDict['resid_jacobian']).sum(axis=0) - (specResidsDict['resid'][:,np.newaxis] * specResidsDict['resid_jacobian']).sum(axis=0) + photResidsDict['lognorm_grad'] +specResidsDict['lognorm_grad']
			return loglike,grad_loglike
		else:
			return loglike
				
	def specchi2(self):

		return chi2
	
	def SALTModel(self,x,evaluatePhase=None,evaluateWave=None):
		"""Returns flux surfaces of SALT model"""
		try: m0pars = x[self.m0min:self.m0max]
		except: import pdb; pdb.set_trace()
		try:
			m0 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
						 self.wave if evaluateWave is None else evaluateWave,
						 (self.phaseknotloc,self.waveknotloc,m0pars,self.bsorder,self.bsorder))
		except:
			import pdb; pdb.set_trace()
			
		if self.n_components == 2:
			m1pars = x[self.parlist == 'm1']
			m1 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
						 self.wave if evaluateWave is None else evaluateWave,
						 (self.phaseknotloc,self.waveknotloc,m1pars,self.bsorder,self.bsorder))
			components = (m0,m1)
		elif self.n_components == 1:
			components = (m0,)
		else:
			raise RuntimeError('A maximum of two principal components is allowed')
			
		return components

	def SALTModelDeriv(self,x,dx,dy,evaluatePhase=None,evaluateWave=None):
		"""Returns derivatives of flux surfaces of SALT model"""
		try: m0pars = x[self.m0min:self.m0max]
		except: import pdb; pdb.set_trace()
		try:
			m0 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
						 self.wave if evaluateWave is None else evaluateWave,
						 (self.phaseknotloc,self.waveknotloc,m0pars,self.bsorder,self.bsorder),
						 dx=dx,dy=dy)
		except:
			import pdb; pdb.set_trace()
			
		if self.n_components == 2:
			m1pars = x[self.parlist == 'm1']
			m1 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
						 self.wave if evaluateWave is None else evaluateWave,
						 (self.phaseknotloc,self.waveknotloc,m1pars,self.bsorder,self.bsorder),
						 dx=dx,dy=dy)
			components = (m0,m1)
		elif self.n_components == 1:
			components = (m0,)
		else:
			raise RuntimeError('A maximum of two principal components is allowed')
			
		return components

	def CorrelationModel(self,x,evaluatePhase=None,evaluateWave=None):
		"""Returns correlation between SALT model components as a function of phase and wavelength"""
		components=[]
		for min,max in zip(self.corrmin,self.corrmax):
			try: errpars = x[min:max]
			except: import pdb; pdb.set_trace()

			components+=[  bisplev(self.phase if evaluatePhase is None else evaluatePhase,
							   self.wave if evaluateWave is None else evaluateWave,
							   (self.errphaseknotloc,self.errwaveknotloc,errpars,self.bsorder,self.bsorder))]
		return components

	
	def ErrModel(self,x,evaluatePhase=None,evaluateWave=None):
		"""Returns modeled variance of SALT model components as a function of phase and wavelength"""
		components=[]
		for min,max in zip(self.errmin,self.errmax):
			try: errpars = x[min:max]
			except: import pdb; pdb.set_trace()

			components+=[  bisplev(self.phase if evaluatePhase is None else evaluatePhase,
							   self.wave if evaluateWave is None else evaluateWave,
							   (self.errphaseknotloc,self.errwaveknotloc,errpars,self.bsorder,self.bsorder))]
		return components

	def getParsGN(self,x):

		m0pars = x[self.parlist == 'm0']
		#m0err = np.zeros(len(x[self.parlist == 'm0']))
		m1pars = x[self.parlist == 'm1']
		#m1err = np.zeros(len(x[self.parlist == 'm1']))
		
		# covmat (diagonals only?)
		m0_m1_cov = np.zeros(len(m0pars))

		modelerrpars = x[self.parlist == 'modelerr']
		#modelerrerr = np.zeros(len(x[self.parlist == 'modelerr']))

		clpars = x[self.parlist == 'cl']
		clerr = np.zeros(len(x[self.parlist == 'cl']))
		
		clscatpars = x[self.parlist == 'clscat']
		clscat=np.exp(np.poly1d(clscatpars)(self.wave/1000))

		#clscaterr = x[self.parlist == 'clscaterr']
		

		resultsdict = {}
		n_sn = len(self.datadict.keys())
		for k in self.datadict.keys():
			tpk_init = self.datadict[k]['photdata']['mjd'][0] - self.datadict[k]['photdata']['tobs'][0]
			resultsdict[k] = {'x0':x[self.parlist == 'x0_%s'%k],
							  'x1':x[self.parlist == 'x1_%s'%k],# - np.mean(x[self.ix1]),
							  'c':x[self.parlist == 'c_%s'%k],
							  'tpkoff':x[self.parlist == 'tpkoff_%s'%k],
							  'x0err':x[self.parlist == 'x0_%s'%k],
							  'x1err':x[self.parlist == 'x1_%s'%k],
							  'cerr':x[self.parlist == 'c_%s'%k],
							  'tpkofferr':x[self.parlist == 'tpkoff_%s'%k]}


		m0 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars,self.bsorder,self.bsorder))
		#m0errp = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars+m0err,self.bsorder,self.bsorder))
		#m0errm = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars-m0err,self.bsorder,self.bsorder))
		#m0err = (m0errp-m0errm)/2.
		if len(m1pars):
			m1 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars,self.bsorder,self.bsorder))
			#m1errp = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars+m1err,self.bsorder,self.bsorder))
			#m1errm = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars-m1err,self.bsorder,self.bsorder))
			#m1err = (m1errp-m1errm)/2.
		else:
			m1 = np.zeros(np.shape(m0))
			#m1err = np.zeros(np.shape(m0))

		#cov_m0_m1 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0_m1_cov,self.bsorder,self.bsorder))
		modelerr = bisplev(self.phase,self.wave,(self.errphaseknotloc,self.errwaveknotloc,modelerrpars,self.bsorder,self.bsorder))
		modelerr[:] = 0.0
		#clscat = splev(self.wave,(self.errwaveknotloc,clscatpars,3))
		if not len(clpars): clpars = []

		# model errors
		m0err,m1err = self.ErrModel(x)
		cov_m0_m1 = self.CorrelationModel(x)[0]*m0err*m1err
		
		return(x,self.phase,self.wave,m0,m0err,m1,m1err,cov_m0_m1,modelerr,
			   clpars,clerr,clscat,resultsdict)

	def getPars(self,loglikes,x,nburn=500,mkplots=False):

		axcount = 0; parcount = 0
		from matplotlib.backends.backend_pdf import PdfPages
		pdf_pages = PdfPages('%s/MCMC_hist.pdf'%self.outputdir)
		fig = plt.figure()
		
		m0pars = np.array([])
		m0err = np.array([])
		for i in self.im0:
			m0pars = np.append(m0pars,x[i,nburn:].mean())#/_SCALE_FACTOR)
			m0err = np.append(m0err,x[i,nburn:].std())#/_SCALE_FACTOR)
			if mkplots:
				if not parcount % 9:
					subnum = axcount%9+1
					ax = plt.subplot(3,3,subnum)
					axcount += 1
					md,std = np.mean(x[i,nburn:]),np.std(x[i,nburn:])
					histbins = np.linspace(md-3*std,md+3*std,50)
					ax.hist(x[i,nburn:],bins=histbins)
					ax.set_title('M0')
					if axcount % 9 == 8:
						pdf_pages.savefig(fig)
						fig = plt.figure()
				parcount += 1

		m1pars = np.array([])
		m1err = np.array([])
		parcount = 0
		for i in self.im1:
			m1pars = np.append(m1pars,x[i,nburn:].mean())
			m1err = np.append(m1err,x[i,nburn:].std())
			if mkplots:
				if not parcount % 9:
					subnum = axcount%9+1
					ax = plt.subplot(3,3,subnum)
					axcount += 1
					md,std = np.mean(x[i,nburn:]),np.std(x[i,nburn:])
					histbins = np.linspace(md-3*std,md+3*std,50)
					ax.hist(x[i,nburn:],bins=histbins)
					ax.set_title('M1')
					if axcount % 9 == 8:
						pdf_pages.savefig(fig)
						fig = plt.figure()
				parcount += 1

		# covmat (diagonals only?)
		m0_m1_cov = np.zeros(len(m0pars))
		chain_len = len(m0pars)
		m0mean = np.repeat(x[self.im0,nburn:].mean(axis=0),np.shape(x[self.im0,nburn:])[0]).reshape(np.shape(x[self.im0,nburn:]))
		m1mean = np.repeat(x[self.im1,nburn:].mean(axis=0),np.shape(x[self.im1,nburn:])[0]).reshape(np.shape(x[self.im1,nburn:]))
		m0var = x[self.im0,nburn:]-m0mean
		m1var = x[self.im1,nburn:]-m1mean
		for i in range(len(m0pars)):
			for j in range(len(m1pars)):
				if i == j: m0_m1_cov[i] = np.sum(m0var[j,:]*m1var[i,:])
		m0_m1_cov /= chain_len


		modelerrpars = np.array([])
		modelerrerr = np.array([])
		for i in np.where(self.parlist == 'modelerr')[0]:
			modelerrpars = np.append(modelerrpars,x[i,nburn:].mean())
			modelerrerr = np.append(modelerrerr,x[i,nburn:].std())

		clpars = np.array([])
		clerr = np.array([])
		for i in self.iCL:
			clpars = np.append(clpars,x[i,nburn:].mean())
			clerr = np.append(clpars,x[i,nburn:].std())

		clscatpars = np.array([])
		clscaterr = np.array([])
		for i in np.where(self.parlist == 'clscat')[0]:
			clscatpars = np.append(clpars,x[i,nburn:].mean())
			clscaterr = np.append(clpars,x[i,nburn:].std())



		result=np.mean(x[:,nburn:],axis=1)

		resultsdict = {}
		n_sn = len(self.datadict.keys())
		for k in self.datadict.keys():
			tpk_init = self.datadict[k]['photdata']['mjd'][0] - self.datadict[k]['photdata']['tobs'][0]
			resultsdict[k] = {'x0':x[self.parlist == 'x0_%s'%k,nburn:].mean(),
							  'x1':x[self.parlist == 'x1_%s'%k,nburn:].mean(),# - x[self.ix1,nburn:].mean(),
							  'c':x[self.parlist == 'c_%s'%k,nburn:].mean(),
							  'tpkoff':x[self.parlist == 'tpkoff_%s'%k,nburn:].mean(),
							  'x0err':x[self.parlist == 'x0_%s'%k,nburn:].std(),
							  'x1err':x[self.parlist == 'x1_%s'%k,nburn:].std(),
							  'cerr':x[self.parlist == 'c_%s'%k,nburn:].std(),
							  'tpkofferr':x[self.parlist == 'tpkoff_%s'%k,nburn:].std()}


		m0 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars,self.bsorder,self.bsorder))
		m0errp = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars+m0err,self.bsorder,self.bsorder))
		m0errm = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0pars-m0err,self.bsorder,self.bsorder))
		m0err = (m0errp-m0errm)/2.
		if len(m1pars):
			m1 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars,self.bsorder,self.bsorder))
			m1errp = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars+m1err,self.bsorder,self.bsorder))
			m1errm = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m1pars-m1err,self.bsorder,self.bsorder))
			m1err = (m1errp-m1errm)/2.
		else:
			m1 = np.zeros(np.shape(m0))
			m1err = np.zeros(np.shape(m0))

		cov_m0_m1 = bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,m0_m1_cov,self.bsorder,self.bsorder))
		modelerr = bisplev(self.phase,self.wave,(self.errphaseknotloc,self.errwaveknotloc,modelerrpars,self.bsorder,self.bsorder))
		#phaseout,waveout,fluxout = np.array([]),np.array([]),np.array([])
		#for i in range(len(self.phase)):
		#	for j in range(len(self.wave)):
		#		phaseout = np.append(phaseout,self.phase[i])
		#		waveout = np.append(waveout,self.wave[j])
		#		fluxout = np.append(fluxout,modelerr[i,j])
		#bspl = bisplrep(phaseout,waveout,fluxout,kx=3,ky=3,tx=self.errphaseknotloc,ty=self.errwaveknotloc,task=-1)
		#modelerr2 = bisplev(self.phase,self.wave,bspl)
		#plt.plot(self.wave,modelerr[self.phase == 0,:].flatten())
		#plt.plot(self.wave,modelerr2[self.phase == 0,:].flatten())
		clscat = splev(self.wave,(self.errwaveknotloc,clscatpars,3))
		if not len(clpars): clpars = []

		for snpar in ['x0','x1','c','tpkoff']:
			subnum = axcount%9+1
			ax = plt.subplot(3,3,subnum)
			axcount += 1
			md = np.mean(x[self.parlist == '%s_%s'%(snpar,k),nburn:])
			std = np.std(x[self.parlist == '%s_%s'%(snpar,k),nburn:])
			histbins = np.linspace(md-3*std,md+3*std,50)
			ax.hist(x[self.parlist == '%s_%s'%(snpar,k),nburn:],bins=histbins)
			ax.set_title('%s_%s'%(snpar,k))
			if axcount % 9 == 8:
				pdf_pages.savefig(fig)
				fig = plt.figure()


		pdf_pages.savefig(fig)			
		pdf_pages.close()
		
		return(result,self.phase,self.wave,m0,m0err,m1,m1err,cov_m0_m1,modelerr,
			   clpars,clerr,clscat,resultsdict)

			
	def updateEffectivePoints(self,x):
		"""
		Updates the "effective number of points" constraining a given bin in 
		phase/wavelength space. Should be called any time tpkoff values are recalculated
		
		Parameters
		----------
			
		x : array
			SALT model parameters
					
		"""
		#Clean out array
		self.neffRaw=np.zeros((self.phaseBinCenters.size,self.waveBinCenters.size))
		phaseIndices,waveIndices=np.unravel_index(np.arange(self.im0.size),(self.phaseRegularizationPoints.size,self.waveRegularizationPoints.size))
# 		start=time.time()
# 		spectime,phottime=0,0
		for sn in (self.datadict.keys()):
			tpkoff=x[self.parlist == 'tpkoff_%s'%sn]
			photdata = self.datadict[sn]['photdata']
			specdata = self.datadict[sn]['specdata']
			survey = self.datadict[sn]['survey']
			filtwave = self.kcordict[survey]['filtwave']
			z = self.datadict[sn]['zHelio']
			pbspl = self.datadict[sn]['pbspl']
			obswave=self.datadict[sn]['obswave']
			idx=self.datadict[sn]['idx']
			#For each spectrum, add one point to each bin for every spectral measurement in that bin
# 			spectime-=time.time()
			for k in specdata.keys():
				restWave=specdata[k]['wavelength']/(1+z)
				restWave=restWave[(restWave>self.waveBins[0][0])&(restWave<self.waveBins[1][-1])]
				phase=(specdata[k]['tobs']+tpkoff)/(1+z)
				phaseAffected= (phase>self.phaseBins[0]) & (phase<self.phaseBins[1])
				if phase <= self.phaseBins[0][0]:
					phaseAffected[0]=True
				elif phase>= self.phaseBins[1][-1]:
					phaseAffected[-1]=True
					
				waveAffected= (restWave.min()<self.waveBins[1])&(restWave.max()>self.waveBins[0])
				basisAffected=(phaseAffected[:,np.newaxis] & waveAffected[np.newaxis,:]).flatten()
				
				result=self.spline_deriv_interp((phase, restWave),method=self.interpMethod )[:,basisAffected].sum(axis=0)
				if phase>=self.phaseBins[1][-1]: result*=10**(-0.4*self.extrapolateDecline*((1+z)*(phase-self.phaseBins[1][-1])))
				self.neffRaw[np.where(basisAffected.reshape(self.neffRaw.shape))]+=result
# 			spectime+=time.time()
# 			phottime-=time.time()
			#For each photometric filter, weight the contribution by  
			for flt in np.unique(photdata['filt']):
				selectFilter=(photdata['filt']==flt)
				phase=(photdata['tobs'][selectFilter]+tpkoff)/(1+z)
				
				waveAffected= (self.waveBins[1] > (self.kcordict[survey][flt]['minlam']/(1+z))) & (self.waveBins[0] < (self.kcordict[survey][flt]['maxlam']/(1+z)))
				phaseAffected= ((phase[:,np.newaxis]>self.phaseBins[0][np.newaxis,:])& (phase[:,np.newaxis]<self.phaseBins[1][np.newaxis,:]))
				phaseAffected[:,0]=phaseAffected[:,0] | (phase<=self.phaseBins[0][0])
				phaseAffected[:,-1]=phaseAffected[:,-1] | (phase>=self.phaseBins[1][-1])
				basisAffected=(phaseAffected[:,:,np.newaxis] & waveAffected[np.newaxis,np.newaxis,:]).reshape((phase.size,self.im0.size))
				for pdx,p in enumerate(np.where(selectFilter)[0]):
					
					derivInterp = self.spline_deriv_interp((phase[pdx],self.wave[idx[flt]]),method=self.interpMethod)[:,basisAffected[pdx]]
					
					summation = np.sum( pbspl[flt].reshape((pbspl[flt].size,1)) * derivInterp, axis=0)/ np.sum(pbspl[flt])
					if phase[pdx]>=self.phaseBins[1][-1]: summation*=10**(-0.4*self.extrapolateDecline*((1+z)*(phase[pdx]-self.phaseBins[1][-1])))
					self.neffRaw[np.where(basisAffected[pdx].reshape(self.neffRaw.shape))]+=summation
# 			phottime+=time.time()
# 		print('Time for total neff is ',time.time()-start)
# 		print('Spectime: ',spectime,'Phottime: ',phottime)
		#import pdb; pdb.set_trace()
		#Smear it out a bit along phase axis
		#self.neff=gaussian_filter1d(self.neff,1,0)
		self.neffRaw=interp2d(self.waveBinCenters,self.phaseBinCenters,self.neffRaw)(self.waveRegularizationPoints,self.phaseRegularizationPoints)		
		
		self.neff=np.clip(self.neffRaw,5e-10,5)
		# hack!
# 		self.plotEffectivePoints([-12.5,0,12.5,40],'neff.png')
# 		self.plotEffectivePoints(None,'neff-heatmap.png')

	def plotEffectivePoints(self,phases=None,output=None):

		import matplotlib.pyplot as plt
		if phases is None:
			plt.imshow(self.neff,cmap='Greys',aspect='auto')
			xticks=np.linspace(0,self.waveBins[0].size,8,False)
			plt.xticks(xticks,['{:.0f}'.format(self.waveBins[int(x)]) for x in xticks])
			plt.xlabel('$\lambda$ / Angstrom')
			yticks=np.linspace(0,self.phaseBins[0].size,8,False)
			plt.yticks(yticks,['{:.0f}'.format(self.phaseBins[int(x)]) for x in yticks])
			plt.ylabel('Phase / days')
		else:
			inds=np.searchsorted(self.phaseRegularizationPoints,phases)
			# hack!
			for i in inds:
				plt.plot(self.waveBins[:-1],self.neff[i,:],label='{:.1f} days'.format(self.phaseRegularizationPoints[i]))
			plt.ylabel('$N_eff$')
			plt.xlabel('$\lambda (\AA)$')
			plt.xlim(self.phaseRegularizationPoints.min(),self.phaseRegularizationPoints.max())
			plt.legend()
		
		if output is None:
			plt.show()
		else:
			plt.savefig(output,dpi=288)
		plt.clf()
	
	def regularizationScale(self,components,fluxes):
		if self.regularizationScaleMethod=='fixed':
			return self.guessScale,[np.zeros(self.im0.size) for component in fluxes]
		elif self.regularizationScaleMethod=='bbandmax':
			maxFlux=[interp1d(self.phase,flux[:,self.bbandoverlap],axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)(0) for flux in components]
			maxB=[np.sum(self.bbandpbspl*mf) for mf in maxFlux]
			derivInterp = self.spline_deriv_interp(
				(0,self.wave[self.bbandoverlap]),
				method=self.interpMethod)
			summation=(derivInterp*self.bbandpbspl[:,np.newaxis]).sum(axis=0)
			return [np.abs(mB) for mB in maxB],[summation*np.sign(mB) for mB in maxB]
		elif self.regularizationScaleMethod == 'rms':
			scale= [np.sqrt(np.mean(flux**2)) for flux in fluxes]
			return scale, [np.mean(flux[:,:,np.newaxis]*self.regularizationDerivs[0],axis=(0,1))/s for s,flux in zip(scale,fluxes)]
		elif self.regularizationScaleMethod=='meanabs':
			scale= [np.mean(np.abs(flux)) for flux in fluxes]
			return scale, [np.mean(np.sign(flux[:,:,np.newaxis])*self.regularizationDerivs[0],axis=(0,1)) for s,flux in zip(scale,fluxes)]
		elif self.regularizationScaleMethod=='mad':
			scale= [np.mean(np.abs(np.median(flux))) for flux in zip(fluxes)]
			return scale, [np.zeros(self.im0.size) for component in fluxes]
		elif self.regularizationScaleMethod=='maxneff':
			iqr=[ (np.percentile(self.neffRaw,80)<self.neffRaw)  for flux in fluxes]
			scale= [np.mean(np.abs(flux[select])) for select,flux in zip(iqr,fluxes)]
			return scale, [np.mean(np.sign(flux[select,np.newaxis])*self.regularizationDerivs[0][select,:],axis=(0)) for s,select,flux in zip(scale,iqr,fluxes)]
		elif self.regularizationScaleMethod=='mid5abs':
			iqr=[ (np.percentile(flux,45)<flux) & (np.percentile(flux,55)>flux)  for flux in fluxes]
			scale= [np.mean(np.abs(flux[select])) for select,flux in zip(iqr,fluxes)]
			return scale, [np.mean(np.sign(flux[select,np.newaxis])*self.regularizationDerivs[0][select,:],axis=(0)) for s,select,flux in zip(scale,iqr,fluxes)]
		elif self.regularizationScaleMethod=='midqtabs':
			iqr=[ (np.percentile(flux,25)<flux) & (np.percentile(flux,75)>flux)  for flux in fluxes]
			scale= [np.mean(np.abs(flux[select])) for select,flux in zip(iqr,fluxes)]
			return scale, [np.mean(np.sign(flux[select,np.newaxis])*self.regularizationDerivs[0][select,:],axis=(0)) for s,select,flux in zip(scale,iqr,fluxes)]
		elif self.regularizationScaleMethod=='midqt':
			iqr=[ (np.percentile(flux,25)<flux) & (np.percentile(flux,75)>flux)  for flux in fluxes]
			scale= [np.sqrt(np.mean(flux[select]**2)) for select,flux in zip(iqr,fluxes)]
			return scale, [np.mean(flux[select,np.newaxis]*self.regularizationDerivs[0][select,:],axis=(0))/s for s,select,flux in zip(scale,iqr,fluxes)]
		else:
			raise ValueError('Regularization scale method invalid: ',self.regularizationScaleMethod)

	def dyadicRegularization(self,x, components,computeJac=True):
		phase=self.phaseRegularizationPoints
		wave=self.waveRegularizationPoints
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdwave=self.SALTModelDeriv(x,0,1,phase,wave)
		dfluxdphase=self.SALTModelDeriv(x,1,0,phase,wave)
		d2fluxdphasedwave=self.SALTModelDeriv(x,1,1,phase,wave)
		scale,scaleDeriv=self.regularizationScale(components,fluxes)
		resids=[]
		jac=[]
		for i in range(len(fluxes)):
			#Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
			normalization=np.sqrt(1/( (self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))
			#0 if model is locally separable in phase and wavelength i.e. flux=g(phase)* h(wavelength) for arbitrary functions g and h
			numerator=(dfluxdphase[i] *dfluxdwave[i] -d2fluxdphasedwave[i] *fluxes[i] )
			dnumerator=( self.regularizationDerivs[1]*dfluxdwave[i][:,:,np.newaxis] + self.regularizationDerivs[2]* dfluxdphase[i][:,:,np.newaxis] - self.regularizationDerivs[3]* fluxes[i][:,:,np.newaxis] - self.regularizationDerivs[0]* d2fluxdphasedwave[i][:,:,np.newaxis] )			
			resids += [normalization* (numerator / (scale[i]**2 * np.sqrt( self.neff ))).flatten()]
			if computeJac: jac += [((dnumerator*(scale[i]**2 )- scaleDeriv[i][np.newaxis,np.newaxis,:]*2*scale[i]*numerator[:,:,np.newaxis])/np.sqrt(self.neff)[:,:,np.newaxis]*normalization / scale[i]**4  ).reshape(-1, self.im0.size)]
			else: jac+=[None]
		return resids,jac 
	
	def phaseGradientRegularization(self, x, components, computeJac=True):
		phase=self.phaseRegularizationPoints
		wave=self.waveRegularizationPoints
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdphase=self.SALTModelDeriv(x,1,0,phase,wave)
		scale,scaleDeriv=self.regularizationScale(components,fluxes)
		resids=[]
		jac=[]
		for i in range(len(fluxes)):
			#Normalize gradient by flux scale
			normedGrad=dfluxdphase[i]/scale[i]
			#Derivative of normalized gradient with respect to model parameters
			normedGradDerivs=(self.regularizationDerivs[1] * scale[i] - scaleDeriv[i][np.newaxis,np.newaxis,:]*dfluxdphase[i][:,:,np.newaxis])/ scale[i]**2
			#Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
			normalization=np.sqrt(1/((self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))
			#Minimize model derivative w.r.t wavelength in unconstrained regions
			resids+= [normalization* ( normedGrad /	np.sqrt( self.neff )).flatten()]
			if computeJac: jac+= [normalization*((normedGradDerivs) / np.sqrt( self.neff )[:,:,np.newaxis]).reshape(-1, self.im0.size)]
			else: jac+=[None]
		return resids,jac  
	
	def waveGradientRegularization(self, x, components,computeJac=True):
		phase=self.phaseRegularizationPoints
		wave=self.waveRegularizationPoints
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdwave=self.SALTModelDeriv(x,0,1,phase,wave)
		scale,scaleDeriv=self.regularizationScale(components,fluxes)
		waveGradResids=[]
		jac=[]
		for i in range(len(fluxes)):
			#Normalize gradient by flux scale
			normedGrad=dfluxdwave[i]/scale[i]
			#Derivative of normalized gradient with respect to model parameters
			normedGradDerivs=(self.regularizationDerivs[2] * scale[i] - scaleDeriv[i][np.newaxis,np.newaxis,:]*dfluxdwave[i][:,:,np.newaxis])/ scale[i]**2
			#Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
			normalization=np.sqrt(1/((self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))
			#Minimize model derivative w.r.t wavelength in unconstrained regions
			waveGradResids+= [normalization* ( normedGrad /	np.sqrt( self.neff )).flatten()]
			if computeJac: jac+= [normalization*((normedGradDerivs) / np.sqrt( self.neff )[:,:,np.newaxis]).reshape(-1, self.im0.size)]
			else: jac+=[None]
		return waveGradResids,jac 
		
def trapIntegrate(a,b,xs,ys):
	if (a<xs.min()) or (b>xs.max()):
		raise ValueError('Bounds of integration outside of provided values')
	aInd,bInd=np.searchsorted(xs,[a,b])
	if aInd==bInd:
		return ((ys[aInd]-ys[aInd-1])/(xs[aInd]-xs[aInd-1])*((a+b)/2-xs[aInd-1])+ys[aInd-1])*(b-a)
	elif aInd+1==bInd:
		return ((ys[aInd]-ys[aInd-1])/(xs[aInd]-xs[aInd-1])*((a+xs[aInd])/2-xs[aInd-1])+ys[aInd-1])*(xs[aInd]-a) + ((ys[bInd]-ys[bInd-1])/(xs[bInd]-xs[bInd-1])*((xs[bInd-1]+b)/2-xs[bInd-1])+ys[bInd-1])*(b-xs[bInd-1])
	else:
		return np.trapz(ys[(xs>=a)&(xs<b)],xs[(xs>=a)&(xs<b)])+((ys[aInd]-ys[aInd-1])/(xs[aInd]-xs[aInd-1])*((a+xs[aInd])/2-xs[aInd-1])+ys[aInd-1])*(xs[aInd]-a) + ((ys[bInd]-ys[bInd-1])/(xs[bInd]-xs[bInd-1])*((xs[bInd-1]+b)/2-xs[bInd-1])+ys[bInd-1])*(b-xs[bInd-1])
