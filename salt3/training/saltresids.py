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
from scipy.linalg import cholesky

import numpy as np
from numpy.random import standard_normal
from numpy.linalg import inv,pinv

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
		self.errmin = tuple([np.min(np.where(self.parlist == 'modelerr_{}'.format(i))[0]) for i in range(3)]) 
		self.errmax = tuple([np.max(np.where(self.parlist == 'modelerr_{}'.format(i))[0]) for i in range(3)]) 
		self.ix1 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x1')])
		self.ix0 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x0')])
		self.ic	 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('c_')])
		self.itpk = np.array([i for i, si in enumerate(self.parlist) if si.startswith('tpkoff')])
		self.im0 = np.where(self.parlist == 'm0')[0]
		self.im1 = np.where(self.parlist == 'm1')[0]
		self.iCL = np.where(self.parlist == 'cl')[0]
		self.ispcrcl = np.array([i for i, si in enumerate(self.parlist) if si.startswith('specrecal')])
		self.imodelerr = np.where((self.parlist=='modelerr_0') | (self.parlist=='modelerr_1') | (self.parlist=='modelerr_2') )[0]
		
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


		starttime=time.time()
		#Store derivatives of a spline with fixed knot locations with respect to each knot value
		self.spline_derivs = np.zeros([len(self.phase),len(self.wave),self.im0.size])
		for i in range(self.im0.size):
			self.spline_derivs[:,:,i]=bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder))
		nonzero=np.nonzero(self.spline_derivs)
		self.spline_deriv_interp= RegularGridInterpolator((self.phase,self.wave),self.spline_derivs,self.interpMethod,False,0)
		
		#Repeat for the error model parameters
		self.errorspline_deriv= np.zeros([len(self.phase),len(self.wave),self.imodelerr.size//3])
		for i in range(self.imodelerr.size//3):
			self.errorspline_deriv[:,:,i]=bisplev(self.phase, self.wave ,(self.errphaseknotloc,self.errwaveknotloc,np.arange(self.imodelerr.size//3)==i,self.bsorder,self.bsorder))
		self.errorspline_deriv_interp= RegularGridInterpolator((self.phase,self.wave),self.errorspline_deriv,self.interpMethod,False,0)
		
		#Store the lower and upper edges of the phase/wavelength basis functions
		self.phaseBins=self.phaseknotloc[:-(self.bsorder+1)],self.phaseknotloc[(self.bsorder+1):]
		self.waveBins=self.waveknotloc[:-(self.bsorder+1)],self.waveknotloc[(self.bsorder+1):]
		
		#Find the weighted centers of the phase/wavelength basis functions
		self.phaseBinCenters=np.array([(self.phase[:,np.newaxis]* self.spline_derivs[:,:,i*(self.waveBins[0].size)]).sum()/self.spline_derivs[:,:,i*(self.waveBins[0].size)].sum() for i in range(self.phaseBins[0].size) ])
		self.waveBinCenters=np.array([(self.wave[np.newaxis,:]* self.spline_derivs[:,:,i]).sum()/self.spline_derivs[:,:,i].sum() for i in range(self.waveBins[0].size)])

		#Find the basis functions evaluated at the centers of the basis functions for use in the regularization derivatives
		self.regularizationDerivs=[np.zeros((self.phaseBinCenters.size,self.waveBinCenters.size,self.im0.size)) for i in range(4)]
		for i in range(len(self.im0)):
			for j,derivs in enumerate([(0,0),(1,0),(0,1),(1,1)]):
				self.regularizationDerivs[j][:,:,i]=bisplev(self.phaseBinCenters,self.waveBinCenters,(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder),dx=derivs[0],dy=derivs[1])
			
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

		# rest-frame B
		filttrans = self.kcordict['default']['Btp']
		filtwave = self.kcordict['default']['Bwave']
			
		pbspl = np.interp(self.wave,filtwave,filttrans)
		pbspl *= self.wave
		denom = np.trapz(pbspl,self.wave)
		pbspl /= denom*HC_ERG_AA
		self.kcordict['default']['Bpbspl'] = pbspl
		self.kcordict['default']['Bdwave'] = self.wave[1] - self.wave[0]
				
	def maxlikefit(self,x,pool=None,debug=False,timeit=False,computeDerivatives=False,computePCDerivs=False):
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
		
		salterr = self.ErrModel(x)
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
		args=[(None,sn,x,components,salterr,colorLaw,colorScat,debug,timeit,computeDerivatives,computePCDerivs) for sn in self.datadict.keys()]
		mapFun=pool.map if pool else starmap
		if computeDerivatives:
			result=list(mapFun(self.loglikeforSN,args))
			loglike=sum([r[0] for r in result])
			grad=sum([r[1] for r in result])
		else:
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
				regResids,regJac=regularization(x,computeDerivatives)
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
				
	def ResidsForSN(self,x,sn,components,colorLaw,saltErr,computeDerivatives,computePCDerivs=False,fixUncertainty=True):
		
		modeldicts=self.modelvalsforSN(x,sn,components,colorLaw,saltErr,computeDerivatives,computePCDerivs,fixUncertainty)
		
		residslist=[]
		for modeldict,name in zip(modeldicts,['phot','spec']):
			uncertainty=np.hypot(modeldict['fluxuncertainty'],modeldict['modeluncertainty'])
			#Suppress the effect of the spectra by multiplying chi^2 by number of photometric points over number of spectral points
			if name =='spec': spectralSuppression=np.sqrt(self.num_phot/self.num_spec)
			else: spectralSuppression=1
			#If fixing the error, retrieve from storage
				
			
			residsdict={'resid': spectralSuppression * (modeldict['modelflux']-modeldict['dataflux'])/uncertainty}
			if not fixUncertainty:
				residsdict['lognorm']=-np.log((np.sqrt(2*np.pi)*uncertainty)).sum()
			
			if computeDerivatives:
				residsdict['resid_jacobian']=spectralSuppression * modeldict['modelflux_jacobian']/(uncertainty[:,np.newaxis])
				if not fixUncertainty:
					uncertainty_jac=  modeldict['modeluncertainty_jacobian'] *(modeldict['modeluncertainty'] / uncertainty)[:,np.newaxis] 
					residsdict['lognorm_grad']= - (uncertainty_jac/uncertainty[:,np.newaxis]).sum(axis=0)
					residsdict['resid_jacobian']-=   uncertainty_jac*(residsdict['resid'] /uncertainty)[:,np.newaxis]

			residslist+=[residsdict]
		return tuple(residslist)
			
	def specValsForSN(self,x,sn,componentsModInterp,colorlaw,colorexp,computeDerivatives,computePCDerivs):
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
			else:
				modinterp = int1d(clippedPhase)
				
			#Check spectrum is inside proper phase range, extrapolate decline if necessary
			if phase < obsphase.min():
				pass
			elif phase > obsphase.max():
				saltfluxinterp*=10**(-0.4* self.extrapolateDecline* (phase-obsphase.max()))

			#Define recalibration factor
			coeffs=x[self.parlist=='specrecal_{}_{}'.format(sn,k)]
			coeffs/=factorial(np.arange(len(coeffs)))
			recalexp = np.exp(np.poly1d(coeffs)((specdata[k]['wavelength']-np.mean(specdata[k]['wavelength']))/self.specrange_wavescale_specrecal))

			if computeDerivatives:
				M0int = interp1d(obswave,M0interp[0],kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				M0interp = M0int(specdata[k]['wavelength'])*recalexp
				M1int = interp1d(obswave,M1interp[0],kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				M1interp = M1int(specdata[k]['wavelength'])*recalexp
				
				colorexpint = interp1d(obswave,colorexp,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				colorexpinterp = colorexpint(specdata[k]['wavelength'])
				colorlawint = interp1d(obswave,colorlaw,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				colorlawinterp = colorlawint(specdata[k]['wavelength'])

				modulatedFlux = x0*(M0interp + x1*M1interp)
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

				if self.specrecal : 
					drecaltermdrecal=(((specdata[k]['wavelength']-np.mean(specdata[k]['wavelength']))/self.specrange_wavescale_specrecal)[:,np.newaxis] ** (coeffs.size-1-np.arange(coeffs.size))[np.newaxis,:]) / factorial(np.arange(coeffs.size))[np.newaxis,:]
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
		
	def specUncertaintyForSN(self,x,sn,componentsModInterp,colorlaw,colorexp,interr1d,computeDerivatives):
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
		specresultsdict['fluxuncertainty'] =  np.zeros(nspecdata)
		specresultsdict['modeluncertainty'] =  np.zeros(nspecdata)
		if computeDerivatives:
			specresultsdict['modeluncertainty_jacobian']=np.zeros([nspecdata,self.npar])
		iSpecStart = 0
		for k in specdata.keys():
			SpecLen = specdata[k]['flux'].size
			phase=specdata[k]['tobs']+tpkoff
			clippedPhase=np.clip(phase,obsphase.min(),obsphase.max())
			
			#Define recalibration factor
			coeffs=x[self.parlist=='specrecal_{}_{}'.format(sn,k)]
			coeffs/=factorial(np.arange(len(coeffs)))
			recalexp = np.exp(np.poly1d(coeffs)((specdata[k]['wavelength']-np.mean(specdata[k]['wavelength']))/self.specrange_wavescale_specrecal))
			modelErrInt = [ interp1d( obswave,interr(clippedPhase)[0],kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True) for  interr in interr1d]

			if computeDerivatives:
				colorexpint = interp1d(obswave,colorexp,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				colorexpinterp = colorexpint(specdata[k]['wavelength'])
				colorlawint = interp1d(obswave,colorlaw,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				colorlawinterp = colorlawint(specdata[k]['wavelength'])

				modelerrnox = [  interr( specdata[k]['wavelength']) *recalexp**2 for interr in (modelErrInt)]
				modelUncertainty=np.sqrt( sum([ modelerr * x1**i for i,modelerr in enumerate(modelerrnox)]))
			else:
				modelUncertainty=recalexp * np.sqrt( sum([ interr( specdata[k]['wavelength']) * x1**i for i,interr in enumerate(modelErrInt)]))
			
			specresultsdict['fluxuncertainty'][iSpecStart:iSpecStart+SpecLen] = specdata[k]['fluxerr']
			specresultsdict['modeluncertainty'][iSpecStart:iSpecStart+SpecLen] = x0* modelUncertainty

			#
		
			# derivatives....
			if computeDerivatives:
				intmultnox = _SCALE_FACTOR/(1+z)*recalexp*colorexpinterp*self.datadict[sn]['mwextcurveint'](specdata[k]['wavelength'])

			
				specresultsdict['modeluncertainty_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist == 'c_{}'.format(sn))[0][0]] = modelUncertainty * x0 *np.log(10)*colorlawinterp
				specresultsdict['modeluncertainty_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist == 'x0_{}'.format(sn))[0][0]] = modelUncertainty
				specresultsdict['modeluncertainty_jacobian'][iSpecStart:iSpecStart+SpecLen,np.where(self.parlist == 'x1_{}'.format(sn))[0][0]] = x0/2 *sum([ (i * modelerr * x1**(i-1))  if i>0 else 0 for i,modelerr in enumerate(modelerrnox)])/modelUncertainty

				if self.specrecal : 
					drecaltermdrecal=(((specdata[k]['wavelength']-np.mean(specdata[k]['wavelength']))/self.specrange_wavescale_specrecal)[:,np.newaxis] ** (coeffs.size-1-np.arange(coeffs.size))[np.newaxis,:]) / factorial(np.arange(coeffs.size))[np.newaxis,:]
					specresultsdict['modeluncertainty_jacobian'][iSpecStart:iSpecStart+SpecLen,self.parlist == 'specrecal_{}_{}'.format(sn,k)]  = x0* modelUncertainty[:,np.newaxis] * drecaltermdrecal
			
				# color law
				for i in range(self.n_colorpars):
					dcolorlaw_dcli = interp1d(obswave,SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)(self.wave)-SALT2ColorLaw(self.colorwaverange, np.zeros(self.n_colorpars))(self.wave),kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)(specdata[k]['wavelength'])
					specresultsdict['modeluncertainty_jacobian'][iSpecStart:iSpecStart+SpecLen,self.iCL[i]] = (-0.4*x0*np.log(10)*c)*modelUncertainty*dcolorlaw_dcli
			
				interpresult= self.errorspline_deriv_interp((clippedPhase[0]/(1+z),specdata[k]['wavelength']/(1+z)),method=self.interpMethod) * (intmultnox**2 * x0/2  / modelUncertainty)[:,np.newaxis]
				for i in range(3):
					mErrIdx=np.where(self.parlist=='modelerr_{}'.format(i))[0]
					specresultsdict['modeluncertainty_jacobian'][iSpecStart:iSpecStart+SpecLen,mErrIdx] = interpresult* x1**i 

			iSpecStart += SpecLen
		
		return specresultsdict

	def photUncertaintyForSN(self,x,sn,componentsModInterp,colorlaw,colorexp,interr1d,computeDerivatives):
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
		photresultsdict['fluxuncertainty'] =  np.zeros(len(photdata['filt']))
		photresultsdict['modeluncertainty'] =  np.zeros(len(photdata['filt']))
		if computeDerivatives:
			photresultsdict['modeluncertainty_jacobian']=np.zeros([photdata['filt'].size,self.npar])
		for flt in np.unique(photdata['filt']):
			#Select data from the appropriate filter filter
			selectFilter=(photdata['filt']==flt)
			phase=photdata['tobs']+tpkoff
			phase=phase[selectFilter]
			clippedPhase=np.clip(phase,obsphase.min(),obsphase.max())
			nphase = len(phase)
			
			modulatedModelErr = [  pbspl[flt] * interr(clippedPhase)[:,idx[flt]] for  interr in interr1d]
			modelErrnox=[ np.sum(modelerr,axis=1) for i,modelerr in enumerate(modulatedModelErr)]
			modelErrNoNorm=np.sqrt( sum([ modelerr * x1**i for i,modelerr in enumerate(modelErrnox)])) 
			modelUncertainty=self.fluxfactor[survey][flt]*np.sqrt(pbspl[flt].sum()) * x0*dwave* modelErrNoNorm 
			
			# modelflux
			photresultsdict['fluxuncertainty'][selectFilter] = photdata['fluxcalerr'][selectFilter]
			photresultsdict['modeluncertainty'][selectFilter] = (modelUncertainty)
			
			if computeDerivatives:					
				intmult=self.fluxfactor[survey][flt]*np.sqrt(pbspl[flt].sum()) * dwave
				photresultsdict['modeluncertainty_jacobian'][selectFilter,self.parlist == 'x0_{}'.format(sn)] = intmult    * modelErrNoNorm
				photresultsdict['modeluncertainty_jacobian'][selectFilter,self.parlist == 'x1_{}'.format(sn)] = intmult* x0 / 2* sum([ i*modelerr * x1**(i-1) if i>0 else 0 for i,modelerr in enumerate(modelErrnox)]) / modelErrNoNorm

				photresultsdict['modeluncertainty_jacobian'][selectFilter,self.parlist == 'c_{}'.format(sn)]  = intmult* x0* np.log(10)*sum([ np.sum(colorlaw[np.newaxis,idx[flt]] * modelerr,axis=1) * x1**i for i,modelerr in enumerate(modulatedModelErr)])/modelErrNoNorm
				
				for i in range(self.n_colorpars):
					#Color law is linear wrt to the color law parameters; therefore derivative of the color law
					# with respect to color law parameter i is the color law with all other values zeroed minus the color law with all values zeroed
					dcolorlaw_dcli = SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)(self.wave[idx[flt]])-SALT2ColorLaw(self.colorwaverange, np.zeros(self.n_colorpars))(self.wave[idx[flt]])
					#Multiply M0 and M1 components (already modulated with passband) by c* d colorlaw / d cl_i, with associated normalizations
#					import pdb;pdb.set_trace()
					photresultsdict['modeluncertainty_jacobian'][selectFilter,self.iCL[i]] =  intmult* x0 *c* -0.4*np.log(10)*sum([ np.sum(dcolorlaw_dcli[np.newaxis,:] * modelerr,axis=1) * x1**i for i,modelerr in enumerate(modulatedModelErr)])/modelErrNoNorm

				
				passbandColorExp=(pbspl[flt]*(colorexp[idx[flt]]*self.datadict[sn]['mwextcurve'][idx[flt]])**2)
				
				intmult = dwave*self.fluxfactor[survey][flt]*x0*np.sqrt(pbspl[flt].sum())*(_SCALE_FACTOR/(1+z))**2
				
				for pdx,p in enumerate(np.where(selectFilter)[0]):
					derivInterp = self.errorspline_deriv_interp(
								(clippedPhase[pdx]/(1+z),self.wave[idx[flt]]),method=self.interpMethod)
					summation = intmult /2 * np.sum( passbandColorExp.reshape(passbandColorExp.size,1) * derivInterp, axis=0) /  modelErrNoNorm[pdx]
					for i in range(3):
						mErrIdx=np.where(self.parlist=='modelerr_{}'.format(i))[0] 
						photresultsdict['modeluncertainty_jacobian'][p,mErrIdx]=  x1**i * summation
						
					
		return photresultsdict
		
	def photValsForSN(self,x,sn,componentsModInterp,colorlaw,colorexp,computeDerivatives,computePCDerivs):
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
		else:
			int1d=componentsModInterp
		photresultsdict={}
		photresultsdict['modelflux'] = np.zeros(len(photdata['filt']))
		photresultsdict['dataflux'] = photdata['fluxcal']
		if computeDerivatives:
			photresultsdict['modelflux_jacobian'] = np.zeros((photdata['filt'].size,self.npar))
			photresultsdict['modeluncertainty_jacobian']=np.zeros([photdata['filt'].size,self.npar])
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

				modulatedM0= pbspl[flt]*M0interp[:,idx[flt]]
				modulatedM1=pbspl[flt]*M1interp[:,idx[flt]]

				if ( (phase>obsphase.max())).any():
					decayFactor=10**(-0.4*self.extrapolateDecline*(phase[phase>obsphase.max()]-obsphase.max()))
					modulatedM0[np.where(phase>obsphase.max())[0]] *= decayFactor
					modulatedM1[np.where(phase>obsphase.max())[0]] *= decayFactor
						
				modelsynM0flux=np.sum(modulatedM0, axis=1)*dwave*self.fluxfactor[survey][flt]
				modelsynM1flux=np.sum(modulatedM1, axis=1)*dwave*self.fluxfactor[survey][flt]
				
				photresultsdict['modelflux_jacobian'][selectFilter,self.parlist == 'x0_{}'.format(sn)] = modelsynM0flux+ x1*modelsynM1flux
				photresultsdict['modelflux_jacobian'][selectFilter,self.parlist == 'x1_{}'.format(sn)] = modelsynM1flux*x0
				
				modulatedFlux= x0*(modulatedM0 +modulatedM1*x1)
				modelflux = x0* (modelsynM0flux+ x1*modelsynM1flux)
			
				#Need to figure out how to handle derivatives wrt time when dealing with nearest neighbor interpolation; maybe require linear?
				for p in np.where(phase>obsphase.max())[0]:
				
					photresultsdict['modelflux_jacobian'][np.where(selectFilter)[0][p],self.parlist=='tpkoff_{}'.format(sn)]=-0.4*np.log(10)*self.extrapolateDecline*modelflux[p]
			else:
				modinterp = int1d(clippedPhase)
				modelflux = np.sum(pbspl[flt]*modinterp[:,idx[flt]], axis=1)*dwave*self.fluxfactor[survey][flt]
				if ( (phase>obsphase.max())).any():
					modelflux[(phase>obsphase.max())]*= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))[(phase>obsphase.max())]
			
			
			photresultsdict['modelflux'][selectFilter] = modelflux
			
			if computeDerivatives:
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
					
						summation = np.sum( passbandColorExp[:,:,np.newaxis] * derivInterp, axis=1)
						photresultsdict['modelflux_jacobian'][p,self.im0]=  summation*intmult
						photresultsdict['modelflux_jacobian'][p,self.im1] = summation*intmult*x1
						self.__dict__['dmodelflux_dM0_phot_%s'%sn][p,:] = summation*intmultnox
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
	
	
	def modelvalsforSN(self,x,sn,components,colorLaw,saltErr,computeDerivatives,computePCDerivs,fixUncertainty):
		# model pars, initialization
		M0,M1 = copy.deepcopy(components)
		z = self.datadict[sn]['zHelio']
		obsphase = self.datadict[sn]['obsphase'] #self.phase*(1+z)
		x0,x1,c,tpkoff = x[self.parlist == 'x0_%s'%sn],x[self.parlist == 'x1_%s'%sn],\
						 x[self.parlist == 'c_%s'%sn],x[self.parlist == 'tpkoff_%s'%sn]

		
		#Apply MW extinction
		M0 *= self.datadict[sn]['mwextcurve'][np.newaxis,:]
		M1 *= self.datadict[sn]['mwextcurve'][np.newaxis,:]
		
		if colorLaw:
			colorlaw = -0.4 * colorLaw(self.wave)
			colorexp = 10. ** (colorlaw * c)
		else:
			colorexp=1

		if computeDerivatives: M0 *= colorexp; M1 *= colorexp
		else: mod = x0*(M0 + x1*M1)*colorexp
				
		if computeDerivatives:
			M0 *= _SCALE_FACTOR/(1+z); M1 *= _SCALE_FACTOR/(1+z)
			int1dM0 = interp1d(obsphase,M0,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
			int1dM1 = interp1d(obsphase,M1,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
		else:
			mod *= _SCALE_FACTOR/(1+z)
			int1d = interp1d(obsphase,mod,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
		
		interr1d = [interp1d(obsphase,err * (self.datadict[sn]['mwextcurve'] *colorexp*  _SCALE_FACTOR/(1+z))**2 ,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True) for err in saltErr]
		returndicts=[]
		for valfun,uncertaintyfun,name in [(self.photValsForSN,self.photUncertaintyForSN,'phot'),(self.specValsForSN,self.specUncertaintyForSN, 'spec')]:
			valdict=valfun(x,sn,(int1dM0,int1dM1) if computeDerivatives else int1d,colorlaw,colorexp,computeDerivatives,computePCDerivs)
			key='__{}_{}_fixed_uncertainty__'.format(name,sn)
			if fixUncertainty: 
				try:
					uncertaintydict=self.__dict__[key]
				except KeyError:
					fixUncertainty=False
			if not fixUncertainty:
				#Otherwise, store current uncertainties 
				uncertaintydict=uncertaintyfun(x,sn,(int1dM0,int1dM1) if computeDerivatives else int1d,colorlaw,colorexp,interr1d,computeDerivatives)
				self.__dict__[key]=uncertaintydict
			valdict.update(uncertaintydict)
			returndicts+=[valdict]
		return returndicts
			
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

	def loglikeforSN(self,args,sn=None,x=None,components=None,salterr=None,
					 colorLaw=None,colorScat=None,
					 debug=False,timeit=False,computeDerivatives=False,computePCDerivs=False):
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
		if salterr is None:
			salterr = self.ErrModel(x)

		photResidsDict,specResidsDict = self.ResidsForSN(x,sn,components,colorLaw,salterr,computeDerivatives,computePCDerivs,fixUncertainty=False)
		
		loglike= - (photResidsDict['resid']**2).sum() / 2.   -(specResidsDict['resid']**2).sum()/2.+photResidsDict['lognorm']+specResidsDict['lognorm']
		if computeDerivatives: 
			grad_loglike=  - (photResidsDict['resid'][:,np.newaxis] * photResidsDict['resid_jacobian']).sum(axis=0) - (specResidsDict['resid'][:,np.newaxis] * specResidsDict['resid_jacobian']).sum(axis=0) + photResidsDict['lognorm_grad'] +specResidsDict['lognorm_grad']
			return loglike,grad_loglike
		else:
			return loglike
				
	def specchi2(self):

		return chi2
	
	def SALTModel(self,x,evaluatePhase=None,evaluateWave=None):
		
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

	
	def ErrModel(self,x,evaluatePhase=None,evaluateWave=None):
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
		m0err = np.zeros(len(x[self.parlist == 'm0']))
		m1pars = x[self.parlist == 'm1']
		m1err = np.zeros(len(x[self.parlist == 'm1']))
		
		# covmat (diagonals only?)
		m0_m1_cov = np.zeros(len(m0pars))

		modelerrpars = x[self.parlist == 'modelerr']
		modelerrerr = np.zeros(len(x[self.parlist == 'modelerr']))

		clpars = x[self.parlist == 'cl']
		clerr = np.zeros(len(x[self.parlist == 'cl']))
		
		clscatpars = x[self.parlist == 'clscat']
		clscaterr = np.zeros(x[self.parlist == 'clscat'])
		

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
		clscat = splev(self.wave,(self.errwaveknotloc,clscatpars,3))
		if not len(clpars): clpars = []

		return(x,self.phase,self.wave,m0,m0err,m1,m1err,cov_m0_m1,modelerr,
			   clpars,clerr,clscat,resultsdict)

	def getPars(self,loglikes,x,nburn=500,mkplots=False):

		axcount = 0; parcount = 0
		from matplotlib.backends.backend_pdf import PdfPages
		pdf_pages = PdfPages('output/MCMC_hist.pdf')
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
		self.neff=np.zeros((self.phaseBinCenters.size,self.waveBinCenters.size))
		phaseIndices,waveIndices=np.unravel_index(np.arange(self.im0.size),(self.phaseBinCenters.size,self.waveBinCenters.size))
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
				self.neff[np.where(basisAffected.reshape(self.neff.shape))]+=result
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
					self.neff[np.where(basisAffected[pdx].reshape(self.neff.shape))]+=summation
# 			phottime+=time.time()
# 		print('Time for total neff is ',time.time()-start)
# 		print('Spectime: ',spectime,'Phottime: ',phottime)

		#Smear it out a bit along phase axis
		#self.neff=gaussian_filter1d(self.neff,1,0)

		self.neff=np.clip(self.neff,1e-10*self.neff.max(),None)
		# hack!
		#self.plotEffectivePoints([-12.5,0,12.5,40],'neff.png')
		#self.plotEffectivePoints(None,'neff-heatmap.png')

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
			inds=np.searchsorted(self.phaseBinCenters,phases)
			# hack!
			for i in inds:
				plt.plot(self.waveBins[:-1],self.neff[i,:],label='{:.1f} days'.format(self.phaseBinCenters[i]))
			plt.ylabel('$N_eff$')
			plt.xlabel('$\lambda (\AA)$')
			plt.xlim(self.phaseBinCenters.min(),self.phaseBinCenters.max())
			plt.legend()
		
		if output is None:
			plt.show()
		else:
			plt.savefig(output,dpi=288)
		plt.clf()


	def dyadicRegularization(self,x, computeJac=True):
		phase=self.phaseBinCenters
		wave=self.waveBinCenters
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdwave=self.SALTModelDeriv(x,0,1,phase,wave)
		dfluxdphase=self.SALTModelDeriv(x,1,0,phase,wave)
		d2fluxdphasedwave=self.SALTModelDeriv(x,1,1,phase,wave)
		resids=[]
		jac=[]
		for i in range(len(fluxes)):
			#Determine a scale for the fluxes by sum of squares (since x1 can have negative values)
			scale=np.sqrt(np.mean(fluxes[i]**2))
			#Derivative of scale with respect to model parameters
			scaleDeriv= np.mean(fluxes[i][:,:,np.newaxis]*self.regularizationDerivs[0],axis=(0,1))/scale
			#Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
			normalization=np.sqrt(1/( (self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))
			#0 if model is locally separable in phase and wavelength i.e. flux=g(phase)* h(wavelength) for arbitrary functions g and h
			numerator=(dfluxdphase[i] *dfluxdwave[i] -d2fluxdphasedwave[i] *fluxes[i] )
			dnumerator=( self.regularizationDerivs[1]*dfluxdwave[i][:,:,np.newaxis] + self.regularizationDerivs[2]* dfluxdphase[i][:,:,np.newaxis] - self.regularizationDerivs[3]* fluxes[i][:,:,np.newaxis] - self.regularizationDerivs[0]* d2fluxdphasedwave[i][:,:,np.newaxis] )			
			resids += [normalization* (numerator / (scale**2 * np.sqrt( self.neff ))).flatten()]
			if computeJac: jac += [((dnumerator*(scale**2 )- scaleDeriv[np.newaxis,np.newaxis,:]*2*scale*numerator[:,:,np.newaxis])/np.sqrt(self.neff)[:,:,np.newaxis]*normalization / scale**4  ).reshape(-1, self.im0.size)]
			else: jac+=[None]
		return resids,jac 
	
	def phaseGradientRegularization(self, x, computeJac=True):
		phase=self.phaseBinCenters
		wave=self.waveBinCenters
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdphase=self.SALTModelDeriv(x,1,0,phase,wave)
		resids=[]
		jac=[]
		for i in range(len(fluxes)):
			#Determine a scale for the fluxes by sum of squares (since x1 can have negative values)
			scale=np.sqrt(np.mean(fluxes[i]**2))
			#Normalize gradient by flux scale
			normedGrad=dfluxdphase[i]/scale
			#Derivative of scale with respect to model parameters
			scaleDeriv= np.mean(fluxes[i][:,:,np.newaxis]*self.regularizationDerivs[0],axis=(0,1))/scale
			#Derivative of normalized gradient with respect to model parameters
			normedGradDerivs=(self.regularizationDerivs[1] * scale - scaleDeriv[np.newaxis,np.newaxis,:]*dfluxdphase[i][:,:,np.newaxis])/ scale**2
			#Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
			normalization=np.sqrt(1/((self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))
			#Minimize model derivative w.r.t wavelength in unconstrained regions
			resids+= [normalization* ( normedGrad /	np.sqrt( self.neff )).flatten()]
			if computeJac: jac+= [normalization*((normedGradDerivs) / np.sqrt( self.neff )[:,:,np.newaxis]).reshape(-1, self.im0.size)]
			else: jac+=[None]
		return resids,jac  
	
	def waveGradientRegularization(self, x,computeJac=True):
		phase=self.phaseBinCenters
		wave=self.waveBinCenters
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdwave=self.SALTModelDeriv(x,0,1,phase,wave)
		waveGradResids=[]
		jac=[]
		for i in range(len(fluxes)):
			#Determine a scale for the fluxes by sum of squares (since x1 can have negative values)
			scale=np.sqrt(np.mean(fluxes[i]**2))
			#Normalize gradient by flux scale
			normedGrad=dfluxdwave[i]/scale
			#Derivative of scale with respect to model parameters
			scaleDeriv= np.mean(fluxes[i][:,:,np.newaxis]*self.regularizationDerivs[0],axis=(0,1))/scale
			#Derivative of normalized gradient with respect to model parameters
			normedGradDerivs=(self.regularizationDerivs[2] * scale - scaleDeriv[np.newaxis,np.newaxis,:]*dfluxdwave[i][:,:,np.newaxis])/ scale**2
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
