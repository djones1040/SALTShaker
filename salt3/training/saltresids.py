from salt3.util.synphot import synphot
from salt3.training import init_hsiao
from salt3.training.priors import SALTPriors
from salt3.util.readutils import SALTtrainingSN,SALTtraininglightcurve,SALTtrainingspectrum

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
from scipy import linalg,sparse

import numpy as np
from numpy.random import standard_normal
from numpy.linalg import slogdet

from astropy.cosmology import Planck15 as cosmo
from multiprocessing import Pool, get_context
from inspect import signature
from functools import partial
from itertools import starmap

import time
import matplotlib as mpl
mpl.use('agg')
import pylab as plt
import extinction
import copy
import warnings

import logging
log=logging.getLogger(__name__)

_SCALE_FACTOR = 1e-12
_B_LAMBDA_EFF = np.array([4302.57])	 # B-band-ish wavelength
_V_LAMBDA_EFF = np.array([5428.55])	 # V-band-ish wavelength
warnings.simplefilter('ignore',category=FutureWarning)
	
def rankOneCholesky(variance,beta,v):
	"""Given variances, a scalar, and a vector, returns the cholesky matrix describing the covariance formed by the sum of the diagonal variance and the self outer product of the vector multiplied by the scalar"""
	b=1
	Lprime=np.zeros((variance.size,variance.size))
	if beta==0: return np.diag(np.sqrt(variance))
	for j in range(v.size):
		Lprime[j,j]=np.sqrt(variance[j]+beta/b*v[j]**2)
		gamma=(b*variance[j]+beta*v[j]**2)
		Lprime[j+1:,j]=Lprime[j,j]*beta*v[j+1:]*v[j]/gamma
		b+=beta*v[j]**2/variance[j]
	return Lprime

#			
#			idx = self.sncachedresults[snid]['idx']
#			tpkoff=X[self.parlist == f'tpkoff_{snid}']
#			obsphase = self.sncachedresults[snid]['obsphase']
#			for k,spectrum in sn.specdata.items():
#				self.pcderivsparse[f'pcDeriv_spec_{snid}_{k}']=sparse.csr_matrix(self.spline_deriv_interp((spectrum.phase,spectrum.wavelength/(1+z)),method=self.interpMethod))
#				
#			for flt in sn.filt:

class SALTfitcachelightcurve(SALTtraininglightcurve):

	__slots__ = ['idx','pbspl','denom','lambdaeff','pcderivsparse']
	
	def __init__(self,sn,lc,residsobj,kcordict):
		for key,val in lc.__dict__.items():
			self.__dict__[key]=copy.deepcopy(val)
		z = sn.zHelio
		
		#Define quantities for synthetic photometry
		filtwave = kcordict[sn.survey][lc.filt]['filtwave']
		filttrans = kcordict[sn.survey][lc.filt]['filttrans']
		self.idx   = (sn.obswave>= kcordict[sn.survey][lc.filt]['minlam']) & \
					 (sn.obswave<= kcordict[sn.survey][lc.filt]['maxlam'])	# overlap range
		pbspl = np.interp(sn.obswave[self.idx],filtwave,filttrans)
		pbspl *= sn.obswave[self.idx]
		denom = np.trapz(pbspl,sn.obswave[self.idx])
		pbspl /= denom*HC_ERG_AA
		self.pbspl = pbspl[np.newaxis,:]
		self.denom = denom
		self.lambdaeff = kcordict[sn.survey][lc.filt]['lambdaeff']
		
		clippedPhase=np.clip(self.phase,residsobj.phase.min(),residsobj.phase.max())
		#Array output indices match time along 0th axis, wavelength along 1st axis
		result=[]
		for pdx in range(len(lc)):
			derivInterp = residsobj.spline_deriv_interp(
				(clippedPhase[pdx],residsobj.wave[self.idx]),
				method=residsobj.interpMethod)
			if self.phase[pdx]>sn.obsphase.max():
				decayFactor= 10**(-0.4*residsobj.extrapolateDecline*(self.phase[pdx]-sn.obsphase.max()))
				derivInterp*=decayFactor
			result+=[sparse.csr_matrix(derivInterp)]
		self.pcderivsparse=result

class SALTfitcachespectrum(SALTtrainingspectrum):

	__slots__ = ['spectrum','pcderivsparse','ispecx0','ispcrcl','phase','wavelength','flux','fluxerr','tobs']
	
	def __init__(self,sn,spectrum,k,residsobj,kcordict):
		#for key,val in spectrum.__dict__.items():
		#	self.__dict__[key]=copy.deepcopy(val)

		self.flux = spectrum.flux
		self.phase = spectrum.phase
		self.wavelength = spectrum.wavelength
		self.flux = spectrum.flux
		self.fluxerr = spectrum.fluxerr
		self.tobs = spectrum.tobs
		
		z = sn.zHelio
		self.pcderivsparse=sparse.csr_matrix(
			residsobj.spline_deriv_interp((spectrum.phase,spectrum.wavelength/(1+z)),
										  method=residsobj.interpMethod))
		self.spectrumid=k
		self.ispecx0=np.where(residsobj.parlist=='specx0_{}_{}'.format(sn.snid,k))[0]
		self.ispcrcl=np.where(residsobj.parlist=='specrecal_{}_{}'.format(sn.snid,k))[0]

class SALTfitcacheSN(SALTtrainingSN):
	"""Class to store SN data in addition to cached results useful in speeding up the fitter
	"""
	__slots__ = ['photdata','specdata','ix0','ix1','ic','itpkoff','mwextcurve','mwextcurve','mwextcurveint','dwave','obswave','obsphase']
	
	def __init__(self,sndata,residsobj,kcordict):
		for key,val in sndata.__dict__.items():
			if key=='photdata' or  key=='specdata':
				pass
			else:
				self.__dict__[key]=copy.deepcopy(val)
		self.obswave = residsobj.wave*(1+self.zHelio)
		self.obsphase = residsobj.phase*(1+self.zHelio)
		self.dwave = residsobj.wave[1]*(1+self.zHelio) - residsobj.wave[0]*(1+self.zHelio)
		self.mwextcurve	  = 10**(-0.4*extinction.fitzpatrick99(self.obswave,sndata.MWEBV*3.1))
		self.mwextcurveint = interp1d(
			self.obswave,self.mwextcurve ,kind=residsobj.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
		
		self.ix0=np.where(residsobj.parlist==f'x0_{self.snid}')[0][0]
		self.ix1=np.where(residsobj.parlist==f'x1_{self.snid}')[0][0]
		self.ic=np.where(residsobj.parlist==f'c_{self.snid}')[0][0]
		self.itpkoff=np.where(residsobj.parlist==f'tpkoff_{self.snid}')[0][0]
		
		self.photdata={flt: SALTfitcachelightcurve(self,sndata.photdata[flt],residsobj,kcordict ) for flt in sndata.photdata}
		self.specdata={k: SALTfitcachespectrum(self,sndata.specdata[k],k,residsobj,kcordict ) for k in sndata.specdata }

	
class SALTResids:
	def __init__(self,guess,datadict,parlist,**kwargs):

		self.photresultsdict = {}
		self.specresultsdict = {}
		self.options=kwargs
		assert type(parlist) == np.ndarray
		self.nstep = 0
		self.parlist = parlist
		self.npar = len(parlist)
		self.datadict = datadict
		
		self.bsorder=3
		self.guess = guess
		self.nsn = len(self.datadict.keys())
		
		for key, value in kwargs.items(): 
			self.__dict__[key] = value

		self.usePriors = []
		self.priorWidths = []
		self.boundedParams = []
		self.bounds = []

		for opt in self.__dict__.keys():
			if opt.startswith('prior_'):
				self.usePriors += [opt[len('prior_'):]]
				self.priorWidths += [self.__dict__[opt]]
			elif opt.startswith('bound_'):
				self.boundedParams += [opt[len('bound_'):]]
				self.bounds += [tuple([float(x) for x in self.__dict__[opt]])]
				
			#self.usePriors=[x.split('prior_')[-1] for x in ] #self.usePriors.split(',')
			#self.priorWidths=[float(x) for x in self.priorWidths.split(',')]

		# pre-set some indices
		self.set_param_indices()
		
		# set some phase/wavelength arrays
		self.phase = np.linspace(self.phaserange[0],self.phaserange[1],
								 int((self.phaserange[1]-self.phaserange[0])/self.phaseinterpres)+1,True)
		self.phaseout = np.linspace(self.phaserange[0],self.phaserange[1],
								 int((self.phaserange[1]-self.phaserange[0])/self.phaseoutres)+1,True)
		self.interpMethod='nearest'
		
		self.wave = np.linspace(self.waverange[0],self.waverange[1],
								int((self.waverange[1]-self.waverange[0])/self.waveinterpres)+1,True)
		self.waveout = np.linspace(self.waverange[0],self.waverange[1],
								int((self.waverange[1]-self.waverange[0])/self.waveoutres)+1,True)
				
		self.neff=0

		# initialize the model
		self.components = self.SALTModel(guess)
		self.salterr = self.ErrModel(guess)

		self.m0guess = -19.49 #10**(-0.4*(-19.49-27.5))
		self.extrapolateDecline=0.015
		# set up the filters
		self.stdmag = {}
		self.fluxfactor = {}
		
		#Check that we've got suitable filters!
		for snid,sn in self.datadict.items():
			if sn.survey not in self.kcordict:
				raise ValueError(f'Could not find corresponding kcor for survey {sn.survey} for SN {snid} ')

		for survey in self.kcordict:
			if survey == 'default': 
				self.stdmag[survey] = {}
				self.bbandoverlap = (
					self.wave>=self.kcordict['default']['Bwave'].min())&\
					(self.wave<=self.kcordict['default']['Bwave'].max())
				self.bbandpbspl = np.interp(
					self.wave[self.bbandoverlap],self.kcordict['default']['Bwave'],
					self.kcordict['default']['Btp'])
				self.bbandpbspl *= self.wave[self.bbandoverlap]
				self.bbandpbspl /= np.trapz(self.bbandpbspl,self.wave[self.bbandoverlap])*HC_ERG_AA
				self.stdmag[survey]['B']=synphot(
					self.kcordict[survey]['primarywave'],self.kcordict[survey]['AB'],
					filtwave=self.kcordict['default']['Bwave'],filttp=self.kcordict[survey]['Btp'],
					zpoff=0)
				self.stdmag[survey]['V']=synphot(
					self.kcordict[survey]['primarywave'],self.kcordict[survey]['AB'],
					filtwave=self.kcordict['default']['Vwave'],filttp=self.kcordict[survey]['Vtp'],
					zpoff=0)
				self.fluxfactor['default']={}
				self.fluxfactor[survey]['B'] = 10**(0.4*(self.stdmag[survey]['B']+27.5))
				self.fluxfactor[survey]['V'] = 10**(0.4*(self.stdmag[survey]['V']+27.5))
				continue

			self.stdmag[survey] = {}
			self.fluxfactor[survey] = {}
			primarywave = self.kcordict[survey]['primarywave']
			for flt in self.kcordict[survey].keys():
				if flt == 'filtwave' or flt == 'primarywave' or \
				   flt == 'snflux' or flt == 'AB' or \
				   flt == 'BD17' or flt == 'Vega': continue
				if self.kcordict[survey][flt]['magsys'] == 'AB': primarykey = 'AB'
				elif self.kcordict[survey][flt]['magsys'].upper() == 'VEGA': primarykey = 'Vega'
				elif self.kcordict[survey][flt]['magsys'] == 'BD17': primarykey = 'BD17'
				self.stdmag[survey][flt] = synphot(
					primarywave,self.kcordict[survey][primarykey],
					filtwave=self.kcordict[survey][flt]['filtwave'],
					filttp=self.kcordict[survey][flt]['filttrans'],
					zpoff=0) - self.kcordict[survey][flt]['primarymag']
				self.fluxfactor[survey][flt] = 10**(0.4*(self.stdmag[survey][flt]+27.5))
				self.kcordict[survey][flt]['minlam'] = \
					np.min(self.kcordict[survey][flt]['filtwave'][self.kcordict[survey][flt]['filttrans'] > 0.01])
				self.kcordict[survey][flt]['maxlam'] = \
					np.max(self.kcordict[survey][flt]['filtwave'][self.kcordict[survey][flt]['filttrans'] > 0.01])
		# rest-frame B
		filttrans = self.kcordict['default']['Btp']
		filtwave = self.kcordict['default']['Bwave']
			
		pbspl = np.interp(self.wave,filtwave,filttrans,left=0,right=0)
		
		pbspl *= self.wave
		denom = np.trapz(pbspl,self.wave)
		pbspl /= denom*HC_ERG_AA
		self.kcordict['default']['Bpbspl'] = pbspl
		self.kcordict['default']['dwave'] = self.wave[1] - self.wave[0]
		
		#rest-frame V
		filttrans = self.kcordict['default']['Vtp']
		filtwave = self.kcordict['default']['Vwave']
			
		pbspl = np.interp(self.wave,filtwave,filttrans,left=0,right=0)
		
		pbspl *= self.wave
		denom = np.trapz(pbspl,self.wave)
		pbspl /= denom*HC_ERG_AA
		self.kcordict['default']['Vpbspl'] = pbspl
				
		#Count number of photometric and spectroscopic points
		self.num_spec=sum([datadict[sn].num_specobs for sn in datadict])
		self.num_spectra=sum([datadict[sn].num_spec for sn in datadict])
		self.num_lc=sum([datadict[sn].num_lc for sn in datadict])
		self.num_phot=sum([datadict[sn].num_photobs for sn in datadict])

		self.fixedUncertainties={}
		starttime=time.time()
		#Store derivatives of a spline with fixed knot locations with respect to each knot value
		self.spline_derivs = np.zeros([len(self.phase),len(self.wave),self.im0.size])
		for i in range(self.im0.size):
			if self.bsorder == 0: continue
			self.spline_derivs[:,:,i]=bisplev(
				self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder))
		nonzero=np.nonzero(self.spline_derivs)
		self.spline_deriv_interp= RegularGridInterpolator((self.phase,self.wave),self.spline_derivs,self.interpMethod,False,0)
		
		#Repeat for the error model parameters
		self.errorspline_deriv= np.zeros([len(self.phase),len(self.wave),self.imodelerr.size//self.n_components])
		for i in range(self.imodelerr.size//self.n_components):
			if self.errbsorder==0:
				continue
			self.errorspline_deriv[:,:,i]=bisplev(
				self.phase, self.wave ,(self.errphaseknotloc,self.errwaveknotloc,
										np.arange(self.imodelerr.size//self.n_components)==i,
										self.errbsorder,self.errbsorder))
		self.errorspline_deriv_interp= RegularGridInterpolator(
			(self.phase,self.wave),self.errorspline_deriv,self.interpMethod,False,0)
		
		#Store the lower and upper edges of the phase/wavelength basis functions
		self.phaseBins=self.phaseknotloc[:-(self.bsorder+1)],self.phaseknotloc[(self.bsorder+1):]
		self.waveBins=self.waveknotloc[:-(self.bsorder+1)],self.waveknotloc[(self.bsorder+1):]
		
		#Find the iqr of the phase/wavelength basis functions
		self.phaseRegularizationBins=np.linspace(self.phase[0],self.phase[-1],self.phaseBins[0].size*2+1,True)
		self.waveRegularizationBins=np.linspace(self.wave[0],self.wave[-1],self.waveBins[0].size*2+1,True)
		# HACK to match snpca
		#self.phaseRegularizationBins=np.linspace(self.phase[0],self.phase[-1],self.phaseBins[0].size+1,True)
		#self.waveRegularizationBins=np.linspace(self.wave[0],self.wave[-1],self.waveBins[0].size+1,True)

		
		self.phaseRegularizationPoints=(self.phaseRegularizationBins[1:]+self.phaseRegularizationBins[:-1])/2
		self.waveRegularizationPoints=(self.waveRegularizationBins[1:]+self.waveRegularizationBins[:-1])/2

		self.phaseBinCenters=np.array(
			[(self.phase[:,np.newaxis]* self.spline_derivs[:,:,i*(self.waveBins[0].size)]).sum()/self.spline_derivs[:,:,i*(self.waveBins[0].size)].sum() for i in range(self.phaseBins[0].size) ])
		self.waveBinCenters=np.array(
			[(self.wave[np.newaxis,:]* self.spline_derivs[:,:,i]).sum()/self.spline_derivs[:,:,i].sum() for i in range(self.waveBins[0].size)])

		#Find the basis functions evaluated at the centers of the basis functions for use in the regularization derivatives
		self.regularizationDerivs=[np.zeros((self.phaseRegularizationPoints.size,self.waveRegularizationPoints.size,self.im0.size)) for i in range(4)]
		for i in range(len(self.im0)):
			for j,derivs in enumerate([(0,0),(1,0),(0,1),(1,1)]):
				if self.bsorder == 0: continue
				self.regularizationDerivs[j][:,:,i]=bisplev(
					self.phaseRegularizationPoints,self.waveRegularizationPoints,
					(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder),
					dx=derivs[0],dy=derivs[1])

		phase=self.phaseRegularizationPoints
		wave=self.waveRegularizationPoints
		fluxes=self.SALTModel(guess,evaluatePhase=self.phaseRegularizationPoints,evaluateWave=self.waveRegularizationPoints)
		# HACK
		self.guessScale=[1.0 for f in fluxes]

		self.colorLawDeriv=np.empty((self.wave.size,self.n_colorpars))
		for i in range(self.n_colorpars):
			self.colorLawDeriv[:,i]=\
				SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)(self.wave)-\
				SALT2ColorLaw(self.colorwaverange, np.zeros(self.n_colorpars))(self.wave)
		self.colorLawDerivInterp=interp1d(
			self.wave,self.colorLawDeriv,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
		
		log.info(f'Time to calculate spline_derivs: {time.time()-starttime:.2f}')
		
		if self.regularize:
			self.updateEffectivePoints(guess)

		self.priors = SALTPriors(self)
		
		if self.fitTpkOff:
			raise NotImplementedError('Fitting the peak of the lightcurve simultaneously with the model is not currently implemented')
			
		log.info('Calculating cached quantities for speed in fitting loop')
		start=time.time()
		self.datadict={snid: SALTfitcacheSN(sn,self,self.kcordict) for snid,sn in self.datadict.items()}
		log.info('Time required to calculate cached quantities {:.1f}s'.format(time.time()-start))
			
	def set_param_indices(self):

		self.parameters = ['x0','x1','c','m0','m1','spcrcl','spcrcl_norm','spcrcl_poly',
						   'modelerr','modelcorr','clscat','clscat_0','clscat_poly']
		self.corrcombinations=sum([[(i,j) for j in range(i+1,self.n_components)]for i in range(self.n_components)] ,[])
		self.m0min = np.min(np.where(self.parlist == 'm0')[0])
		self.m0max = np.max(np.where(self.parlist == 'm0')[0])
		self.errmin = tuple([np.min(np.where(self.parlist == 'modelerr_{}'.format(i))[0]) for i in range(self.n_components)]) 
		self.errmax = tuple([np.max(np.where(self.parlist == 'modelerr_{}'.format(i))[0]) for i in range(self.n_components)]) 
		self.corrmin = tuple([np.min(np.where(self.parlist == 'modelcorr_{}{}'.format(i,j))[0]) for i,j in self.corrcombinations]) 
		self.corrmax = tuple([np.max(np.where(self.parlist == 'modelcorr_{}{}'.format(i,j))[0]) for i,j in self.corrcombinations]) 
		self.im0 = np.where(self.parlist == 'm0')[0]
		self.im1 = np.where(self.parlist == 'm1')[0]

		wavemin = []
		for i in range(self.im0.size):
			wavemin += [self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),
										  i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]][0]]
		self.im0new = np.where(self.parlist == 'm0')[0][np.array(wavemin) > 8500]
		self.im1new = np.where(self.parlist == 'm1')[0][np.array(wavemin) > 8500]
		
		self.iCL = np.where(self.parlist == 'cl')[0]
		self.ix1 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x1')],dtype=int)
		self.ix0 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x0') or si.startswith('specx0')],dtype=int)
		self.ic	 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('c_')],dtype=int)
		self.itpk = np.array([i for i, si in enumerate(self.parlist) if si.startswith('tpkoff')],dtype=int)
		self.ispcrcl_norm = np.array([i for i, si in enumerate(self.parlist) if si.startswith('specx0')],dtype=int)
		if self.ispcrcl_norm.size==0: self.ispcrcl_norm=np.zeros(self.npar,dtype=bool)
		self.ispcrcl = np.array([i for i, si in enumerate(self.parlist) if si.startswith('spec')],dtype=int) # used to be specrecal
		if self.ispcrcl.size==0: self.ispcrcl=np.zeros(self.npar,dtype=bool)
		self.imodelerr = np.array([i for i, si in enumerate(self.parlist) if si.startswith('modelerr')],dtype=int)
		self.imodelerr0 = np.array([i for i, si in enumerate(self.parlist) if si ==('modelerr_0')],dtype=int)
		self.imodelerr1 = np.array([i for i, si in enumerate(self.parlist) if si==('modelerr_1')],dtype=int)
		self.imodelcorr = np.array([i for i, si in enumerate(self.parlist) if si.startswith('modelcorr')],dtype=int)
		self.iclscat = np.where(self.parlist=='clscat')[0]

		self.iModelParam=np.ones(self.npar,dtype=bool)
		self.iModelParam[self.imodelerr]=False
		self.iModelParam[self.imodelcorr]=False
		self.iModelParam[self.iclscat]=False

		self.iclscat_0,self.iclscat_poly = np.array([],dtype='int'),np.array([],dtype='int')
		if len(self.ispcrcl):
			for i,parname in enumerate(np.unique(self.parlist[self.iclscat])):
				self.iclscat_0 = np.append(self.iclscat_0,np.where(self.parlist == parname)[0][-1])
				self.iclscat_poly = np.append(self.iclscat_poly,np.where(self.parlist == parname)[0][:-1])
		
	
	def lsqwrap(self,guess,storedResults,varyParams=None,doPriors=True,doSpecResids=True,usesns=None):
		if varyParams is None:
			varyParams=np.zeros(self.npar,dtype=bool)
		self.fillStoredResults(guess,storedResults)
		if varyParams[self.imodelerr].any() or varyParams[self.imodelcorr].any() or varyParams[self.iclscat].any():
			raise ValueError('lsqwrap not allowed to handle varying model uncertainties')

		residuals = []
		jacobian = [] # Jacobian matrix from r
		for sn in self.datadict.keys() if usesns is None else usesns:
			photresidsdict,specresidsdict=self.ResidsForSN(guess,sn,storedResults,varyParams,fixUncertainty=True)
			for residsdict in ([photresidsdict,specresidsdict] if doSpecResids else [photresidsdict]):
				residuals+=[residsdict[k]['resid'] for k in residsdict]
				jacobian+=[residsdict[k]['resid_jacobian'] for k in residsdict]

		if doPriors:
			priorResids,priorVals,priorJac=self.priors.priorResids(self.usePriors,self.priorWidths,guess)
			residuals+=[priorResids]
			jacobian+=[priorJac]
			BoundedPriorResids,BoundedPriorVals,BoundedPriorJac = \
				self.priors.BoundedPriorResids(self.bounds,self.boundedParams,guess)
			residuals+=[BoundedPriorResids]
			jacobian+=[sparse.csr_matrix(BoundedPriorJac)]

		if self.regularize:
			for regularization, weight,regKey in [(self.phaseGradientRegularization, self.regulargradientphase,'regresult_phase'),
										   (self.waveGradientRegularization,self.regulargradientwave,'regresult_wave' ),
										   (self.dyadicRegularization,self.regulardyad,'regresult_dyad')]:
				if weight ==0:
					continue
				if regKey in storedResults and not (varyParams[self.im0].any() or varyParams[self.im1].any()):
					residuals += storedResults[regKey]
					jacobian +=	 [sparse.csr_matrix((r.size,self.parlist.size)) for r in storedResults[regKey]]
				else:
					for regResids,regJac,relativeweight in zip(
							*regularization(guess,storedResults,varyParams),[1,self.m1regularization]):
						residuals += [regResids*np.sqrt(weight*relativeweight)]
						if np.isnan(residuals[-1]).any(): import pdb;pdb.set_trace()
						jacobian+=[sparse.csr_matrix(regJac)*np.sqrt(weight*relativeweight)]
					storedResults[regKey]=residuals[-self.n_components:]

		if varyParams.any():
			return np.concatenate(residuals),sparse.vstack(jacobian)
		else:
			return	np.concatenate(residuals)

	def maxlikefit(
			self,x,storedResults=None,varyParams=None,pool=None,debug=False,
			SpecErrScale=1.0,fixFluxes=False,dospec=True,usesns=None):
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
		
		if storedResults is None: storedResults={}
		if varyParams is None:
			varyParams=np.zeros(self.npar,dtype=bool)
		computeDerivatives=np.any(varyParams)
		#Set up SALT model
		# HACK
		self.fillStoredResults(x,storedResults)	
		# timing stuff
		
		chi2 = 0
		#Construct arguments for maxlikeforSN method
		#If worker pool available, use it to calculate chi2 for each SN; otherwise, do it in this process
		args=[(x,sn,storedResults,varyParams,debug,SpecErrScale,fixFluxes,dospec) \
			  for sn in (self.datadict.keys() if usesns is None else usesns)]
		#args2 = (x,components,componentderivs,salterr,saltcorr,colorLaw,debug,timeit,computeDerivatives,computePCDerivs,SpecErrScale)
		mapFun=pool.map if pool else starmap

		#result = list(pyParz.foreach(self.datadict.keys(),self.loglikeforSN,args2))
		result=list(mapFun(self.loglikeforSN,args))

		# hack!
		loglike=sum(result)
		logp = loglike

		if computeDerivatives:
			loglike=sum([r[0] for r in result])
			grad=sum([r[1] for r in result])
		else:
			loglike=sum(result)
		logp = loglike
		if len(self.usePriors):
			priorResids,priorVals,priorJac=self.priors.priorResids(self.usePriors,self.priorWidths,x)	
			logp -=(priorResids**2).sum()/2
			if computeDerivatives:
				grad-= (priorResids [:,np.newaxis] * priorJac).sum(axis=0)
			BoundedPriorResids,BoundedPriorVals,BoundedPriorJac = \
			self.priors.BoundedPriorResids(self.bounds,self.boundedParams,x)
			logp -=(BoundedPriorResids**2).sum()/2
			if computeDerivatives:
				grad-= (BoundedPriorResids [:,np.newaxis] * BoundedPriorJac).sum(axis=0)


		if self.regularize:
			for regularization, weight in [
					(self.phaseGradientRegularization, self.regulargradientphase),
					(self.waveGradientRegularization,self.regulargradientwave ),
					(self.dyadicRegularization,self.regulardyad)]:
				if weight ==0:
					continue
				regResids,regJac=regularization(x,storedResults,varyParams)
				logp-= sum([(res**2).sum()*weight*componentweight/2 \
							for res,componentweight in zip(regResids,[1,self.m1regularization])])
				if computeDerivatives:
					for res,jac,componentweight in zip(regResids,regJac,[1,self.m1regularization]):
						grad -= (res[:,np.newaxis]*jac *weight*componentweight).sum(axis=0)
		self.nstep += 1

		if computeDerivatives:
			return logp,grad
		else:
			return logp
					
	def ResidsForSN(self,x,sn,storedResults,varyParams=None,fixUncertainty=False,SpecErrScale=1.,fixFluxes=False):
		""" This method should be the only one required for any fitter to process the supernova data. 
		Find the residuals of a set of parameters to the photometric and spectroscopic data of a given supernova. 
		Photometric residuals are first decorrelated to diagonalize color scatter"""

		if varyParams is None:
			varyParams=np.zeros(self.npar,dtype=bool)
		self.fillStoredResults(x,storedResults)	
		photmodel,specmodel=self.modelvalsforSN(x,sn,storedResults,varyParams)

		photresids={}
		for flt in photmodel:
			filtmodel=photmodel[flt]
			filtresids={}
			photresids[flt]=filtresids
			choleskykey=f'photCholesky_{sn}_{flt}'
			if not choleskykey in storedResults:
				#Calculate cholesky matrix for each set of photometric measurements in each filter
				if (filtmodel['modelvariance']<0).any():
					warnings.warn('Negative variance in photometry',RuntimeWarning)
					negVals=filtmodel['modelvariance']<0
					filtmodel['modelvariance'][negVals]=0
				variance=filtmodel['fluxvariance']+filtmodel['modelvariance']
				clscat,dclscatdx=filtmodel['colorvariance']
				#Find cholesky matrix as sqrt of diagonal uncertainties, then perform rank one update to incorporate color scatter
				L = rankOneCholesky(variance,clscat**2,filtmodel['modelflux'])
				invL=sparse.csr_matrix(linalg.solve_triangular(L,np.diag(np.ones(L.shape[0])),lower=True))
				
				storedResults[choleskykey]=L,invL,clscat,dclscatdx
			else:
				L,invL,clscat,dclscatdx=storedResults[choleskykey]
				
			if not fixUncertainty: 
				filtresids['lognorm']=0
				filtresids['lognorm_grad']=np.zeros(self.parlist.size)
		
			#More stable to solve by backsubstitution than to invert and multiply
			fluxdiff=filtmodel['modelflux']-filtmodel['dataflux']
			filtresids['resid']=linalg.solve_triangular(L, fluxdiff,lower=True)

			if not fixUncertainty:
				filtresids['lognorm']-= (np.log(np.diag(L)).sum())
			
			if varyParams.any():
				if not fixFluxes: 
					filtresids['resid_jacobian']=invL.dot(filtmodel['modelflux_jacobian'])
					#filtresids['resid_jacobian']=linalg.solve_triangular(L,filtmodel['modelflux_jacobian'],lower=True)
				else: 
					filtresids['resid_jacobian']=sparse.csr_matrix(filtmodel['modelflux_jacobian'].shape)
				if not fixUncertainty:
					raise NotImplementedError('Derivative of variance is still a work in progress')
					#Cut out zeroed jacobian entries to save time
					nonzero=(~((filtmodel['modelvariance_jacobian']==0) & \
							   (filtmodel['modelflux_jacobian']==0 if not fixFluxes else True)).all(axis=0)) | \
							   (self.parlist=='clscat')
					reducedJac=filtmodel['modelvariance_jacobian'][:,nonzero]
					#import pdb;pdb.set_trace()
					#Calculate L^-1 (necessary for the diagonal derivative)
					invL=linalg.solve_triangular(L,np.diag(np.ones(fluxdiff.size)),lower=True)
			
					# Calculate the fractional derivative of L w.r.t model parameters
					# L^-1 dL/dx = {L^-1 x d Sigma / dx x L^-T, with the upper triangular part zeroed and the diagonal halved}
			
					#First calculating diagonal part
					fractionalLDeriv=np.dot(invL,np.swapaxes(reducedJac[:,np.newaxis,:]* invL.T[:,:,np.newaxis],0,1))
		
					fluxPrime= linalg.solve_triangular(L,filtmodel['modelflux'])
					fractionalLDeriv[:,:,self.iclscat]+= 2*clscat*dclscat[np.newaxis,np.newaxis,:]* \
														 np.outer(fluxPrime,fluxPrime)[:,:,np.newaxis]
					if not fixFluxes:
						jacPrime = linalg.solve_triangular(L,filtmodel['modelflux_jacobian'][:,nonzero])
						#Derivative w.r.t  color scatter 
						#Derivative w.r.t model flux
						dfluxSquared = jacPrime[:,np.newaxis,:]*fluxPrime[np.newaxis,:,np.newaxis]
						#Symmetrize
						dfluxSquared = np.swapaxes(dfluxSquared,0,1)+dfluxSquared
						fractionalLDeriv +=clscat**2 * dfluxSquared
				
					fractionalLDeriv[np.triu(np.ones((fluxdiff.size,fluxdiff.size),dtype=bool),1),:]=0
					fractionalLDeriv[np.diag(np.ones((fluxdiff.size),dtype=bool)),:]/=2
			
					#Multiply by size of transformed residuals and apply to resiudal jacobian
				
					filtresids['resid_jacobian'][:,nonzero]-=np.dot(np.swapaxes(fractionalLDeriv,1,2),filtresids['resid'])
				
					#Trace of fractional derivative gives the gradient of the lognorm term, since determinant is product of diagonal
					filtresids['lognorm_grad'][nonzero]-= np.trace(fractionalLDeriv)
			else: 
				filtresids['resid_jacobian']=sparse.csr_matrix(filtmodel['modelflux_jacobian'].shape)

		#Handle spectra
		specresids={}
		for k in specmodel:
			spectralmodel=specmodel[k]
			variance=spectralmodel['fluxvariance'] + spectralmodel['modelvariance']
			if (spectralmodel['modelvariance']<0).any():
				warnings.warn('Negative variance in spectra',RuntimeWarning)
				negVals=spectralmodel['modelvariance']<0
				spectralmodel['modelvariance'][negVals]=0

			uncertainty=np.sqrt(variance)*SpecErrScale

			spectralSuppression=np.sqrt(self.num_phot/self.num_spec)*self.spec_chi2_scaling

			spectralresids={'resid': spectralSuppression * (spectralmodel['modelflux']-spectralmodel['dataflux'])/uncertainty}
			specresids[k]=spectralresids
			if not fixUncertainty:
				spectralresids['lognorm']=-np.log((np.sqrt(2*np.pi)*uncertainty)).sum()
		
			#Create a diagonal matrix with the inverse uncertainty on the diagonal
			if not fixFluxes: spectralresids['resid_jacobian']= \
			   sparse.diags([spectralSuppression/uncertainty],offsets=[0])*spectralmodel['modelflux_jacobian']
			else: spectralresids['resid_jacobian']=sparse.csr_matrix((spectralmodel['modelflux'].size,self.parlist.size))

			if not fixUncertainty:
				#Calculate jacobian of total spectral uncertainty including reported uncertainties
				uncertainty_jac=  spectralmodel['modelvariance_jacobian'][:,varyParams] / (2*uncertainty[:,np.newaxis])
				spectralresids['lognorm_grad']= - (uncertainty_jac/uncertainty[:,np.newaxis]).sum(axis=0)
				spectralresids['resid_jacobian'][:,varyParams] -= uncertainty_jac*(spectralresids['resid'] /uncertainty)[:,np.newaxis]

		tstart = time.time()
		return photresids,specresids
	
	def bestfitsinglebandnormalizationsforSN(self,x,sn,storedResults):
		photmodel,specmodel=self.modelvalsforSN(x,sn,storedResults,np.zeros(self.parlist.size,dtype=bool))
		results={}
		for flt in photmodel:
			fluxes=photmodel[flt]
			designmatrix=photmodel[flt]['modelflux']
			vals=photmodel[flt]['dataflux']
			variance=photmodel[flt]['fluxvariance']+photmodel[flt]['modelvariance']
			normvariance=1/(designmatrix**2/variance).sum()
			normalization=(designmatrix*vals/variance).sum()*normvariance
			results[flt]=normalization,normvariance
		return results
		
	def modelvalsforSN(self,x,sn,storedResults,varyParams):		
		
		temporaryResults={}
		self.fillStoredResults(x,storedResults)

		fluxphotkey='phot'+'fluxes_{}'.format(sn)
		varphotkey= 'phot'+'variances_{}'.format(sn)
		fluxspeckey='spec'+'fluxes_{}'.format(sn)
		varspeckey= 'spec'+'variances_{}'.format(sn)
		z = self.datadict[sn].zHelio
		sndata=self.datadict[sn]
		x1Deriv= varyParams[sndata.ix1] 
		obsphase = sndata.obsphase #self.phase*(1+z)
		x1,c,tpkoff = x[sndata.ix1],\
						 x[sndata.ic],x[sndata.itpkoff]
		colorexp= 10. ** (storedResults['colorLaw'] * c)
		temporaryResults['colorexp'] =colorexp

		calculateFluxes= not ('photfluxes_{}'.format(sn) in storedResults  and	'specfluxes_{}'.format(sn) in storedResults)
		calculateVariances= not ('photvariances_{}'.format(sn) in storedResults and	 'specvariances_{}'.format(sn) in storedResults)
		if calculateFluxes and not fluxphotkey in storedResults and not fluxspeckey in storedResults:
			tpkDerivs=varyParams[sndata.itpkoff]
			
			#Apply MW extinction
			M0,M1 = storedResults['components']
					
			prefactor=_SCALE_FACTOR/(1+z)*(sndata.mwextcurve[np.newaxis,:])
			mod = prefactor*(M0 + x1*M1)
			#temporaryResults['fluxInterp'] = interp1d(obsphase,mod,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
			temporaryResults['fluxInterp'] = mod #interp1d(obsphase,mod,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
			if x1Deriv:
				int1dM1 = interp1d(obsphase,prefactor*M1,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
				temporaryResults['M1Interp']=int1dM1
				
			if tpkDerivs:
				M0phasederiv,M1phasederiv = storedResults['componentderivs']
				phaseDeriv=prefactor*(M0phasederiv +x1*M1phasederiv)
				temporaryResults['phaseDerivInterp'] = interp1d(
					obsphase,phaseDeriv,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
				
		if calculateVariances and not varphotkey in storedResults and not varspeckey in storedResults:
			prefactor=(sndata.mwextcurve *	_SCALE_FACTOR/(1+z))
			if x1Deriv or varyParams[self.imodelerr].any() or varyParams[self.imodelcorr].any():
				interr1d = [interp1d(
					obsphase,err * prefactor ,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True) \
							for err in storedResults['saltErr']]
				intcorr1d= [interp1d(
					obsphase,corr ,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True) \
							for corr in storedResults['saltCorr'] ]
				temporaryResults['uncertaintyComponentsInterp']=interr1d
				temporaryResults['uncertaintyCorrelationsInterp']=intcorr1d

			saltErr=storedResults['saltErr']
			saltCorr=storedResults['saltCorr'] 
			modelUncertainty= prefactor**2 *(saltErr[0]**2	+ 2*x1* saltCorr[0]*saltErr[0]*saltErr[1] + x1**2 *saltErr[1]**2)

			#interr1d=interp1d(obsphase,modelUncertainty ,axis=0,kind=self.interpMethod,bounds_error=True,assume_sorted=True)
			#temporaryResults['modelUncertaintyInterp']=interr1d
			temporaryResults['modelUncertainty']=modelUncertainty

		returndicts=[]
		for valfun,uncertaintyfun,name in \
			[(self.photValsForSN,self.photVarianceForSN,'phot'),
			 (self.specValsForSN,self.specVarianceForSN, 'spec')]:

			fluxkey=name+'fluxes_{}'.format(sn)
			if not fluxkey in storedResults:
				storedResults[fluxkey]=valfun(x,sn, storedResults,temporaryResults,varyParams)
			valdict=storedResults[fluxkey]
			varkey=name+'variances_{}'.format(sn)	
			if not varkey in storedResults:
				storedResults[varkey]=uncertaintyfun(x,sn,storedResults,temporaryResults,varyParams)

			uncertaintydict=storedResults[varkey]
			returndicts+=[{key:{**valdict[key],**uncertaintydict[key]} for key in valdict}]
		
		return returndicts		
	
	def photValsForSN(self,x,sn,storedResults,temporaryResults,varyParams):
		sndata= self.datadict[sn]
		z	  =	   sndata.zHelio
		survey	 = sndata.survey
		dwave =	   sndata.dwave
		obswave =  sndata.obswave #self.wave*(1+z)
		obsphase = sndata.obsphase #self.phase*(1+z)
		wavedelt = obswave[1]-obswave[0]
		phasedelt = obsphase[1]-obsphase[0]
		x0,x1,c,tpkoff = x[sndata.ix0],x[sndata.ix1],\
						 x[sndata.ic],x[sndata.itpkoff]
		
		x0Deriv= varyParams[sndata.ix0]
		x1Deriv= varyParams[sndata.ix1]
		tpkDeriv=varyParams[sndata.itpkoff]
		cDeriv=	 varyParams[sndata.ic]
		requiredPCDerivs=varyParams[self.im0]|varyParams[self.im1]
		
		
		
		colorlaw,colorexp=storedResults['colorLaw'],temporaryResults['colorexp']
		reddenedsalt2flux=colorexp*temporaryResults['fluxInterp']
		photresultsdict={}
		for flt in sndata.filt:
			lcdata = sndata.photdata[flt]
			idx =  lcdata.idx		
			pbspl = lcdata.pbspl
			fluxfactor=self.fluxfactor[sndata.survey][flt]
			filtresultsdict={}
			photresultsdict[flt]=filtresultsdict
			filtresultsdict['dataflux'] = lcdata.fluxcal
			filtresultsdict['modelflux_jacobian'] = sparse.csr_matrix((filtresultsdict['dataflux'].size,self.parlist.size))

			pcderivkey=f'pcDeriv_phot_{sn}_{flt}'
			if requiredPCDerivs.all() and not pcderivkey in storedResults:
				summationCache=np.zeros((filtresultsdict['dataflux'].size,self.im0.size))

			intmult = dwave*fluxfactor*_SCALE_FACTOR/(1+z)*x0
			#Select data from the appropriate filter filter
			phase=lcdata.tobs+tpkoff
			clippedPhase=np.clip(phase,obsphase.min(),obsphase.max())
			nphase = len(phase)
			#Array output indices match time along 0th axis, wavelength along 1st axis
			modulatedFlux = lcdata.pbspl*reddenedsalt2flux[np.round((clippedPhase-obsphase[0])/phasedelt).astype(int),:][:,idx]
			if ( (phase>obsphase.max())).any():
				decayFactor=10**(-0.4*self.extrapolateDecline*(phase[phase>obsphase.max()]-obsphase.max()))
				modulatedFlux[np.where(phase>obsphase.max())[0]] *= decayFactor[:,np.newaxis]
			dmodelfluxdx0 = np.sum(modulatedFlux, axis=1)*dwave*fluxfactor
			modelflux=dmodelfluxdx0*x0
			filtresultsdict['modelflux']=modelflux

			modulatedFlux*=(dwave*fluxfactor*x0)
			if	x0Deriv:
				filtresultsdict['modelflux_jacobian'][:,(sndata.ix0)] = dmodelfluxdx0[np.newaxis].transpose()
			if x1Deriv:
				int1dM1=temporaryResults['M1Interp']
				M1interp = int1dM1(clippedPhase)
				modulatedM1=lcdata.pbspl*colorexp[idx]*M1interp[:,idx]
				if ( (phase>obsphase.max())).any():
					decayFactor=10**(-0.4*self.extrapolateDecline*(phase[phase>obsphase.max()]-obsphase.max()))
					modulatedM1[np.where(phase>obsphase.max())[0]] *= decayFactor[:,np.newaxis]
				filtresultsdict['modelflux_jacobian'][:,(sndata.ix1)] = \
					(np.sum(modulatedM1, axis=1)*(dwave*fluxfactor*x0))[np.newaxis].transpose()
				
			if tpkDeriv:
				#Need to figure out how to handle derivatives wrt time when dealing with nearest neighbor interpolation; maybe require linear?
				modulatedPhaseDeriv= lcdata.pbspl*temporaryResults['phaseDerivInterp'](clippedPhase)[:,idx]
				filtresultsdict['modelflux_jacobian'][:,sndata.itpkoff] = \
					(np.sum(modulatedPhaseDeriv, axis=1)*(dwave*fluxfactor*x0) ) #[:,np.newaxis]
				for p in np.where(phase>obsphase.max())[0]:
					filtresultsdict['modelflux_jacobian'][p,sndata.itpkoff]=-0.4*np.log(10)*self.extrapolateDecline*modelflux[p]
			
			if cDeriv:
				#d model / dc is total flux (M0 and M1 components (already modulated with passband)) times the color law and a factor of ln(10)
				filtresultsdict['modelflux_jacobian'][:,(sndata.ic)]=\
					np.sum((modulatedFlux)*np.log(10)*colorlaw[np.newaxis,idx], axis=1)[np.newaxis].transpose()
				
			#Multiply M0 and M1 components (already modulated with passband) by c* d colorlaw / d cl_i, with associated normalizations
			if varyParams[self.iCL].any():
				filtresultsdict['modelflux_jacobian'][:,self.iCL]= \
					(np.sum((modulatedFlux)[:,:,np.newaxis]*\
					self.colorLawDeriv[:,varyParams[self.iCL]][np.newaxis,idx,:], axis=1))*-0.4*np.log(10)*c	
				
			if requiredPCDerivs.any():
					passbandColorExp=lcdata.pbspl*colorexp[idx]*sndata.mwextcurve[idx]
					for p in range(filtresultsdict['modelflux'].size):
						if self.fitTpkOff:
							derivInterp = self.spline_deriv_interp(
								(clippedPhase[p]/(1+z),self.wave[idx]),
								method=self.interpMethod)[:,requiredPCDerivs]
							summation = np.sum( passbandColorExp.T * derivInterp, axis=0)
							if phase[p]>obsphase.max():
								decayFactor= 10**(-0.4*self.extrapolateDecline*(phase[p]-obsphase.max()))
								summation*=decayFactor
						else:
							derivInterp=lcdata.pcderivsparse[p][:,requiredPCDerivs]
							summation=derivInterp.T.dot(passbandColorExp[0])

						filtresultsdict['modelflux_jacobian'][p,self.im0[varyParams[self.im0]]]=\
							summation[varyParams[self.im0]]*intmult
						filtresultsdict['modelflux_jacobian'][p,self.im1[varyParams[self.im1]]]=\
							summation[varyParams[self.im1]]*intmult*x1
					
			filtresultsdict['modelflux_jacobian']=sparse.csr_matrix(filtresultsdict['modelflux_jacobian'])
			if np.isnan(filtresultsdict['modelflux']).any() or np.isinf(filtresultsdict['modelflux']).any():
				raise RuntimeError(f'phot model fluxes are nonsensical for SN {sn}')
		return photresultsdict


	def specValsForSN(self,x,sn,storedResults,temporaryResults,varyParams):

		sndata=self.datadict[sn]
		z =		   sndata.zHelio
		survey =   sndata.survey

		obswave =  sndata.obswave #self.wave*(1+z)
		obsphase = sndata.obsphase #self.phase*(1+z)
		wavedelt = obswave[1]-obswave[0]
		phasedelt = obsphase[1]-obsphase[0]
		x1,c,tpkoff = x[sndata.ix1],\
						 x[sndata.ic],x[sndata.itpkoff]

		x1Deriv= varyParams[sndata.ix1]
		tpkDeriv=varyParams[sndata.itpkoff]
		cDeriv=	 varyParams[sndata.ic]
		requiredPCDerivs=varyParams[self.im0]|varyParams[self.im1]
		
		resultsdict={}

		for k,spectrum in sndata.specdata.items():
			specresultsdict={}
			resultsdict[k]=specresultsdict

			specresultsdict['dataflux'] = spectrum.flux
			if (requiredPCDerivs).any():
				specresultsdict['modelflux_jacobian'] = np.zeros((specresultsdict['dataflux'].size,self.parlist.size))
			else:
				specresultsdict['modelflux_jacobian'] = sparse.csr_matrix((specresultsdict['dataflux'].size,self.parlist.size))
			#colorlawinterp=storedResults['colorLawInterp'](spectrum.wavelength/(1+z))
			#colorexpinterp=10**(c*colorlawinterp)
		


			x0=x[spectrum.ispecx0]
			x0Deriv=varyParams[spectrum.ispecx0]
			
			phase=spectrum.tobs+tpkoff

			#Define recalibration factor
			coeffs=x[spectrum.ispcrcl]
			pow=coeffs.size-np.arange(coeffs.size)
			recalCoord=(spectrum.wavelength-np.mean(spectrum.wavelength))/self.specrange_wavescale_specrecal
			drecaltermdrecal=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
			recalterm=(drecaltermdrecal*coeffs[np.newaxis,:]).sum(axis=1)
			pastbounds=np.abs(recalterm)>100
			recalterm=np.clip(recalterm,-100,100)
			drecaltermdrecal[pastbounds,:]=0
			recalexp=np.exp(recalterm)
			
			if len(np.where((recalexp != recalexp) | (recalexp == np.inf))[0]):
				raise RuntimeError(f'error....spec recalibration problem for SN {sn}!')

			dmodelfluxdx0 = recalexp*temporaryResults['fluxInterp'][int(np.round((phase-obsphase[0])/phasedelt)),
																(np.round((spectrum.wavelength-obswave[0])/wavedelt)).astype(int)]
			
			modulatedFlux = dmodelfluxdx0*x0
			specresultsdict['modelflux']  = modulatedFlux
			if len(np.where((specresultsdict['modelflux'] != specresultsdict['modelflux']) |
							(specresultsdict['modelflux'] == np.inf))[0]):
				raise RuntimeError(f'spec model flux nonsensical for SN {sn}')
			
			if x0Deriv:
				specresultsdict['modelflux_jacobian'][:,spectrum.ispecx0] = dmodelfluxdx0[:,np.newaxis]
			
			if x1Deriv:
				M1interp = temporaryResults['M1Interp'](phase)
				M1int = interp1d(obswave,M1interp,kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
				M1interp = M1int(spectrum.wavelength)*recalexp
				if (requiredPCDerivs).any():
					specresultsdict['modelflux_jacobian'][:,(sndata.ix1)] = x0*M1interp[np.newaxis]
				else:
					specresultsdict['modelflux_jacobian'][:,(sndata.ix1)] = x0*M1interp[np.newaxis].transpose()
				
				
			if tpkDeriv:
				phaseDerivInterp = temporaryResults['phaseDerivInterp'](phase)
				phaseDerivInterp = interp1d(obswave,phaseDerivInterp[0],kind=\
					self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)(spectrum.wavelength)
				modPhaseDeriv = phaseDerivInterp*recalexp*x0
				specresultsdict['modelflux_jacobian'][:,self.parlist == 'tpkoff_{}'.format(sn)] = modPhaseDeriv[:,np.newaxis]

			if self.specrecal:
				varySpecRecal=varyParams[self.parlist == 'specrecal_{}_{}'.format(sndata.snid,k)]
				if varySpecRecal.sum():
					specresultsdict['modelflux_jacobian'][:,(self.parlist == 'specrecal_{}_{}'.format(sndata.snid,k))]	= modulatedFlux[:,np.newaxis] * drecaltermdrecal[:,varySpecRecal]
			# M0, M1
			if (requiredPCDerivs).any():
				intmult = _SCALE_FACTOR/(1+z)*recalexp*sndata.mwextcurveint(spectrum.wavelength)
				if self.fitTpkOff:
					derivInterp=self.spline_deriv_interp(
						(phase[0]/(1+z),spectrum.wavelength/(1+z)),
						method=self.interpMethod)*intmult[:,np.newaxis]
					specresultsdict['modelflux_jacobian'][:,(self.parlist=='m0')]  = derivInterp[:,varyParams[self.im0]]*(x0)
					specresultsdict['modelflux_jacobian'][:,(self.parlist=='m1')] =	 derivInterp[:,varyParams[self.im1]]*(x1*x0)
				else:
					derivInterp=spectrum.pcderivsparse.multiply(intmult[:,np.newaxis]).tocsc()
					specresultsdict['modelflux_jacobian'][:,(self.parlist=='m0')]  = \
						(derivInterp[:,varyParams[self.im0]]*(x0[0])).toarray()
					specresultsdict['modelflux_jacobian'][:,(self.parlist=='m1')] = \
						( derivInterp[:,varyParams[self.im1]]*(x1*x0[0])).toarray()

			if (requiredPCDerivs).any():
				resultsdict[k]['modelflux_jacobian']=sparse.csr_matrix(resultsdict[k]['modelflux_jacobian'])
		
		return resultsdict

	def specVarianceForSN(self,x,sn,storedResults,temporaryResults,varyParams):
		sndata=self.datadict[sn]
		z =		   sndata.zHelio
		survey =   sndata.survey
		dwave =	   sndata.dwave
		obswave =  sndata.obswave #self.wave*(1+z)
		obsphase = sndata.obsphase #self.phase*(1+z)
		wavedelt = obswave[1]-obswave[0]
		phasedelt = obsphase[1]-obsphase[0]
		x1,c,tpkoff = x[sndata.ix1],\
						 x[sndata.ic],x[sndata.itpkoff]

		x1Deriv= varyParams[sndata.ix1]
		tpkDeriv=varyParams[sndata.itpkoff]
		cDeriv=	 varyParams[sndata.ic]
		
		
		resultsdict={}
		for k,spectrum in sndata.specdata.items():
			x0=x[spectrum.ispecx0]
			x0Deriv=varyParams[spectrum.ispecx0][0]
						
			specresultsdict={}
			resultsdict[k]=specresultsdict
			specresultsdict['fluxvariance'] =  spectrum.fluxerr**2
			specresultsdict['modelvariance_jacobian']=np.zeros([len(spectrum),self.parlist.size]) #sparse.csr_matrix([len(spectrum),self.parlist.size])

			phase=spectrum.tobs+tpkoff*(1+z)
			
			#Define recalibration factor
			if self.specrecal:
				coeffs=x[spectrum.ispcrcl]
				pow=coeffs.size-np.arange(coeffs.size)
				recalCoord=(spectrum.wavelength-np.mean(spectrum.wavelength))/self.specrange_wavescale_specrecal
				drecaltermdrecal=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
				recalexp=np.exp((drecaltermdrecal*coeffs[np.newaxis,:]).sum(axis=1))
			else:
				recalexp=1
				
			if	x1Deriv or varyParams[self.imodelerr].any() or varyParams[self.imodelcorr].any():
				modelErrInt = [ interp1d(
					obswave, interr(phase),kind=self.interpMethod,bounds_error=False,
					fill_value=0,assume_sorted=True)  for interr in temporaryResults['uncertaintyComponentsInterp']]
				modelCorrInt= [ interp1d(
					obswave, intcorr(phase),kind=self.interpMethod,bounds_error=False,
					fill_value=0,assume_sorted=True) for intcorr in temporaryResults['uncertaintyCorrelationsInterp']]
				modelerrnox = [	 interr( spectrum.wavelength) *recalexp for interr in (modelErrInt)]
				corr=  [intcorr(spectrum.wavelength) for intcorr in modelCorrInt]
				
			uncertaintyNoX0 = recalexp**2. * temporaryResults['modelUncertainty'][
				int(np.round((phase-obsphase[0])/phasedelt)),
				(np.round((spectrum.wavelength-obswave[0])/wavedelt)).astype(int)]

			modelUncertainty=uncertaintyNoX0*x0**2

			specresultsdict['modelvariance'] = modelUncertainty

			if x0Deriv:
				specresultsdict['modelvariance_jacobian'][:,spectrum.ispecx0] = (uncertaintyNoX0[:,np.newaxis]*2*x0)[np.newaxis].transpose()
			if x1Deriv:
				specresultsdict['modelvariance_jacobian'][:,(sndata.ix1)] = \
					x0**2 * 2*(modelerrnox[0]*modelerrnox[1]*corr[0]+ x1* modelerrnox[1]**2) #[:,np.newaxis]

			if self.specrecal : 
				varySpecRecal=varyParams[spectrum.ispcrcl]
				specresultsdict['modelvariance_jacobian'][:,spectrum.ispcrcl[varySpecRecal]]  =	 \
					modelUncertainty[:,np.newaxis] * drecaltermdrecal[:,varySpecRecal] * 2

			if varyParams[self.imodelerr].any() or varyParams[self.imodelcorr].any():
				interpresult=  self.errorspline_deriv_interp((phase[0]/(1+z),spectrum.wavelength/(1+z)),method=self.interpMethod) 
				extinctionexp=(recalexp* _SCALE_FACTOR/(1+z)*sndata.mwextcurveint(spectrum.wavelength))
				specresultsdict['modelvariance_jacobian'][:,(self.parlist=='modelerr_0'	 )]	  = \
					2* x0**2  * (extinctionexp *( modelerrnox[0] + \
					corr[0]*modelerrnox[1]*x1))[:,np.newaxis] * interpresult[:,varyParams[self.parlist=='modelerr_0']]
				specresultsdict['modelvariance_jacobian'][:,(self.parlist=='modelerr_1'	 )]	  = \
					2* x0**2  * (extinctionexp *(modelerrnox[1]*x1**2 + \
					corr[0]*modelerrnox[0]*x1))[:,np.newaxis] * interpresult[:,varyParams[self.parlist=='modelerr_1']]
				specresultsdict['modelvariance_jacobian'][:,(self.parlist=='modelcorr_01')] = \
					2* x0**2  * (modelerrnox[1]*modelerrnox[0]*x1)[:,np.newaxis]  * \
					interpresult[:,varyParams[self.parlist=='modelcorr_01']]

		#import pdb; pdb.set_trace()
		return resultsdict

	def photVarianceForSN(self,x,sn,storedResults,temporaryResults,varyParams):
		"""Currently calculated only at the effective wavelength of the filter, not integrated over."""
		sndata= self.datadict[sn]
		z	  =	   sndata.zHelio
		survey	 = sndata.survey
		
		dwave =	   sndata.dwave
		obswave =  sndata.obswave #self.wave*(1+z)
		obsphase = sndata.obsphase #self.phase*(1+z)
		wavedelt = obswave[1]-obswave[0]
		phasedelt = obsphase[1]-obsphase[0]
		x0,x1,c,tpkoff = x[self.parlist == f'x0_{sn}'],x[sndata.ix1],\
						 x[sndata.ic],x[sndata.itpkoff]
		
		x0Deriv= varyParams[sndata.ix0]
		x1Deriv= varyParams[sndata.ix1]
		tpkDeriv=varyParams[sndata.itpkoff]
		cDeriv=	 varyParams[sndata.ic]
		
		photresultsdict={}

		for flt,lcdata in sndata.photdata.items():
			filtresultsdict={}
			photresultsdict[flt]=filtresultsdict

			#Select data from the appropriate filter filter
			phase=lcdata.tobs+tpkoff*(1+z)
			clippedPhase=np.clip(phase,obsphase.min(),obsphase.max())
			nphase = len(phase)
			
			#Calculate color scatter
			if self.iclscat.size:
				colorscat,dcolorscatdx=self.colorscatter(x,lcdata.lambdaeff/(1+z),varyParams)
				if colorscat == np.inf:
					log.error('infinite color scatter!')
					import pdb; pdb.set_trace()
			else:
				colorscat=0
				dcolorscatdx=np.array([])

			filtresultsdict['colorvariance']= colorscat,dcolorscatdx

			colorlawinterp=storedResults['colorLawInterp'](lcdata.lambdaeff/(1+z))
			colorexpinterp=10**(c*colorlawinterp)
			
			fluxfactor=(colorexpinterp*self.fluxfactor[survey][flt]*(lcdata.pbspl.sum())*dwave)**2
			
			if	x1Deriv or varyParams[self.imodelerr].any() or varyParams[self.imodelcorr].any():
				modelErrInt = [ interp1d(
					obswave, interr(clippedPhase) ,kind=self.interpMethod,bounds_error=False,
					fill_value=0,assume_sorted=True)  for interr in temporaryResults['uncertaintyComponentsInterp']]
				modelCorrInt= [ interp1d(
					obswave, intcorr(clippedPhase),kind=self.interpMethod,bounds_error=False,
					fill_value=0,assume_sorted=True) for intcorr in temporaryResults['uncertaintyCorrelationsInterp']]
				
				modelerrnox = [	 interr( lcdata.lambdaeff)	for interr in (modelErrInt)]
				corr=  [intcorr(lcdata.lambdaeff) for intcorr in modelCorrInt]
			
			#modelErrInt = interp1d( obswave, temporaryResults['modelUncertaintyInterp'](clippedPhase),kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)(lameff)
			modelErrInt =  temporaryResults['modelUncertainty'][np.round((clippedPhase-obsphase[0])/phasedelt).astype(int),
																int(np.round((lcdata.lambdaeff-obswave[0])/wavedelt))]
			uncertaintyNoX0=  fluxfactor *	modelErrInt
			modelUncertainty=uncertaintyNoX0*x0**2
			negativevariance=modelUncertainty<0
			modelUncertainty[negativevariance]=0
			filtresultsdict['fluxvariance'] = lcdata.fluxcalerr**2
			filtresultsdict['modelvariance']=  modelUncertainty
			filtresultsdict['modelvariance_jacobian']=sparse.csr_matrix((filtresultsdict['fluxvariance'].size,self.parlist.size))

			if x0Deriv:
				filtresultsdict['modelvariance_jacobian'][:,(sndata.ix0)] = (uncertaintyNoX0*2*x0)[np.newaxis].transpose()
			if x1Deriv:
				filtresultsdict['modelvariance_jacobian'][:,(sndata.ix1)] = \
					(x0**2 *fluxfactor * 2*(modelerrnox[0]*modelerrnox[1]*corr[0]+ x1* modelerrnox[1]**2))[np.newaxis].transpose()

			if cDeriv:
				filtresultsdict['modelvariance_jacobian'][:,(sndata.ic)] = \
					(modelUncertainty * 2 *np.log(10)*colorlawinterp)[np.newaxis].transpose()

			for i,varIndex in enumerate(np.where(self.parlist=='cl')[0]):
				if varyParams[self.iCL][i]:
					filtresultsdict['modelvariance_jacobian'][:,varIndex] =	 \
						(2* (-0.4 *np.log(10)*c)*modelUncertainty*self.colorLawDerivInterp(lcdata.lambdaeff/(1+z))[varyParams[self.iCL]][i])[np.newaxis].transpose()

			if varyParams[self.imodelerr].any() or varyParams[self.imodelcorr].any():
				extinctionexp=(colorexpinterp* _SCALE_FACTOR/(1+z)*sndata.mwextcurveint(lcdata.lambdaeff))
			
				interpresult =	self.errorspline_deriv_interp((clippedPhase/(1+z),lcdata.lambdaeff/(1+z)),method=self.interpMethod) 
				filtresultsdict['modelvariance_jacobian'][:,(self.parlist=='modelerr_0') ]	=\
					( 2* fluxfactor* x0**2	* extinctionexp) *((( modelerrnox[0] + corr[0]*modelerrnox[1]*x1))[:,np.newaxis] * \
					interpresult[:,varyParams[self.parlist=='modelerr_0']])
				filtresultsdict['modelvariance_jacobian'][:,(self.parlist=='modelerr_1')  ]	 =\
					( 2* fluxfactor* x0**2	* extinctionexp )*(((modelerrnox[1]*x1**2 + corr[0]*modelerrnox[0]*x1))[:,np.newaxis] * \
					interpresult[:,varyParams[self.parlist=='modelerr_1']])
				filtresultsdict['modelvariance_jacobian'][:,(self.parlist=='modelcorr_01')]	 =\
					( 2*fluxfactor* x0**2 *x1) * ((modelerrnox[1]*modelerrnox[0])[:,np.newaxis]	 * \
					interpresult[:,varyParams[self.parlist=='modelcorr_01']])

		return photresultsdict

	
	def fillStoredResults(self,x,storedResults):
		if self.n_colorpars:
			if not 'colorLaw' in storedResults:
				storedResults['colorLaw'] = -0.4 * SALT2ColorLaw(self.colorwaverange, x[self.parlist == 'cl'])(self.wave)
				storedResults['colorLawInterp']= interp1d(
					self.wave,storedResults['colorLaw'],kind=self.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
		else: storedResults['colorLaw'] = 1
				
		if not 'components' in storedResults:
			storedResults['components'] =self.SALTModel(x)
		if not 'componentderivs' in storedResults:
			if self.bsorder != 0: storedResults['componentderivs'] = self.SALTModelDeriv(x,1,0,self.phase,self.wave)
		
		if 'saltErr' not in storedResults:
			storedResults['saltErr']=self.ErrModel(x)
		if 'saltCorr' not in storedResults:
			storedResults['saltCorr']=self.CorrelationModel(x)

	
	def loglikeforSN(self,x,sn,storedResults,varyParams=None,debug=False,SpecErrScale=1.0,fixFluxes=False,dospec=True):
		
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

		photResidsDict,specResidsDict = self.ResidsForSN(
			x,sn,storedResults,varyParams,fixUncertainty=False,fixFluxes=True,SpecErrScale=SpecErrScale)

		loglike=0
		grad_loglike=0
		calculategrad=(not varyParams is None) and np.any(varyParams)
		for residsdict in ([photResidsDict,specResidsDict] if dospec else [photResidsDict]):
			for key in residsdict:
				resids=residsdict[key]
				loglike+= resids['lognorm']- (resids['resid']**2).sum() / 2.  
				if calculategrad:
					grad_loglike+=resids['lognorm_grade']-resids['resid_jacobian'].T.dot(resids['resid'])

		if calculategrad:
			return loglike,grad_loglike
		else:
			return loglike
				
	def specchi2(self):

		return chi2
	
	def SALTModel(self,x,evaluatePhase=None,evaluateWave=None):
		"""Returns flux surfaces of SALT model"""
		try: m0pars = x[self.m0min:self.m0max+1]
		except: import pdb; pdb.set_trace()
		try:
			if self.bsorder != 0:
				m0 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
							 self.wave if evaluateWave is None else evaluateWave,
							 (self.phaseknotloc,self.waveknotloc,m0pars,self.bsorder,self.bsorder))
			else:
				phase = self.phase if evaluatePhase is None else evaluatePhase
				wave = self.wave if evaluateWave is None else evaluateWave
				n_repeat_phase = int(phase.size/(self.phaseknotloc.size-1))+1
				n_repeat_phase_extra = -1*(n_repeat_phase*(self.phaseknotloc.size-1) % phase.size)
				if n_repeat_phase_extra == 0: n_repeat_phase_extra = None
				n_repeat_wave = int(wave.size/(self.waveknotloc.size-1))+1
				n_repeat_wave_extra = -1*(n_repeat_wave*(self.waveknotloc.size-1) % wave.size)
				if n_repeat_wave_extra == 0: n_repeat_wave_extra = None
				m0 = np.repeat(np.repeat(m0pars.reshape([self.phaseknotloc.size-1,self.waveknotloc.size-1]),n_repeat_phase,axis=0),
							   n_repeat_wave,axis=1)[:n_repeat_phase_extra,:n_repeat_wave_extra]
		except:
			import pdb; pdb.set_trace()

		# make sure M0 is positive - (clunky)
		#m0[m0 < 0] = 0
		
		if self.n_components == 2:
			m1pars = x[self.im1]
			if self.bsorder != 0:
				m1 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
							 self.wave if evaluateWave is None else evaluateWave,
							 (self.phaseknotloc,self.waveknotloc,m1pars,self.bsorder,self.bsorder))
			else:
				m1 = np.repeat(np.repeat(m1pars.reshape([self.phaseknotloc.size-1,self.waveknotloc.size-1]),n_repeat_phase,axis=0),n_repeat_wave,axis=1)[:n_repeat_phase_extra,:n_repeat_wave_extra]

			components = (m0,m1)
		elif self.n_components == 1:
			components = (m0,)
		else:
			raise RuntimeError('A maximum of two principal components is allowed')
		#if self.bsorder == 0: import pdb; pdb.set_trace()
		return components

	def SALTModelDeriv(self,x,dx,dy,evaluatePhase=None,evaluateWave=None):
		"""Returns derivatives of flux surfaces of SALT model"""
		try: m0pars = x[self.m0min:self.m0max+1]
		except: import pdb; pdb.set_trace()
		try:
			m0 = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
						 self.wave if evaluateWave is None else evaluateWave,
						 (self.phaseknotloc,self.waveknotloc,m0pars,self.bsorder,self.bsorder),
						 dx=dx,dy=dy)
		except:
			import pdb; pdb.set_trace()
			
		if self.n_components == 2:
			m1pars = x[self.im1]
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
		phase=self.phase if evaluatePhase is None else evaluatePhase
		wave=self.wave if evaluateWave is None else evaluateWave
		for min,max in zip(self.corrmin,self.corrmax):
			try: errpars = x[min:max+1]
			except: import pdb; pdb.set_trace()
			if self.errbsorder == 0:
				binphasecenter=((self.errphaseknotloc)[1:]+(self.errphaseknotloc)[:-1])/2
				binwavecenter =((self.errwaveknotloc)[1:]+(self.errwaveknotloc)[:-1])/2
				interp=RegularGridInterpolator((binphasecenter,binwavecenter),errpars.reshape(binphasecenter.size,binwavecenter.size),'nearest',False,0)
				gridwave,gridphase=np.meshgrid(wave,phase)
				clipinterp=lambda x,y: interp((np.clip(x,binphasecenter.min(),binphasecenter.max()),np.clip(y,binwavecenter.min(),binwavecenter.max())))
				result=clipinterp(gridphase.flatten(),gridwave.flatten()).reshape((phase.size,wave.size))
				components+=[result]
			else:
				components+=[  bisplev(phase,
								   wave,
								   (self.errphaseknotloc,self.errwaveknotloc,errpars,self.errbsorder,self.errbsorder))]
		return components
	
	def colorscatter(self,x,wave,varyParams=None):
		clscatpars = x[self.parlist == 'clscat']
		pow=clscatpars.size-1-np.arange(clscatpars.size)
		coeffs=clscatpars/factorial(pow)
		clscat=np.exp(np.poly1d(coeffs)((wave-5500)/1000))
		if varyParams is None:
			return clscat
		else:
			pow=pow[varyParams[self.parlist=='clscat']]
			dcolorscatdx= clscat*(((wave-5500)/1000) ** (pow) )/ factorial(pow)
			return clscat,dcolorscatdx
		
	
	def ErrModel(self,x,evaluatePhase=None,evaluateWave=None):
		"""Returns modeled variance of SALT model components as a function of phase and wavelength"""
		phase=self.phase if evaluatePhase is None else evaluatePhase
		wave=self.wave if evaluateWave is None else evaluateWave
		components=[]
		for min,max in zip(self.errmin,self.errmax):
			try: errpars = x[min:max+1]
			except: import pdb; pdb.set_trace()
			if self.errbsorder == 0:
				binphasecenter=((self.errphaseknotloc)[1:]+(self.errphaseknotloc)[:-1])/2
				binwavecenter =((self.errwaveknotloc)[1:]+(self.errwaveknotloc)[:-1])/2
				
				interp=RegularGridInterpolator((binphasecenter,binwavecenter),errpars.reshape(binphasecenter.size,binwavecenter.size),'nearest',False,0)
				clipinterp=lambda x,y: interp((np.clip(x,binphasecenter.min(),binphasecenter.max()),np.clip(y,binwavecenter.min(),binwavecenter.max())))
				gridwave,gridphase=np.meshgrid(wave,phase)
				result=clipinterp(gridphase.flatten(),gridwave.flatten()).reshape((phase.size,wave.size))
				components+=[result]
			else:
				components+=[  bisplev(phase,
								   wave,
								   (self.errphaseknotloc,self.errwaveknotloc,errpars,self.errbsorder,self.errbsorder))]
			
		return components

	def CustomErrModel(self,m0var,m1var,m01cov,evaluatePhase=None,evaluateWave=None):
		"""Returns modeled variance of SALT model components as a function of phase and wavelength"""

				
		return components

	
	def getParsGN(self,x):

		m0pars = x[self.parlist == 'm0']
		#m0err = np.zeros(len(x[self.parlist == 'm0']))
		m1pars = x[self.parlist == 'm1']
		#m1err = np.zeros(len(x[self.parlist == 'm1']))
		
		clpars = x[self.parlist == 'cl']
		clerr = np.zeros(len(x[self.parlist == 'cl']))
		

		#clscaterr = x[self.parlist == 'clscaterr']
		clscat=self.colorscatter(x,self.wave)
		resultsdict = {}
		n_sn = len(self.datadict.keys())
		for k in self.datadict.keys():
			resultsdict[k] = {'x0':x[self.parlist == f'x0_{k}'][0],
							  'x1':x[self.parlist == f'x1_{k}'][0],# - np.mean(x[self.ix1]),
							  'c':x[self.parlist == f'c_{k}'][0],
							  'tpkoff':x[self.parlist == f'tpkoff_{k}'][0],
							  'x0err':x[self.parlist == f'x0_{k}'][0],
							  'x1err':x[self.parlist == f'x1_{k}'][0],
							  'cerr':x[self.parlist == f'c_{k}'][0],
							  'tpkofferr':x[self.parlist == f'tpkoff_{k}'][0]}				

		m0,m1=self.SALTModel(x,evaluatePhase=self.phaseout,evaluateWave=self.waveout)

		#clscat = splev(self.wave,(self.errwaveknotloc,clscatpars,3))
		if not len(clpars): clpars = []

		# model errors
		m0err,m1err = self.ErrModel(x,evaluatePhase=self.phaseout,evaluateWave=self.waveout)
		cov_m0_m1 = self.CorrelationModel(x,evaluatePhase=self.phaseout,evaluateWave=self.waveout)[0]*m0err*m1err
		modelerr=np.ones(m0err.shape)
		return(x,self.phaseout,self.waveout,m0,m0err,m1,m1err,cov_m0_m1,modelerr,
			   clpars,clerr,clscat,resultsdict)

	def getErrsGN(self,m0var,m1var,m0m1cov):

		errs=[]
		for errpars in [m0var,m1var,m0m1cov]:
			if self.bsorder != 0:
				errs+=[	 bisplev(self.phase,
								 self.wave,
								 (self.phaseknotloc,self.waveknotloc,errpars,self.bsorder,self.bsorder))]
			else:
				n_repeat_phase = int(self.phase.size/(self.phaseknotloc.size-1))+1
				n_repeat_phase_extra = -1*(n_repeat_phase*(self.phaseknotloc.size-1) % self.phase.size)
				if n_repeat_phase_extra == 0: n_repeat_phase_extra = None
				n_repeat_wave = int(self.wave.size/(self.waveknotloc.size-1))+1
				n_repeat_wave_extra = -1*(n_repeat_wave*(self.waveknotloc.size-1) % self.wave.size)
				if n_repeat_wave_extra == 0: n_repeat_wave_extra = None
				errs += [np.repeat(np.repeat(errpars.reshape([self.phaseknotloc.size-1,self.waveknotloc.size-1]),n_repeat_phase,axis=0),n_repeat_wave,axis=1)[:n_repeat_phase_extra,:n_repeat_wave_extra]]


		return(np.sqrt(errs[0]),np.sqrt(errs[1]),errs[2])

	
	def getPars(self,loglikes,x,nburn=500,mkplots=False):

		axcount = 0; parcount = 0
		from matplotlib.backends.backend_pdf import PdfPages
		pdf_pages = PdfPages(f'{self.outputdir}/MCMC_hist.pdf')
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
			resultsdict[k] = {'x0':x[self.parlist == f'x0_{k}',nburn:].mean(),
							  'x1':x[self.parlist == f'x1_{k}',nburn:].mean(),# - x[self.ix1,nburn:].mean(),
							  'c':x[self.parlist == f'c_{k}',nburn:].mean(),
							  'tpkoff':x[self.parlist == f'tpkoff_{k}',nburn:].mean(),
							  'x0err':x[self.parlist == f'x0_{k}',nburn:].std(),
							  'x1err':x[self.parlist == f'x1_{k}',nburn:].std(),
							  'cerr':x[self.parlist == f'c_{k}',nburn:].std(),
							  'tpkofferr':x[self.parlist == f'tpkoff_{k}',nburn:].std()}


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
		clscat = self.colorscatter(np.mean(x[:,nburn:],axis=1),self.wave)
		if not len(clpars): clpars = []

		for snpar in ['x0','x1','c','tpkoff']:
			subnum = axcount%9+1
			ax = plt.subplot(3,3,subnum)
			axcount += 1
			md = np.mean(x[self.parlist == f'{snpar}_{k}',nburn:])
			std = np.std(x[self.parlist == f'{snpar}_{k}',nburn:])
			histbins = np.linspace(md-3*std,md+3*std,50)
			ax.hist(x[self.parlist == f'{snpar}_{k}',nburn:],bins=histbins)
			ax.set_title(f'{snpar}_{k}')
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
		self.neffRaw=np.zeros((self.phaseRegularizationPoints.size,self.waveRegularizationPoints.size))

		for snid,sn in self.datadict.items():
			tpkoff=x[self.parlist == f'tpkoff_{snid}']
			for k,spectrum in sn.specdata.items():
				# weight by ~mag err?
				err=spectrum.fluxerr/spectrum.flux
				snr=spectrum.flux/spectrum.fluxerr
				restWave=spectrum.wavelength/(1+sn.zHelio)
				phase=spectrum.phase
				
				if phase<self.phaseRegularizationBins[0]:
					phaseIndex=0
				elif phase>self.phaseRegularizationBins[-1]:
					phaseIndex=-1
				else:
					phaseIndex= np.where( (phase>=self.phaseRegularizationBins[:-1]) & (phase<self.phaseRegularizationBins[1:]))[0][0]
				#Weight each spectroscopic point's contribution relative to total flux

				self.neffRaw[phaseIndex,:]+=ss.binned_statistic(
					restWave,spectrum.flux/spectrum.flux.max()/len(spectrum),
					bins=self.waveRegularizationBins,statistic='sum').statistic
				

		self.neffRaw=gaussian_filter1d(self.neffRaw,self.phaseSmoothingNeff,0)
		self.neffRaw=gaussian_filter1d(self.neffRaw,self.waveSmoothingNeff,1)

		# hack!
		# D. Jones - just testing this out
		#for j,p in enumerate(self.phaseBinCenters):
		#	if np.max(self.neffRaw[j,:]) > 0: self.neffRaw[j,:] /= np.max(self.neffRaw[j,:])
		# HACK
		#self.neffRaw[self.neffRaw > 1] = 1
		#self.neffRaw[self.neffRaw < 1e-6] = 1e-6

#		self.plotEffectivePoints([-12.5,0.5,16.5,26],'neff.png')
#		self.plotEffectivePoints(None,'neff-heatmap.png')
		self.neff=self.neffRaw.copy()
		self.neff[self.neff>self.neffMax]=np.inf
		# HACK!
		#self.neff[self.neff<self.neffMax]=1e-6 #self.nefffloor
		#self.neff/=self.neffMax
		if not np.any(np.isinf(self.neff)): log.warning('Regularization is being applied to the entire phase/wavelength space: consider lowering neffmax (currently {:.2e})'.format(self.neffMax))
		
		self.neff=np.clip(self.neff,self.neffFloor,None)
		#self.neff=np.sqrt(self.neff)
		
	def plotEffectivePoints(self,phases=None,output=None):
		import matplotlib.pyplot as plt
		if phases is None:
			plt.imshow(self.neffRaw,cmap='Greys',aspect='auto')
			xticks=np.linspace(0,self.waveRegularizationPoints.size,8,False)
			plt.xticks(xticks,['{:.0f}'.format(self.waveRegularizationPoints[int(x)]) for x in xticks])
			plt.xlabel('$\lambda$ / Angstrom')
			yticks=np.linspace(0,self.phaseRegularizationPoints.size,8,False)
			plt.yticks(yticks,['{:.0f}'.format(self.phaseRegularizationPoints[int(x)]) for x in yticks])
			plt.ylabel('Phase / days')
		else:
			inds=np.searchsorted(self.phaseRegularizationPoints,phases)
			# hack!
			for i in inds:
				plt.plot(self.waveRegularizationPoints[:],self.neffRaw[i,:],label='{:.1f} days'.format(self.phaseRegularizationPoints[i]))
			plt.ylabel('$N_eff$')
			plt.xlabel('$\lambda (\AA)$')
			plt.xlim(self.waveRegularizationPoints.min(),self.waveRegularizationPoints.max())
			plt.legend()

		if output is None:
			plt.show()
		else:
			plt.savefig(output,dpi=288)
		plt.clf()
	
	def regularizationScale(self,components,fluxes,regmethod='none'):
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
			iqr=[ (np.percentile(self.neffRaw,80)<self.neffRaw)	 for flux in fluxes]
			scale= [np.mean(np.abs(flux[select])) for select,flux in zip(iqr,fluxes)]
			return scale, [np.mean(np.sign(flux[select,np.newaxis])*self.regularizationDerivs[0][select,:],axis=(0)) for s,select,flux in zip(scale,iqr,fluxes)]
		elif self.regularizationScaleMethod=='snpca':
			iqr=[ (np.percentile(self.neffRaw,80)<self.neffRaw)	 for flux in fluxes]
			if regmethod == 'gradient': scale = [2746.769325]*len(fluxes)
			elif regmethod == 'dyadic': scale = [361838.9736]*len(fluxes)
			return scale, [np.mean(self.regularizationDerivs[0][select,:],axis=(0)) for select,flux in zip(iqr,fluxes)]		
		else:
			raise ValueError('Regularization scale method invalid: ',self.regularizationScaleMethod)

	def dyadicRegularization(self,x, storedResults,varyParams):
		phase=self.phaseRegularizationPoints
		wave=self.waveRegularizationPoints
		
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdwave=self.SALTModelDeriv(x,0,1,phase,wave)
		dfluxdphase=self.SALTModelDeriv(x,1,0,phase,wave)
		d2fluxdphasedwave=self.SALTModelDeriv(x,1,1,phase,wave)
		scale,scaleDeriv=self.regularizationScale(storedResults['components'],fluxes,regmethod='dyadic')
		resids=[]
		jac=[]

		for i in range(len(fluxes)):
			indices=[self.im0,self.im1][i]
			boolIndex=np.zeros(self.npar,dtype=bool)
			boolIndex[indices]=True

			#Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
			normalization=np.sqrt(1/( (self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))**2.
			#0 if model is locally separable in phase and wavelength i.e. flux=g(phase)* h(wavelength) for arbitrary functions g and h
			numerator=(dfluxdphase[i] *dfluxdwave[i] -d2fluxdphasedwave[i] *fluxes[i] )
			
			dnumerator=( self.regularizationDerivs[1][:,:,varyParams[indices]]*dfluxdwave[i][:,:,np.newaxis] + self.regularizationDerivs[2][:,:,varyParams[indices]]* dfluxdphase[i][:,:,np.newaxis] 
				- self.regularizationDerivs[3][:,:,varyParams[indices]]* fluxes[i][:,:,np.newaxis] - self.regularizationDerivs[0][:,:,varyParams[indices]]* d2fluxdphasedwave[i][:,:,np.newaxis] )			
			resids += [normalization* (numerator / (scale[i]**2 * self.neff)).flatten()]
			if boolIndex[varyParams].any(): jacobian=jacobian=np.zeros((resids[-1].size,self.parlist.size))
			else: jacobian = jacobian = sparse.csr_matrix((resids[-1].size,self.parlist.size))
			
			if boolIndex[varyParams].any():
				jacobian[:,boolIndex]=((dnumerator*(scale[i]**2 )- scaleDeriv[i][np.newaxis,np.newaxis,varyParams[indices]]*2*scale[i]*numerator[:,:,np.newaxis])/self.neff[:,:,np.newaxis]*normalization / scale[i]**4	 ).reshape(-1, varyParams[indices].sum())
			jac += [sparse.csr_matrix(jacobian)]
			
		return resids,jac

	def dyadicRegularizationTest(self,x, storedResults,varyParams):
		phase=self.phaseRegularizationPoints
		wave=self.waveRegularizationPoints
		
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdwave=self.SALTModelDeriv(x,0,1,phase,wave)
		dfluxdphase=self.SALTModelDeriv(x,1,0,phase,wave)
		d2fluxdphasedwave=self.SALTModelDeriv(x,1,1,phase,wave)
		#scale,scaleDeriv=self.regularizationScale(storedResults['components'],fluxes,regmethod='dyadic')
		resids=[]
		jac=[]
		for i in range(len(fluxes)):
			indices=[self.im0,self.im1][i]
			boolIndex=np.zeros(self.npar,dtype=bool)
			boolIndex[indices]=True

			#Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
			normalization=361838973.6
			#0 if model is locally separable in phase and wavelength i.e. flux=g(phase)* h(wavelength) for arbitrary functions g and h
			numerator=(dfluxdphase[i] *dfluxdwave[i] -d2fluxdphasedwave[i] *fluxes[i] )
			dnumerator=( self.regularizationDerivs[1][:,:,varyParams[indices]]*dfluxdwave[i][:,:,np.newaxis] + self.regularizationDerivs[2][:,:,varyParams[indices]]* dfluxdphase[i][:,:,np.newaxis] 
				- self.regularizationDerivs[3][:,:,varyParams[indices]]* fluxes[i][:,:,np.newaxis] - self.regularizationDerivs[0][:,:,varyParams[indices]]* d2fluxdphasedwave[i][:,:,np.newaxis] )			
			resids += [normalization* (numerator).flatten()]
			jacobian=sparse.csr_matrix((resids[-1].size,self.parlist.size))
			if boolIndex[varyParams].any():
				jacobian[:,boolIndex[varyParams]]=(dnumerator).reshape(-1, varyParams[indices].sum())
			jac += [jacobian]
		return resids,jac

	
	def phaseGradientRegularization(self, x, storedResults,varyParams):
		phase=self.phaseRegularizationPoints
		wave=self.waveRegularizationPoints
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdphase=self.SALTModelDeriv(x,1,0,phase,wave)
		scale,scaleDeriv=self.regularizationScale(storedResults['components'],fluxes,regmethod='gradient')
		resids=[]
		jac=[]
		for i in range(len(fluxes)):
			indices=[self.im0,self.im1][i]
			boolIndex=np.zeros(self.npar,dtype=bool)
			boolIndex[indices]=True
			#Normalize gradient by flux scale
			normedGrad=dfluxdphase[i]/scale[i]
			#Derivative of normalized gradient with respect to model parameters
			normedGradDerivs=(self.regularizationDerivs[1][:,:,varyParams[indices]] * scale[i] - scaleDeriv[i][np.newaxis,np.newaxis,varyParams[indices]]*dfluxdphase[i][:,:,np.newaxis])/ scale[i]**2
			#Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
			normalization=np.sqrt(1/((self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))
			#Minimize model derivative w.r.t wavelength in unconstrained regions
			resids+= [normalization* ( normedGrad /	self.neff).flatten()]
			#jacobian=np.zeros((resids[-1].size,self.parlist.size)) #sparse.csr_matrix((resids[-1].size,self.parlist.size))
			if boolIndex[varyParams].any(): jacobian=jacobian=np.zeros((resids[-1].size,self.parlist.size))
			else: jacobian = jacobian = sparse.csr_matrix((resids[-1].size,self.parlist.size))

			if boolIndex[varyParams].any():
				jacobian[:,boolIndex]=normalization*((normedGradDerivs) / self.neff[:,:,np.newaxis]).reshape(-1, varyParams[indices].sum())
				#jacobian[:,boolIndex[varyParams]]=normalization*((normedGradDerivs) / self.neff[:,:,np.newaxis]).reshape(-1, varyParams[indices].sum())
			jac+= [sparse.csr_matrix(jacobian)]
				
		return resids,jac
	
	def waveGradientRegularization(self, x, storedResults,varyParams):
		#Declarations
		phase=self.phaseRegularizationPoints
		wave=self.waveRegularizationPoints
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdwave=self.SALTModelDeriv(x,0,1,phase,wave)
		scale,scaleDeriv=self.regularizationScale(storedResults['components'],fluxes,regmethod='gradient')
		waveGradResids=[]
		jac=[]
		for i in range(len(fluxes)):
			indices=[self.im0,self.im1][i]
			boolIndex=np.zeros(self.npar,dtype=bool)
			boolIndex[indices]=True

			#Normalize gradient by flux scale
			normedGrad=dfluxdwave[i]/scale[i]
			#Derivative of normalized gradient with respect to model parameters
			normedGradDerivs=(self.regularizationDerivs[2][:,:,varyParams[indices]] * scale[i] - scaleDeriv[i][np.newaxis,np.newaxis,varyParams[indices]]*dfluxdwave[i][:,:,np.newaxis])/ scale[i]**2
			#Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
			normalization=np.sqrt(1/((self.waveBins[0].size-1) *(self.phaseBins[0].size-1)))
			#Minimize model derivative w.r.t wavelength in unconstrained regions
			waveGradResids+= [normalization* ( normedGrad /	self.neff).flatten()]
			#jacobian=np.zeros((waveGradResids[-1].size,self.parlist.size)) #sparse.csr_matrix((waveGradResids[-1].size,self.parlist.size))
			if boolIndex[varyParams].any(): jacobian=jacobian=np.zeros((waveGradResids[-1].size,self.parlist.size))
			else: jacobian = jacobian = sparse.csr_matrix((waveGradResids[-1].size,self.parlist.size))

			if boolIndex[varyParams].any():
				jacobian[:,boolIndex]=normalization*((normedGradDerivs) / self.neff[:,:,np.newaxis]).reshape(-1, varyParams[indices].sum())
				#jacobian[:,boolIndex[varyParams]]=normalization*((normedGradDerivs) / self.neff[:,:,np.newaxis]).reshape(-1, varyParams[indices].sum())
			jac+= [jacobian]
		#if len(waveGradResids[waveGradResids != waveGradResids]): import pdb; pdb.set_trace()
		return waveGradResids,jac

	#def waveGradientRegularization_snpca(self, x, storedResults,varyParams):
	#	nx = len(self.phaseRegularizationPoints)
	#	ny = len(self.waveRegularizationPoints)
	#	
	#	def total_weight():
#
#			reference_flux = 7.1739e+13
#			bmaxintegral = 2.008780068e+14
#			
#			scale = 1./(nx*(ny-1)) # remove dependence on number of bins
#			scale *= (bmaxintegral/reference_flux)**2. # remove dependence on both total flux and values of the function basis
#			# now chi2 is in unit of (% variation per A)^2
#
#			# calibrated using SNLS3 template 0:
#			scale *= 4.85547e5 # so approx. 0.15% variation allowed per A (wrt to flux at max)
#
#			scale *= weight; # tunable weight
#			return scale
#
#
#		def chi2():
#
#			waveGradResids=[]
#			
#			result = 0
#			#v = SV.Data()
#			y = self.waveRegularizationPoints #SV.MeanLambdas().Data()
#			#m = 0; if(mask) m = mask->Data();
#
#			phase=self.phaseRegularizationPoints
#			wave=self.waveRegularizationPoints
#			fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave).flatten()
 # 
#			for j in range(ny):
#				for i in range(nx):
 #	   
#					i0 = i+nx*j
#					i1 = i+nx*(j+1)
 #	  
#					waveGradResids += [( ( fluxes[i1] - fluxes[i0] ) / ( y[j+1] - y[j] ) )**2.]
#			
#			return total_weight()*np.array(waveGradResids)
#
#		def chi2_derivatives(jacobian): #const FunctionBasis &V, Mat &A, Vect& B, const Vect* mask) const {
 # 
#			# A and B can be already filled with some other regularization term, so allocate only if necessary	
 #	 
#			residual = 0
#			dy_inv = 0
#			dy_inv2 = 0
#
#			phase=self.phaseRegularizationPoints
#			wave=self.waveRegularizationPoints
#			fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave).flatten()
#			y = self.waveRegularizationPoints #SV.MeanLambdas().Data()
#			#m = 0; if(mask) m = mask->Data()
#			scale = total_weight()
 # 
#			for j in range(ny):
#				for i in range(nx):
#					i0 = i+nx*j
#					i1 = i+nx*(j+1)
#					
#					dy_inv =  1. / ( y[j+1] - y[j] )
#					residual = ( fluxes[i1] - fluxes[i0] ) * dy_inv
 #	   
#					B[i0] += scale * residual * dy_inv # -0.5*dchi2/di = -residual*dres/di
#					B[i1] -= scale * residual * dy_inv
#					dy_inv2 = np.sqrt(dy_inv);
 #	   
#					jacobian[i0,i0] += scale * dy_inv2 # 0.5*d2chi2/didj = dres/di*dres/dj
#					jacobian[i1,i1] += scale * dy_inv2
#					jacobian[i0,i1] -= scale * dy_inv2
#					jacobian[i1,i0] -= scale * dy_inv2
#
#
#		waveGradResids = chi2()
#		jacobian=np.zeros((waveGradResids[-1].size,varyParams.sum()))
#		jacobian = chi2_derivatives(jacobian)
		
	def waveGradientRegularizationTest(self, x, storedResults,varyParams):
		#Declarations
		phase=self.phaseRegularizationPoints
		wave=self.waveRegularizationPoints
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdwave=self.SALTModelDeriv(x,0,1,phase,wave)
		#scale,scaleDeriv=self.regularizationScale(storedResults['components'],fluxes,regmethod='gradient')
		waveGradResids=[]
		jac=[]
		for i in range(len(fluxes)):
			indices=[self.im0,self.im1][i]
			boolIndex=np.zeros(self.npar,dtype=bool)
			boolIndex[indices]=True

			#Normalize gradient by flux scale
			normedGrad=dfluxdwave[i]
			#Derivative of normalized gradient with respect to model parameters
			normedGradDerivs=(self.regularizationDerivs[2][:,:,varyParams[indices]])
			#Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
			normalization=27467.69325
			#Minimize model derivative w.r.t wavelength in unconstrained regions
			waveGradResids+= [normalization* ( normedGrad /	self.neff).flatten()]
			jacobian=sparse.csr_matrix((waveGradResids[-1].size,self.parlist.size))
			if boolIndex[varyParams].any():
				jacobian[:,boolIndex[varyParams]]=normalization*((normedGradDerivs)).reshape(-1, varyParams[indices].sum())
			jac+= [jacobian]

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
