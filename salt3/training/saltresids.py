from scipy.interpolate import splprep,splev,bisplev,bisplrep,interp1d,interp2d
from scipy.integrate import trapz
from salt3.util.synphot import synphot
from sncosmo.salt2utils import SALT2ColorLaw
from itertools import starmap
from salt3.training import init_hsiao
from sncosmo.models import StretchSource
from scipy.optimize import minimize, least_squares
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from scipy.special import factorial
from astropy.cosmology import Planck15 as cosmo
from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
from multiprocessing import Pool, get_context
from numpy.random import standard_normal
from scipy.linalg import cholesky
from sncosmo.utils import integration_grid
from numpy.linalg import inv,pinv
import time
import numpy as np
import pylab as plt
import extinction
import copy
import scipy.stats as ss

_SCALE_FACTOR = 1e-12
_B_LAMBDA_EFF = np.array([4302.57])	 # B-band-ish wavelength
_V_LAMBDA_EFF = np.array([5428.55])	 # V-band-ish wavelength


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

		
		# initialize derivatives for the priors
		self.priorderivdict = {'Bmax':np.zeros(self.npar),
							   'x1mean':np.zeros(self.npar),
							   'x1std':np.zeros(self.npar),
							   'colorlaw':np.zeros(self.npar),
							   'M0end':np.zeros(self.npar),
							   'M1end':np.zeros(self.npar),
							   'tBmax':np.zeros(self.npar)}
		self.npriors = 7
		self.priors = np.array(self.priorderivdict.keys())
		self.bsorder=3
		self.guess = guess
		self.lsqfit = False
		self.nsn = len(self.datadict.keys())
		
		for key, value in kwargs.items(): 
			self.__dict__[key] = value

		# pre-set some indices
		self.m0min = np.min(np.where(self.parlist == 'm0')[0])
		self.m0max = np.max(np.where(self.parlist == 'm0')[0])
		self.errmin = np.min(np.where(self.parlist == 'modelerr')[0])
		self.errmax = np.max(np.where(self.parlist == 'modelerr')[0])
		self.ix1 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x1')])
		self.ix0 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('x0')])
		self.ic	 = np.array([i for i, si in enumerate(self.parlist) if si.startswith('c')])
		self.im0 = np.where(self.parlist == 'm0')[0]
		self.im1 = np.where(self.parlist == 'm1')[0]
		self.iCL = np.where(self.parlist == 'cl')[0]
		
		# set some phase/wavelength arrays
		self.splinecolorwave = np.linspace(self.colorwaverange[0],self.colorwaverange[1],self.n_colorpars)
		self.phasebins = np.linspace(self.phaserange[0],self.phaserange[1],
							 1+ (self.phaserange[1]-self.phaserange[0])/self.phaseres)
		self.wavebins = np.linspace(self.waverange[0],self.waverange[1],
							 1+(self.waverange[1]-self.waverange[0])/self.waveres)

		self.phase = np.linspace(self.phaserange[0],self.phaserange[1],
								 (self.phaserange[1]-self.phaserange[0])/self.phaseoutres,False)
		self.wave = np.linspace(self.waverange[0],self.waverange[1],
								(self.waverange[1]-self.waverange[0])/self.waveoutres,False)
		self.maxPhase=np.where(abs(self.phase) == np.min(abs(self.phase)))[0]
		
		self.splinephase = np.linspace(self.phaserange[0]-self.phaseres*0,
									   self.phaserange[1]+self.phaseres*0,
									   (self.phaserange[1]-self.phaserange[0])/self.phaseres+0*2,False)
		self.splinewave = np.linspace(self.waverange[0]-self.waveres*0,
									  self.waverange[1]+self.waveres*0,
									  (self.waverange[1]-self.waverange[0])/self.waveres+0*2,False)


		self.neff=0
		self.updateEffectivePoints(guess)

		# initialize the model
		self.components = self.SALTModel(guess)
		self.salterr = self.ErrModel(guess)
		self.salterr[:] = 1e-10
		
		self.m0guess = -19.49 #10**(-0.4*(-19.49-27.5))
		self.m1guess = 1
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
			
			self.datadict[sn]['mwextcurve'] = np.zeros([len(self.phase),len(self.wave)])
			for i in range(len(self.phase)):
				self.datadict[sn]['mwextcurve'][i,:] = 10**(-0.4*extinction.fitzpatrick99(self.wave*(1+z),self.datadict[sn]['MWEBV']*3.1))

			self.num_spec += sum([specdata[key]['flux'].size for key in specdata])

			for flt in np.unique(photdata['filt']):
				self.num_phot+=(photdata['filt']==flt).sum()

		#Store derivatives of a spline with fixed knot locations with respect to each knot value
		starttime=time.time()
		self.spline_derivs=[ bisplev(self.phase,self.wave,(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder)) for i in range(self.im0.size)]
		
		phase=(self.phasebins[:-1]+self.phasebins[1:])/2
		wave=(self.wavebins[:-1]+self.wavebins[1:])/2
		self.regularizationDerivs=[np.zeros((phase.size,wave.size,self.im0.size)) for i in range(4)]
		for i in range(len(self.im0)):
			for j,derivs in enumerate([(0,0),(1,0),(0,1),(1,1)]):
				self.regularizationDerivs[j][:,:,i]=bisplev(phase,wave,(self.phaseknotloc,self.waveknotloc,np.arange(self.im0.size)==i,self.bsorder,self.bsorder),dx=derivs[0],dy=derivs[1])
			
		print('Time to calculate spline_derivs: %.2f'%(time.time()-starttime))

		M0derivinterp,M1derivinterp,modcompall = [],[],[]
		for i,h,j in zip(self.im0,self.im1,range(len(self.im0))):
			xtmp = copy.deepcopy(guess)
			xtmp[i] += 1.0e-3; xtmp[h] += 1.0e-3
			modcomp = self.SALTModel(xtmp)
			modcompall += [modcomp]
		self.modcomp = modcompall
		
		self.getobswave()

							
	def getobswave(self):
		"for each filter, setting up some things needed for synthetic photometry"
		
		for sn in self.datadict.keys():
			z = self.datadict[sn]['zHelio']
			survey = self.datadict[sn]['survey']
			filtwave = self.kcordict[survey]['filtwave']

			self.datadict[sn]['obswave'] = self.wave*(1+z)
			self.datadict[sn]['obsphase'] = self.phase*(1+z)
			self.datadict[sn]['pbspl'] = {}
			self.datadict[sn]['denom'] = {}
			self.datadict[sn]['idx'] = {}
			self.datadict[sn]['dwave'] = self.wave[1]*(1+z) - self.wave[0]*(1+z)
			for flt in np.unique(self.datadict[sn]['photdata']['filt']):

				filttrans = self.kcordict[survey][flt]['filttrans']

				g = (self.datadict[sn]['obswave'] >= filtwave[0]) & (self.datadict[sn]['obswave'] <= filtwave[-1])	# overlap range
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
				
	def maxlikefit(self,x,pool=None,debug=False,timeit=False):
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
		args=[(None,sn,x,components,self.salterr,colorLaw,colorScat,debug,timeit) for sn in self.datadict.keys()]
		if pool:
			loglike=sum(pool.map(self.loglikeforSN,args))
		else:
			loglike=sum(starmap(self.loglikeforSN,args))

		loglike -= self.regularizationChi2(x,self.regulargradientphase,self.regulargradientwave,self.regulardyad)/2

		
		logp = loglike + self.m0prior(components)[0] + self.m1prior(x[self.ix1]) + self.endprior(components)+self.peakprior(x,components)
			
		self.nstep += 1
		if not self.lsqfit: print(logp*-2)
		else: print(logp.sum()*-2)

		if timeit:
			print('%.3f %.3f %.3f %.3f %.3f'%(self.tdelt0,self.tdelt1,self.tdelt2,self.tdelt3,self.tdelt4))

		return logp

	def m0prior(self,components,m0=False):
		int1d = interp1d(self.phase,components[0],axis=0)
		m0Bflux = np.sum(self.kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
			self.kcordict['default']['Bdwave']*self.kcordict['default']['fluxfactor']

		if m0Bflux > 0:
			m0B = -2.5*np.log10(m0Bflux) + 27.5
		else: m0B = 99
		if m0: return -2.5*np.log10(m0Bflux) + 27.5
		else:
			logprior = norm.logpdf(m0B,self.m0guess,0.1)
			return logprior

	def m0prior_lsq(self,components,m0=False):
		int1d = interp1d(self.phase,components[0],axis=0)
		m0Bflux = np.sum(self.kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
			self.kcordict['default']['Bdwave']*self.kcordict['default']['fluxfactor']
		m0B = -2.5*np.log10(m0Bflux) + 27.5
		return (m0B-self.m0guess)/0.01

	def m1prior(self,x1pars):

		logprior = norm.logpdf(np.std(x1pars),self.m1guess,0.02)

		return logprior

	def m1priormean(self,x1pars):

		logprior = norm.logpdf(np.mean(x1pars),0,0.02)

		return logprior

	def m1priorstd(self,x1pars):

		logprior = norm.logpdf(np.std(x1pars),self.m1guess,0.02)

		return logprior
		
	def peakprior(self,x,components):
		wave=self.wave[self.bbandoverlap]
		lightcurve=np.sum(self.bbandpbspl[np.newaxis,:]*components[0][:,self.bbandoverlap],axis=1)
		# from D'Arcy - disabled for now!!	(barfs if model goes crazy w/o regularization)
		#maxPhase=np.argmax(lightcurve)
		#finePhase=np.arange(self.phase[maxPhase-1],self.phase[maxPhase+1],0.1)
		finePhase=np.arange(self.phase[self.maxPhase-1],self.phase[self.maxPhase+1],0.1)
		fineGrid=self.SALTModel(x,evaluatePhase=finePhase,evaluateWave=wave)[0]
		lightcurve=np.sum(self.bbandpbspl[np.newaxis,:]*fineGrid,axis=1)
		logprior = norm.logpdf(finePhase[np.argmax(lightcurve)],0,0.5)
		
		return logprior

	def endprior(self,components):
		
		logprior = norm.logpdf(np.sum(components[0][0,:]),0,0.1)
		logprior += norm.logpdf(np.sum(components[1][0,:]),0,0.1)
		
		return logprior

	def m0endprior(self,components):
		
		logprior = norm.logpdf(np.sum(components[0][0,:]),0,0.1)
		
		return logprior

	def m1endprior(self,components):
		
		logprior = norm.logpdf(np.sum(components[1][0,:]),0,0.1)
		
		return logprior

	def m0endprior_lsq(self,components):
		
		return np.sum(components[0][0,:])/0.1

	def m1endprior_lsq(self,components):
		
		return np.sum(components[1][0,:])/0.1

	
	def EBVprior(self,colorLaw):
		# 0.4*np.log(10) = 0.921
		logpriorB = norm.logpdf(colorLaw(_B_LAMBDA_EFF)[0], 0.0, 0.02)
		logpriorV = norm.logpdf(colorLaw(_V_LAMBDA_EFF)[0], 0.921, 0.02)
		return logpriorB + logpriorV
				
	def ResidsForSN(self,x,sn,components,colorLaw,computeDerivatives,computePCDerivs=True):
		photmodeldict,specmodeldict=self.modelvalsforSN(x,sn,components,colorLaw,computeDerivatives,computePCDerivs)
		
		photresidsdict={key.replace('dmodelflux','dphotresid'):photmodeldict[key]/(photmodeldict['uncertainty'][:,np.newaxis]) for key in photmodeldict if 'modelflux' in key}
		photresidsdict['photresid']=(photmodeldict['modelflux']-self.datadict[sn]['photdata']['fluxcal'])/photmodeldict['uncertainty']
		photresidsdict['lognorm']=np.log(1/(np.sqrt(2*np.pi)*photmodeldict['uncertainty'])).sum()

		#Suppress the effect of the spectra by multiplying chi^2 by number of photometric points over number of spectral points
		spectralSuppression=np.sqrt(self.num_phot/self.num_spec)
		
		specresidsdict={key.replace('dmodelflux','dspecresid'):specmodeldict[key]/(specmodeldict['uncertainty'][:,np.newaxis])*spectralSuppression for key in specmodeldict if 'modelflux' in key}
		specresidsdict['specresid']=(specmodeldict['modelflux']-specmodeldict['dataflux'])/specmodeldict['uncertainty']*spectralSuppression
		#Not sure if this is the best way to account for the spectral suppression in the log normalization term?
		specresidsdict['lognorm']=np.log(spectralSuppression/(np.sqrt(2*np.pi)*specmodeldict['uncertainty'])).sum()

		return photresidsdict,specresidsdict
		
	def modelvalsforSN(self,x,sn,components,colorLaw,computeDerivatives,computePCDerivs):
		# model pars, initialization
		if self.n_components == 1:
			M0 = copy.deepcopy(components[0])
		elif self.n_components == 2:
			M0,M1 = copy.deepcopy(components)
		
		z = self.datadict[sn]['zHelio']
		survey = self.datadict[sn]['survey']
		filtwave = self.kcordict[survey]['filtwave']
		obswave = self.datadict[sn]['obswave'] #self.wave*(1+z)
		obsphase = self.datadict[sn]['obsphase'] #self.phase*(1+z)
		photdata = self.datadict[sn]['photdata']
		specdata = self.datadict[sn]['specdata']
		pbspl = self.datadict[sn]['pbspl']
		dwave = self.datadict[sn]['dwave']
		idx = self.datadict[sn]['idx']

		nspecdata = sum([specdata[key]['flux'].size for key in specdata])
		interr1d = interp1d(obsphase,self.salterr,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")

		x0,x1,c,tpkoff = x[self.parlist == 'x0_%s'%sn],x[self.parlist == 'x1_%s'%sn],\
						 x[self.parlist == 'c_%s'%sn],x[self.parlist == 'tpkoff_%s'%sn]
		clpars = x[self.parlist == 'cl']
		##x1 = x1 - np.mean(x[self.ix1])
		
		#Calculate spectral model
		M0 *= self.datadict[sn]['mwextcurve']
		M1 *= self.datadict[sn]['mwextcurve']
		
		if colorLaw:
			colorlaw = -0.4 * colorLaw(self.wave)
			colorexp = 10. ** (colorlaw * c)
			M0 *= colorexp; M1 *= colorexp
			
		M0 *= _SCALE_FACTOR/(1+z); M1 *= _SCALE_FACTOR/(1+z)
		int1dM0 = interp1d(obsphase,M0,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
		int1dM1 = interp1d(obsphase,M1,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")

		# spectra
		intspecerr1d=interr1d

		specresultsdict={}
		specresultsdict['modelflux'] = np.zeros(nspecdata)
		specresultsdict['dataflux'] = np.zeros(nspecdata)
		specresultsdict['uncertainty'] =  np.zeros(nspecdata)
		if computeDerivatives:
			specresultsdict['dmodelflux_dx0'] = np.zeros((nspecdata,1))
			specresultsdict['dmodelflux_dx1'] = np.zeros((nspecdata,1))
			specresultsdict['dmodelflux_dc']  = np.zeros((nspecdata,1))
			specresultsdict['dmodelflux_dM0'] = np.zeros([nspecdata,len(self.im0)])
			specresultsdict['dmodelflux_dM1'] = np.zeros([nspecdata,len(self.im1)])
			specresultsdict['dmodelflux_dcl'] = np.zeros([nspecdata,self.n_colorpars])


		iSpecStart = 0
		for k in specdata.keys():
			SpecLen = specdata[k]['flux'].size
			phase=specdata[k]['tobs']+tpkoff
			M0interp = int1dM0(phase)
			M1interp = int1dM1(phase)

			#Check spectrum is inside proper phase range, extrapolate decline if necessary
			if phase < obsphase.min():
				pass
			elif phase > obsphase.max():
				saltfluxinterp*=10**(-0.4* self.extrapolateDecline* (phase-obsphase.max()))

			#Define recalibration factor
			coeffs=x[self.parlist=='specrecal_{}_{}'.format(sn,k)]
			coeffs/=factorial(np.arange(len(coeffs)))
			recalexp = np.exp(np.poly1d(coeffs)((specdata[k]['wavelength']-np.mean(specdata[k]['wavelength']))/self.specrange_wavescale_specrecal))


			M0interp = np.interp(specdata[k]['wavelength'],obswave,M0interp[0])*recalexp
			M1interp = np.interp(specdata[k]['wavelength'],obswave,M1interp[0])*recalexp
			colorexpinterp = np.interp(specdata[k]['wavelength'],obswave,colorexp)
			colorlawinterp = np.interp(specdata[k]['wavelength'],obswave,colorlaw)

			modulatedFlux = x0*(M0interp + x1*M1interp)*recalexp
			
			specresultsdict['modelflux'][iSpecStart:iSpecStart+SpecLen] = modulatedFlux
			specresultsdict['dataflux'][iSpecStart:iSpecStart+SpecLen] = specdata[k]['flux']

			
			modelerr = np.interp( specdata[k]['wavelength'],obswave,intspecerr1d(phase)[0])
			specvar=(specdata[k]['fluxerr']**2.) # + (1e-3*saltfluxinterp)**2.)
			specresultsdict['uncertainty'][iSpecStart:iSpecStart+SpecLen] = np.sqrt(specvar) #+ (modelerr*specresultsdict['modelflux'][iSpecStart:iSpecStart+SpecLen])**2
			
			# derivatives....
			if computeDerivatives:
				specresultsdict['dmodelflux_dc'][iSpecStart:iSpecStart+SpecLen,0] = x0*(M0interp + x1*M1interp)*np.log(10)*colorlawinterp
				specresultsdict['dmodelflux_dx0'][iSpecStart:iSpecStart+SpecLen,0] = (M0interp + x1*M1interp)
				specresultsdict['dmodelflux_dx1'][iSpecStart:iSpecStart+SpecLen,0] = x0*M1interp

				
				# color law
				for i in range(self.n_colorpars):
					dcolorlaw_dcli = SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)(specdata[k]['wavelength']/(1+z))-SALT2ColorLaw(self.colorwaverange, np.zeros(self.n_colorpars))(specdata[k]['wavelength']/(1+z))
					specresultsdict['dmodelflux_dcl'][iSpecStart:iSpecStart+SpecLen,i] = modulatedFlux*-0.4*np.log(10)*c*dcolorlaw_dcli
					
				# M0, M1
				if computePCDerivs:
					intmult = _SCALE_FACTOR/(1+z)*x0*recalexp*colorexpinterp
					for i in range(len(self.im0)):
						#Range of wavelength and phase values affected by changes in knot i
						waverange=self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
						phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-self.bsorder-1),i//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
						#Check if this spectrum is inside values affected by changes in knot i
						if waverange[0]*(1+z) > specdata[k]['wavelength'].min() or waverange[1]*(1+z) < specdata[k]['wavelength'].max():
							pass
						#Check which phases are affected by knot i
						inPhase=(phase>phaserange[0]*(1+z) ) & (phase<phaserange[1]*(1+z) )
						if inPhase.any():
							#Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
							#Integrate only over wavelengths within the relevant range
							inbounds=(self.wave>waverange[0]) & (self.wave<waverange[1])
							derivInterp = interp1d(obsphase/(1+z),self.spline_derivs[i][:,inbounds],axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
							#import pdb; pdb.set_trace()
							derivInterp2 = np.interp(specdata[k]['wavelength'],obswave[inbounds],derivInterp(phase[0]/(1+z)))*intmult
							
							specresultsdict['dmodelflux_dM0'][iSpecStart:iSpecStart+SpecLen,i] = derivInterp2
							#Dependence is the same for dM1, except with an extra factor of x1
							specresultsdict['dmodelflux_dM1'][iSpecStart:iSpecStart+SpecLen,i] =  specresultsdict['dmodelflux_dM0'][iSpecStart:iSpecStart+SpecLen,i]*x1
					
					if ( (phase>obsphase.max())).any():
						if phase > obsphase.max():
							specresultsdict['dmodelflux_dM0'][iSpecStart:iSpecStart+SpecLen,:] *= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))
							specresultsdict['dmodelflux_dM1'][iSpecStart:iSpecStart+SpecLen,:] *= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))

			iSpecStart += SpecLen

		
		photresultsdict={}
		photresultsdict['modelflux'] = np.zeros(len(photdata['filt']))
		photresultsdict['uncertainty'] =  np.zeros(len(photdata['filt']))
		if computeDerivatives:
			photresultsdict['dmodelflux_dx0'] = np.zeros((photdata['filt'].size,1))
			photresultsdict['dmodelflux_dx1'] = np.zeros((photdata['filt'].size,1))
			photresultsdict['dmodelflux_dc']  = np.zeros((photdata['filt'].size,1))
			photresultsdict['dmodelflux_dM0'] = np.zeros([photdata['filt'].size,len(self.im0)])#*1e-6 #+ 1e-5
			photresultsdict['dmodelflux_dM1'] = np.zeros([photdata['filt'].size,len(self.im1)])#*1e-6 #+ 1e-5
			photresultsdict['dmodelflux_dcl'] = np.zeros([photdata['filt'].size,self.n_colorpars])

		for flt in np.unique(photdata['filt']):
			phase=photdata['tobs']+tpkoff
			#Select data from the appropriate filter filter
			selectFilter=(photdata['filt']==flt)
			
			filtPhot={key:photdata[key][selectFilter] for key in photdata}
			phase=phase[selectFilter]
			nphase = len(phase)

			salterrinterp = interr1d(phase)
			modelerr = np.sum(pbspl[flt]*salterrinterp[:,idx[flt]], axis=1) * dwave*HC_ERG_AA
			colorerr = 0
			
			
			#Array output indices match time along 0th axis, wavelength along 1st axis
			M0interp = int1dM0(phase)
			M1interp = int1dM1(phase)
			if computeDerivatives:
				modulatedM0= pbspl[flt]*M0interp[:,idx[flt]]
				modulatedM1=pbspl[flt]*M1interp[:,idx[flt]]
				
				if ( (phase>obsphase.max())).any():
					iMax = np.where(phase>obsphase.max())[0]
					for i in iMax:
						modulatedM0[i] *= 10**(-0.4*self.extrapolateDecline*(phase[i]-obsphase.max()))
						modulatedM1[i] *= 10**(-0.4*self.extrapolateDecline*(phase[i]-obsphase.max()))
						
				modelsynM0flux=np.sum(modulatedM0, axis=1)*dwave*self.fluxfactor[survey][flt]
				modelsynM1flux=np.sum(modulatedM1, axis=1)*dwave*self.fluxfactor[survey][flt]
				
				photresultsdict['dmodelflux_dx0'][selectFilter,0] = modelsynM0flux+ x1*modelsynM1flux
				photresultsdict['dmodelflux_dx1'][selectFilter,0] = modelsynM1flux*x0
				
				modulatedFlux= x0*(modulatedM0 +modulatedM1*x1)
				modelflux = x0* (modelsynM0flux+ x1*modelsynM1flux)
			else:
				modelflux = x0* np.sum(pbspl[flt]*(M0interp[:,idx[flt]]+x1*M1interp[:,idx[flt]]), axis=1)*dwave*self.fluxfactor[survey][flt]
				if ( (phase>obsphase.max())).any():
					modelflux[(phase>obsphase.max())]*= 10**(-0.4*self.extrapolateDecline*(phase-obsphase.max()))[(phase>obsphase.max())]
			
			photresultsdict['modelflux'][selectFilter] = modelflux
			# modelflux
			photresultsdict['uncertainty'][selectFilter] = np.sqrt(photdata['fluxcalerr'][selectFilter]**2. + (0*modelerr)**2. + colorerr**2.)

			
			if computeDerivatives:
				#d model / dc is total flux (M0 and M1 components (already modulated with passband)) times the color law and a factor of ln(10)
				#import pdb; pdb.set_trace()
				photresultsdict['dmodelflux_dc'][selectFilter,0]=np.sum((modulatedFlux)*np.log(10)*colorlaw[np.newaxis,idx[flt]], axis=1)*dwave*self.fluxfactor[survey][flt]
				#empderiv = np.sum((modulatedFlux/(10. ** (colorlaw * c))*10. ** (colorlaw * (c+0.001)) - modulatedFlux)/1.0e-3, axis=1)*dwave*self.fluxfactor[survey][flt]
				#import pdb; pdb.set_trace()
				
				for i in range(self.n_colorpars):
					#Color law is linear wrt to the color law parameters; therefore derivative of the color law
					# with respect to color law parameter i is the color law with all other values zeroed minus the color law with all values zeroed
					dcolorlaw_dcli = SALT2ColorLaw(self.colorwaverange, np.arange(self.n_colorpars)==i)(self.wave[idx[flt]])-SALT2ColorLaw(self.colorwaverange, np.zeros(self.n_colorpars))(self.wave[idx[flt]])
					#Multiply M0 and M1 components (already modulated with passband) by c* d colorlaw / d cl_i, with associated normalizations
					photresultsdict['dmodelflux_dcl'][selectFilter,i] = np.sum((modulatedFlux)*-0.4*np.log(10)*c*dcolorlaw_dcli[np.newaxis,:], axis=1)*dwave*self.fluxfactor[survey][flt]
					
				if computePCDerivs:
					passbandColorExp=pbspl[flt]*colorexp[idx[flt]]
					intmult = dwave*self.fluxfactor[survey][flt]*_SCALE_FACTOR/(1+z)*x0
					for i in range(len(self.im0)):
						#Range of wavelength and phase values affected by changes in knot i
						waverange=self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
						phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-self.bsorder-1),i//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
						#Check if this filter is inside values affected by changes in knot i
						if waverange[0]*(1+z) > self.kcordict[survey][flt]['maxlam'] or waverange[1]*(1+z) < self.kcordict[survey][flt]['minlam']:
							pass
						#Check which phases are affected by knot i
						inPhase=(phase>phaserange[0]*(1+z) ) & (phase<phaserange[1]*(1+z) )
						if inPhase.any():
							#Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
							#Integrate only over wavelengths within the relevant range
							inbounds=(self.wave[idx[flt]]>waverange[0])	 & (self.wave[idx[flt]]<waverange[1])
							derivInterp = interp1d(obsphase/(1+z),self.spline_derivs[i][:,inbounds],axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")

							photresultsdict['dmodelflux_dM0'][np.where(selectFilter)[0][inPhase],i] =  \
								np.sum( passbandColorExp[:,inbounds] * derivInterp(phase[inPhase]/(1+z)), axis=1)*\
								intmult

							#Dependence is the same for dM1, except with an extra factor of x1
							photresultsdict['dmodelflux_dM1'][np.where(selectFilter)[0][inPhase],i] =  photresultsdict['dmodelflux_dM0'][selectFilter,i][inPhase]*x1
					if ( (phase>obsphase.max())).any():
						iMax = np.where(phase>obsphase.max())[0]
						for i in iMax:
							photresultsdict['dmodelflux_dM0'][np.where(selectFilter)[0][i],:] *= \
								10**(-0.4*self.extrapolateDecline*(phase[i]-obsphase.max()))
							photresultsdict['dmodelflux_dM1'][np.where(selectFilter)[0][i],:] *= \
								10**(-0.4*self.extrapolateDecline*(phase[i]-obsphase.max()))
			
		return photresultsdict,specresultsdict


	def derivativesforPrior(self,x,components,colorLaw):

		passbandColorExp = self.kcordict['default']['Bpbspl'] #*10**(-0.4*colorLaw(self.wave))
		intmult = (self.wave[1] - self.wave[0])*self.kcordict['default']['fluxfactor']
		xtmp = copy.deepcopy(x)
		m0B = self.m0prior(components,m0=True)
		for i,h,j in zip(self.im0,self.im1,range(len(self.im0))):
			xtmp[i] += 1.0e-3; xtmp[h] += 1.0e-3
			modcomp = self.SALTModel(xtmp)
			xtmp[i] -= 1.0e-3; xtmp[h] -= 1.0e-3
			
			waverange=self.waveknotloc[[i%(self.waveknotloc.size-self.bsorder-1),i%(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
			phaserange=self.phaseknotloc[[i//(self.waveknotloc.size-self.bsorder-1),i//(self.waveknotloc.size-self.bsorder-1)+self.bsorder+1]]
			#Check if this filter is inside values affected by changes in knot i
			if waverange[0] > self.kcordict['default']['maxlam'] or waverange[1] < self.kcordict['default']['minlam']:
				pass
			#Check which phases are affected by knot i
			inPhase=(0>phaserange[0] ) & (0<phaserange[1])
			if inPhase.any():
				#Bisplev with only this knot set to one, all others zero, modulated by passband and color law, multiplied by flux factor, scale factor, dwave, redshift, and x0
				#Integrate only over wavelengths within the relevant range
				inbounds=(self.wave>waverange[0]) & (self.wave<waverange[1])
				derivInterp = interp1d(self.phase,self.spline_derivs[i][:,inbounds],axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")

				m0Bmod = self.m0prior(self.modcomp[j],m0=True)
				self.priorderivdict['Bmax'][i] = (m0Bmod-m0B)/1.0e-3/0.01
				#\
				#	np.sum( passbandColorExp[inbounds] * derivInterp(0))*\
				#	intmult/0.01

			self.priorderivdict['M0end'][i] = (self.m0endprior_lsq(modcomp)-self.m0endprior_lsq(components))/1.0e-3
			self.priorderivdict['M1end'][h] = (self.m1endprior_lsq(modcomp)-self.m1endprior_lsq(components))/1.0e-3
		#import pdb; pdb.set_trace()
			
		#colorpars = x[self.parlist == 'cl']			
		#for i,j in zip(self.iCL,range(self.n_colorpars)):
		#	colorpars[j] += 1.0e-3
		#	modcolorLaw = SALT2ColorLaw(self.colorwaverange, colorpars)
		#	colorpars[j] -= 1.0e-3
		#	self.priorderivdict['colorlaw'][i] = (self.EBVprior(modcolorLaw)-self.EBVprior(colorLaw))/1.0e-3
			
		#x1tmp = np.zeros(self.nsn)
		#for sn,i in zip(self.datadict.keys(),range(len(self.datadict.keys()))):
		#	x1tmp[i] = x[self.parlist == 'x1_%s'%sn] # - np.mean(x[self.ix1])
		#x1 = copy.deepcopy(x1tmp)

		x1tmp = x[self.ix1]
		for sn,i in zip(self.datadict.keys(),range(len(self.datadict.keys()))):

			x1tmp[i] += 1.0e-4
			self.priorderivdict['x1std'][self.parlist == 'x1_%s'%sn] = (np.std(x1tmp)/0.01-np.std(x[self.ix1])/0.01)/1.0e-4
			self.priorderivdict['x1mean'][self.parlist == 'x1_%s'%sn] = (np.mean(x1tmp)/0.01-np.mean(x[self.ix1])/0.01)/1.0e-4
			x1tmp[i] -= 1.0e-4



	def loglikeforSN(self,args,sn=None,x=None,components=None,salterr=None,
					 colorLaw=None,colorScat=None,
					 debug=False,timeit=False):
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
		if self.n_components == 1: M0 = copy.deepcopy(components[0])
		elif self.n_components == 2: M0,M1 = copy.deepcopy(components)

		photResidsDict,specResidsDict = self.ResidsForSN(x,sn,components,colorLaw,False)
		

		loglike= - (photResidsDict['photresid']**2).sum() / 2.   -(specResidsDict['specresid']**2).sum()/2.+photResidsDict['lognorm']+specResidsDict['lognorm']
		
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

		try: errpars = x[self.errmin:self.errmax]
		except: import pdb; pdb.set_trace()

		modelerr = bisplev(self.phase if evaluatePhase is None else evaluatePhase,
						   self.wave if evaluateWave is None else evaluateWave,
						   (self.errphaseknotloc,self.errwaveknotloc,errpars,self.bsorder,self.bsorder))

		return modelerr

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
		self.neff=np.zeros((self.phasebins.size-1,self.wavebins.size-1))
		for sn in self.datadict.keys():
			tpkoff=x[self.parlist == 'tpkoff_%s'%sn]
			photdata = self.datadict[sn]['photdata']
			specdata = self.datadict[sn]['specdata']
			survey = self.datadict[sn]['survey']
			filtwave = self.kcordict[survey]['filtwave']
			z = self.datadict[sn]['zHelio']

			#For each spectrum, add one point to each bin for every spectral measurement in that bin
			for k in specdata.keys():
				restWave=specdata[k]['wavelength']/(1+z)
				restWave=restWave[(restWave>self.wavebins.min())&(restWave<self.wavebins.max())]
				phase=(specdata[k]['tobs']+tpkoff)/(1+z)
				phaseIndex=np.clip(np.searchsorted(self.phasebins,phase)-1,0,self.phasebins.size-2)[0]
				waveIndices=np.clip(np.searchsorted(self.wavebins,restWave)-1,0,self.wavebins.size-2)
				self.neff[phaseIndex][waveIndices]+=1			
			
			#For each photometric filter, weight the contribution by  
			for flt in np.unique(photdata['filt']):
				g = (self.wavebins[:-1]	 >= filtwave[0]/(1+z)) & (self.wavebins[1:] <= filtwave[-1]/(1+z))	# overlap range
				if flt+'-neffweighting' in self.datadict[sn]:
					pbspl=self.datadict[sn][flt+'-neffweighting']
				else:
					filttrans = self.kcordict[survey][flt]['filttrans']
					pbspl = np.zeros(g.sum())
					for i in range(g.sum()):
						j=np.where(g)[0][i]
						pbspl[i]=trapIntegrate(self.wavebins[j],self.wavebins[j+1],filtwave/(1+z),filttrans*filtwave/(1+z))
					#Normalize it so that total number of points added is 1
					pbspl /= np.sum(pbspl)
					self.datadict[sn][flt+'-neffweighting']=pbspl
				#Couple of things going on: -1 to ensure everything lines up, clip to put extrapolated points on last phasebin
				phaseindices=np.clip(np.searchsorted(self.phasebins,(photdata['tobs'][(photdata['filt']==flt)]+tpkoff)/(1+z))-1,0,self.phasebins.size-2)
				#Consider weighting neff by variance for each measurement?
				self.neff[phaseindices,:][:,g]+=pbspl[np.newaxis,:]

		#Smear it out a bit along phase axis
		self.neff=gaussian_filter1d(self.neff,1,0)

		self.neff=np.clip(self.neff,1e-2*self.neff.max(),None)
		self.plotEffectivePoints([-12.5,0,12.5,40],'neff.png')
		self.plotEffectivePoints(None,'neff-heatmap.png')

	def plotEffectivePoints(self,phases=None,output=None):

		import matplotlib.pyplot as plt
		if phases is None:
			plt.imshow(self.neff,cmap='Greys',aspect='auto')
			xticks=np.linspace(0,self.wavebins.size,8,False)
			plt.xticks(xticks,['{:.0f}'.format(self.wavebins[int(x)]) for x in xticks])
			plt.xlabel('$\lambda$ / Angstrom')
			yticks=np.linspace(0,self.phasebins.size,8,False)
			plt.yticks(yticks,['{:.0f}'.format(self.phasebins[int(x)]) for x in yticks])
			plt.ylabel('Phase / days')
		else:
			inds=np.searchsorted(self.phasebins,phases)
			for i in inds:
				plt.plot(self.wavebins[:-1],self.neff[i,:],label='{:.1f} days'.format(self.phasebins[i]))
			plt.ylabel('$N_eff$')
			plt.xlabel('$\lambda (\AA)$')
			plt.xlim(self.wavebins.min(),self.wavebins.max())
			plt.legend()
		
		if output is None:
			plt.show()
		else:
			plt.savefig(output,dpi=288)
		plt.clf()


	def dyadicRegularization(self,x):
				
		phase=(self.phasebins[:-1]+self.phasebins[1:])/2
		wave=(self.wavebins[:-1]+self.wavebins[1:])/2
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdwave=self.SALTModelDeriv(x,0,1,phase,wave)
		dfluxdphase=self.SALTModelDeriv(x,1,0,phase,wave)
		d2fluxdphasedwave=self.SALTModelDeriv(x,1,1,phase,wave)
=		resids=[]
		jac=[]
		for i in range(len(fluxes)):
			#Determine a scale for the fluxes by sum of squares (since x1 can have negative values)
			scale=np.sqrt(np.mean(fluxes[i]**2))
			#Derivative of scale with respect to model parameters
			scaleDeriv= np.mean(fluxes[i][:,:,np.newaxis]*self.regularizationDerivs[0],axis=(0,1))/scale
			#Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
			normalization=np.sqrt(1/( (self.wavebins.size-1) *(self.phasebins.size-1)))
			#0 if model is locally separable in phase and wavelength i.e. flux=g(phase)* h(wavelength) for arbitrary functions g and h
			numerator=(dfluxdphase[i] *dfluxdwave[i] -d2fluxdphasedwave[i] *fluxes[i] )
			dnumerator=( self.regularizationDerivs[1]*dfluxdwave[i][:,:,np.newaxis] + self.regularizationDerivs[2]* dfluxdphase[i][:,:,np.newaxis] - self.regularizationDerivs[3]* fluxes[i][:,:,np.newaxis] - self.regularizationDerivs[0]* d2fluxdphasedwave[i][:,:,np.newaxis] )
			
			resids += [normalization* (numerator / (scale**2 * np.sqrt( self.neff ))).flatten()]
			jac    += [normalization* (dnumerator*scale**2 - scaleDeriv[np.newaxis,np.newaxis,:]*2*scale*numerator[:,:,np.newaxis] / (scale**4 * np.sqrt( self.neff )[:,:,np.newaxis])).reshape(-1, self.im0.size)]
		return resids,jac
		
	def gradientRegularization(self, x):
		
		phase=(self.phasebins[:-1]+self.phasebins[1:])/2
		wave=(self.wavebins[:-1]+self.wavebins[1:])/2
		fluxes=self.SALTModel(x,evaluatePhase=phase,evaluateWave=wave)
		dfluxdwave=self.SALTModelDeriv(x,0,1,phase,wave)
		waveGradResids=[]
		waveGradJac=[]
		for i in range(len(fluxes)):
			import pdb;pdb.set_trace()
			#Determine a scale for the fluxes by sum of squares (since x1 can have negative values)
			scale=np.sqrt(np.mean(fluxes[i]**2))
			#Normalize gradient by flux scale
			normedGrad=dfluxdwave[i]/scale
			#Derivative of scale with respect to model parameters
			scaleDeriv= np.mean(fluxvals[:,:,np.newaxis]*self.regularizationDerivs[0],axis=(0,1))
			#Derivative of normalized gradient with respect to model parameters
			normedGradDerivs=(self.regularizationDerivs[2] * scale - scaleDeriv[np.newaxis,np.newaxis,:]*dfluxdwave[i][:,:,np.newaxis])/ scale**2
			#Normalization (divided by total number of bins so regularization weights don't have to change with different bin sizes)
			normalization=np.sqrt(1/((self.wavebins.size-1) *(self.phasebins.size-1)))
			#Minimize model derivative w.r.t wavelength in unconstrained regions
			waveGradResids+= [normalization* ( normedGrad /	np.sqrt( self.neff )).flatten()]
			waveGradJac+= [normalization*((normedGradDerivs) / np.sqrt( self.neff )[:,:,np.newaxis]).reshape(-1, self.im0.size)]
		return waveGradResids,waveGradJac
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
