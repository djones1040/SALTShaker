import unittest,os
from salt3.training import saltresids
from salt3.training.saltresids import _SCALE_FACTOR

from salt3.training.TrainSALT import TrainSALT
import numpy as np
import pickle, argparse, configparser,warnings
from salt3.util import snana,readutils
from sncosmo.salt2utils import SALT2ColorLaw
from scipy.interpolate import interp1d
import os
from salt3.config import config_rootdir

class TRAINING_Test(unittest.TestCase):
	
	
	
	def setUp(self):
		ts=TrainSALT()

		config = configparser.ConfigParser()
		config.read('testdata/test.conf')

		user_parser = ts.add_user_options(usage='',config=config)
		user_options = user_parser.parse_known_args()[0]
		
		trainingconfig = configparser.ConfigParser()
		trainingconfig.read(user_options.trainingconfig)

		training_parser = ts.add_training_options(
			usage='',config=trainingconfig)
		training_options = training_parser.parse_known_args(namespace=user_options)[0]
		
		ts.options=training_options
		kcordict=readutils.rdkcor(ts.surveylist,ts.options,addwarning=ts.addwarning)

		#if not os.path.exists(user_options.trainingconfig):
		#	print('warning : training config file %s doesn\'t exist.  Trying package directory'%user_options.trainingconfig)
		#	user_options.trainingconfig = '%s/%s'%(config_rootdir,user_options.trainingconfig)
		#if not os.path.exists(user_options.trainingconfig):
		#	raise RuntimeError('can\'t find training config file!  Checked %s'%user_options.trainingconfig)
	
		#ts.kcordict=kcordict

		# TODO: ASCII filter files
		# read the data
		datadict = readutils.rdAllData(ts.options.snlist,ts.options.estimate_tpk,kcordict,
									   ts.addwarning,dospec=ts.options.dospec)
		ts.kcordict=kcordict
		self.parlist,self.guess,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc = ts.initialParameters(datadict)
		saltfitkwargs = ts.get_saltkw(phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)
				 
		self.resids = saltresids.SALTResids(self.guess,datadict,self.parlist,**saltfitkwargs)	
		
		sn=self.resids.datadict.keys().__iter__().__next__()
		self.guess[self.parlist == 'x0_%s'%sn]=1.31e-4
		self.guess[self.parlist == 'x1_%s'%sn]=0.7707
		self.guess[self.parlist == 'c_%s'%sn]=0.0514
		self.guess[self.parlist == 'tpkoff_%s'%sn] = 0

		
	def defineStoredResults(self,guess,defineInterpolations=True):
		sn=self.resids.datadict.keys().__iter__().__next__()
		
		components = self.resids.SALTModel(guess)
		componentderivs = self.resids.SALTModelDeriv(guess,1,0,self.resids.phase,self.resids.wave)
		salterr = self.resids.ErrModel(guess)
		colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.resids.parlist == 'cl'])
		
		M0,M1=self.resids.SALTModel(guess)
		M0deriv,M1deriv=self.resids.SALTModelDeriv(guess,1,0,self.resids.phase,self.resids.wave)
		colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])

		z = self.resids.datadict[sn]['zHelio']
		obsphase = self.resids.datadict[sn]['obsphase'] #self.phase*(1+z)
		x0,x1,c,tpkoff = guess[self.parlist == 'x0_%s'%sn],guess[self.parlist == 'x1_%s'%sn],\
						 guess[self.parlist == 'c_%s'%sn],guess[self.parlist == 'tpkoff_%s'%sn]

		#Apply MW extinction
		M0 *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
		M1 *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]

		M0deriv *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
		M1deriv *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
	
		colorlaw = -0.4 * colorLaw(self.resids.wave)
		colorexp = 10. ** (colorlaw * c)
		M0 *= colorexp; M1 *= colorexp
		M0 *= _SCALE_FACTOR/(1+z); M1 *= _SCALE_FACTOR/(1+z)

		int1dM0 = interp1d(obsphase,M0,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)
		int1dM1 = interp1d(obsphase,M1,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)

		# derivs
		M0deriv *= colorexp; M1deriv *= colorexp
		M0deriv *= _SCALE_FACTOR/(1+z); M1deriv *= _SCALE_FACTOR/(1+z)
		
		mod = x0*(M0 + x1*M1)
		int1d = interp1d(obsphase,mod,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)
		
		modderiv = x0*(M0deriv + x1*M1deriv)
		int1dderiv = interp1d(obsphase,modderiv,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)
		
		saltErr=self.resids.ErrModel(guess)
		saltCorr=self.resids.CorrelationModel(guess)
		
		prefactor=(self.resids.datadict[sn]['mwextcurve'] *colorexp*  _SCALE_FACTOR/(1+z))
		interr1d = [interp1d(obsphase,err * prefactor ,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True) for err in saltErr ]
		intcorr1d= [interp1d(obsphase,corr ,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True) for corr in saltCorr ]

		modelUncertainty= x0**2 * prefactor**2 *(saltErr[0]**2  + 2*x1* saltCorr[0]*saltErr[0]*saltErr[1] + x1**2 *saltErr[1]**2)
		uncertaintyInterp=interp1d(obsphase,modelUncertainty ,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)

		storedResults={}
		storedResults['components']=components
		storedResults['componentderivs']=componentderivs
		storedResults['saltErr']=saltErr
		storedResults['saltCorr']=saltCorr
		storedResults['colorLaw']=colorlaw
		storedResults['colorLawInterp']= interp1d(self.resids.wave,storedResults['colorLaw'],kind=self.resids.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
		if defineInterpolations:
			storedResults['fluxInterp_{}'.format(sn)]=int1d
			storedResults['componentsInterp_{}'.format(sn)]=int1dM0,int1dM1
			storedResults['phaseDerivInterp_{}'.format(sn)]=int1dderiv
			storedResults['colorexp_{}'.format(sn)]=colorexp
			storedResults['uncertaintyComponentsInterp_{}'.format(sn)]=interr1d
			storedResults['uncertaintyCorrelationsInterp_{}'.format(sn)]=intcorr1d
			storedResults['modelUncertaintyInterp_{}'.format(sn)]=uncertaintyInterp
	
		return storedResults
	
	def test_prior_jacobian(self):
		"""Checks that all designated priors are properly calculating the jacobian of their residuals to within 1%"""
				#Define simple models for m0,m1

		for prior in self.resids.priors.priors:

			print('Testing prior', prior)
			components=self.resids.SALTModel(self.guess)
			resid,val,jacobian=self.resids.priors.priors[prior](0.3145,self.guess,components)
			self.assertTrue(self.resids.priors.priors[prior].numResids==resid.size)
			dx=1e-3
			rtol=1e-2
			def incrementOneParam(i):
				guess=self.guess.copy()
				guess[i]+=dx
				components=self.resids.SALTModel(guess)
				return self.resids.priors.priors[prior](0.3145,guess,components)[0]
			dPriordX=np.zeros((resid.size,self.guess.size))
			for i in range(self.guess.size):
				dPriordX[:,i]=(incrementOneParam(i)-resid)/dx

			#Check that all derivatives that should be 0 are zero
			if  not np.allclose(jacobian,dPriordX,rtol): print('Problems with derivatives for prior {} : '.format(prior),np.unique(self.parlist[np.where(~np.isclose(jacobian,dPriordX,rtol))]))
			self.assertTrue(np.all((dPriordX==0)==(jacobian==0)))
			self.assertTrue(np.allclose(jacobian,dPriordX,rtol))

	def test_photresid_jacobian(self):
		"""Checks that the the jacobian of the photometric residuals is being correctly calculated to within 1%"""
		print("Checking photometric derivatives")
		dx=1e-3
		rtol=1e-2
		sn=self.resids.datadict.keys().__iter__().__next__()
		
		storedResults=self.defineStoredResults(self.guess,defineInterpolations=False)
		#import pdb;pdb.set_trace()
		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,storedResults,varyParams=np.zeros(self.resids.npar,dtype=bool),fixUncertainty=True)
		residuals = photresidsdict['resid']
		
		uncertainties={key:storedResults[key] for key in storedResults if key.startswith('photvariances_') or key.startswith('specvariances_') or key.startswith('photCholesky_') }
		storedResults=self.defineStoredResults(self.guess,defineInterpolations=False)
		storedResults.update(uncertainties)

		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,storedResults,varyParams=np.ones(self.resids.npar,dtype=bool),fixUncertainty=True)
		jacobian=photresidsdict['resid_jacobian']
		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			storedResults=self.defineStoredResults(guess,defineInterpolations=False)
			storedResults.update(uncertainties)
			return self.resids.ResidsForSN(guess,sn,storedResults ,np.zeros(self.resids.npar,dtype=bool),fixUncertainty=True)[0]['resid']

		dResiddX=np.zeros((residuals.size,self.parlist.size))
		for i in range(self.guess.size):
			dResiddX[:,i]=(incrementOneParam(i,dx/2)-incrementOneParam(i,-dx/2))/dx

		if not np.allclose(jacobian,dResiddX,rtol): print('Problems with derivatives: ',np.unique(self.parlist[np.where(~np.isclose(jacobian,dResiddX,rtol))[1]]))
		self.assertTrue(np.allclose(jacobian,dResiddX,rtol))

	def test_photresid_jacobian_vary_uncertainty(self):
		"""Check that lognorm gradient and residuals' jacobian are correctly calculated with fixUncertainty=False"""
		print("Checking photometric derivatives including uncertainties")
		dx=3e-8
		rtol=1e-2
		atol=1e-4
		sn=self.resids.datadict.keys().__iter__().__next__()
		components = self.resids.SALTModel(self.resids.guess)
		componentderivs = self.resids.SALTModelDeriv(self.resids.guess,1,0,self.resids.phase,self.resids.wave)
		salterr = self.resids.ErrModel(self.resids.guess)
		saltcorr=saltcorr=self.resids.CorrelationModel(self.resids.guess)
		colorLaw = SALT2ColorLaw(self.resids.colorwaverange, self.resids.guess[self.resids.parlist == 'cl'])
		

		residsdict=self.resids.ResidsForSN(self.guess,sn,components,componentderivs,colorLaw,salterr,saltcorr,True,True)[0]
		grad=residsdict['lognorm_grad']
		jacobian=residsdict['resid_jacobian']

		residsdict=self.resids.ResidsForSN(self.guess,sn,components,componentderivs,colorLaw,salterr,saltcorr,False,False)[0]

		residuals = residsdict['resid']
		lognorm= residsdict['lognorm']

		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			components=self.resids.SALTModel(guess)
			componentderivs = self.resids.SALTModelDeriv(guess,1,0,self.resids.phase,self.resids.wave)
			
			colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])
			salterr=self.resids.ErrModel(guess)
			saltcorr=self.resids.CorrelationModel(guess)
			return self.resids.ResidsForSN(guess,sn,components,componentderivs,colorLaw,salterr,saltcorr,False)[0]

		dResiddX=np.zeros((residuals.size,self.parlist.size))
		dLognormdX=np.zeros(self.parlist.size)
		for i in range(self.guess.size):
			residsdict=incrementOneParam(i)
			dResiddX[:,i]=(residsdict['resid']-residuals)/dx
			dLognormdX[i]=(residsdict['lognorm']-lognorm)/dx

		if not np.allclose(jacobian,dResiddX,rtol,atol): print('Problems with residual derivatives: ',np.unique(self.parlist[np.where(~np.isclose(jacobian,dResiddX,rtol,atol))[1]]))
		if not np.allclose(grad,dLognormdX,rtol,atol): print('Problems with lognorm derivatives: ',np.unique(self.parlist[np.where(~np.isclose(grad,dLognormdX,rtol,atol))[0]]))
		self.assertTrue(np.allclose(jacobian,dResiddX,rtol,atol))
		self.assertTrue(np.allclose(grad,dLognormdX,rtol,atol))

	def test_photvals_jacobian(self):
		"""Check that the model is correctly calculating the jacobian of the photometry"""
		print("Checking photometric value derivatives")
		dx=1e-3
		rtol=1e-2
		atol=1e-8
		storedResults=self.defineStoredResults(self.guess)
		sn=self.resids.datadict.keys().__iter__().__next__()
		
		valsdict=self.resids.photValsForSN(self.guess,sn, storedResults,np.ones(self.guess.size,dtype=bool))
		jacobian=valsdict['modelflux_jacobian']

		valsdict=self.resids.photValsForSN(self.guess,sn, storedResults,np.zeros(self.guess.size,dtype=bool))
		vals = valsdict['modelflux']
		
		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			storedResults=self.defineStoredResults(guess)		
			return self.resids.photValsForSN(guess,sn, storedResults,np.zeros(self.guess.size,dtype=bool))
			
		dValdX=np.zeros((vals.size,self.parlist.size))
		for i in range(self.guess.size):
			valsdict=incrementOneParam(i)
			dValdX[:,i]=(valsdict['modelflux']-vals)/dx
		
		import pdb;pdb.set_trace()
		if not np.allclose(dValdX,jacobian,rtol,atol): print('Problems with model value derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dValdX,jacobian,rtol,atol))[1]]))

		self.assertTrue(np.allclose(dValdX,jacobian,rtol,atol))

	def test_photuncertainty_jacobian(self):
		"""Check that the model is correctly calculating the jacobian of the photometric model uncertainties"""
		print("Checking photometric uncertainty derivatives")
		dx=1e-8
		rtol=1e-2
		atol=1e-10
		storedResults=self.defineStoredResults(self.guess)
		sn=self.resids.datadict.keys().__iter__().__next__()
		
		uncertaintydict=self.resids.photVarianceForSN(self.guess,sn,storedResults,np.ones(self.guess.size,dtype=bool))
		uncJac=	uncertaintydict['modelvariance_jacobian']

		uncertaintydict=self.resids.photVarianceForSN(self.guess,sn,storedResults,np.zeros(self.guess.size,dtype=bool))
		uncertainty = uncertaintydict['modelvariance']
		
		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			newResults=self.defineStoredResults(guess)
			return self.resids.photVarianceForSN(guess,sn,newResults,np.zeros(self.guess.size,dtype=bool))
		
		
		dUncertaintydX=np.zeros((uncertainty.size,self.parlist.size))
		for i in range(self.guess.size):
			uncertaintydict=incrementOneParam(i,dx)
			dUncertaintydX[:,i]=(uncertaintydict['modelvariance']-uncertainty)/dx

		if not np.allclose(uncJac,dUncertaintydX,rtol,atol): print('Problems with model uncertainty derivatives: ',np.unique(self.parlist[np.where(~np.isclose(uncJac,dUncertaintydX,rtol,atol))[1]]))
		self.assertTrue(np.allclose(uncJac,dUncertaintydX,rtol,atol))

	def test_specuncertainty_jacobian(self):
		"""Check that jacobian of spectral variance is correct"""
		print("Checking spectral variance derivatives")
		dx=1e-8
		rtol=1e-2
		atol=1e-48
		storedResults=self.defineStoredResults(self.guess)
		sn=self.resids.datadict.keys().__iter__().__next__()
		
		uncertaintydict=self.resids.specVarianceForSN(self.guess,sn,storedResults,np.ones(self.guess.size,dtype=bool))
		uncJac=	uncertaintydict['modelvariance_jacobian']

		uncertaintydict=self.resids.specVarianceForSN(self.guess,sn,storedResults,np.zeros(self.guess.size,dtype=bool))
		uncertainty = uncertaintydict['modelvariance']
		
		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			newResults=self.defineStoredResults(guess)
			return self.resids.specVarianceForSN(guess,sn,newResults,np.zeros(self.guess.size,dtype=bool))
		

		dUncertaintydX=np.zeros((uncertainty.size,self.parlist.size))
		for i in range(self.guess.size):
			uncertaintydict=incrementOneParam(i,dx)
			dUncertaintydX[:,i]=(uncertaintydict['modelvariance']-uncertainty)/dx
		if not np.allclose(uncJac,dUncertaintydX,rtol,atol): print('Problems with model variance derivatives: ',np.unique(self.parlist[np.where(~np.isclose(uncJac,dUncertaintydX,rtol,atol))[1]]))

		self.assertTrue(np.allclose(uncJac,dUncertaintydX,rtol,atol))


	def test_specvals_jacobian(self):
		"""Check that lognorm gradient and residuals' jacobian are correctly calculated with fixUncertainty=False"""
		print("Checking spectral derivatives")
		dx=1e-8
		rtol=1e-2
		atol=1e-20
		sn=self.resids.datadict.keys().__iter__().__next__()

		storedResults=self.defineStoredResults(self.guess)
		
		valsdict=self.resids.specValsForSN(self.resids.guess,sn, storedResults,np.ones(self.guess.size,dtype=bool))
		jacobian=valsdict['modelflux_jacobian']

		valsdict=self.resids.specValsForSN(self.resids.guess,sn, storedResults,np.zeros(self.guess.size,dtype=bool))
		vals = valsdict['modelflux']
		
		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			storedResults=self.defineStoredResults(guess)
			return self.resids.specValsForSN(guess,sn, storedResults,np.zeros(self.guess.size,dtype=bool))
		dValdX=np.zeros((vals.size,self.parlist.size))
		for i in range(self.guess.size):
			valsdict=incrementOneParam(i)
			dValdX[:,i]=(valsdict['modelflux']-vals)/dx

		if not np.allclose(jacobian,dValdX,rtol,atol): print('Problems with model value derivatives: ',np.unique(self.parlist[np.where(~np.isclose(jacobian,dValdX,rtol,atol))[1]]))

		self.assertTrue(np.allclose(jacobian,dValdX,rtol,atol))

	def test_specresid_jacobian(self):
		"""Checks that the the jacobian of the spectroscopic residuals is being correctly calculated to within 1%"""
		print("Checking spectral derivatives")
		dx=1e-8
		rtol=1e-2
		atol=1e-4
		dx=1e-3
		rtol=1e-2
		sn=self.resids.datadict.keys().__iter__().__next__()
		
		storedResults=self.defineStoredResults(self.guess,defineInterpolations=False)
		#import pdb;pdb.set_trace()
		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,storedResults,varyParams=np.zeros(self.resids.npar,dtype=bool),fixUncertainty=True)
		residuals = specresidsdict['resid']
		
		uncertainties={key:storedResults[key] for key in storedResults if key.startswith('photvariances_') or key.startswith('specvariances_') or key.startswith('photCholesky_') }
		storedResults=self.defineStoredResults(self.guess,defineInterpolations=False)
		storedResults.update(uncertainties)

		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,storedResults,varyParams=np.ones(self.resids.npar,dtype=bool),fixUncertainty=True)
		jacobian=specresidsdict['resid_jacobian']
		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			storedResults=self.defineStoredResults(guess,defineInterpolations=False)
			storedResults.update(uncertainties)
			return self.resids.ResidsForSN(guess,sn,storedResults ,np.zeros(self.resids.npar,dtype=bool),fixUncertainty=True)[1]['resid']

		dResiddX=np.zeros((residuals.size,self.parlist.size))
		for i in range(self.guess.size):
			dResiddX[:,i]=(incrementOneParam(i,dx/2)-incrementOneParam(i,-dx/2))/dx

		if not np.allclose(jacobian,dResiddX,rtol): print('Problems with derivatives: ',np.unique(self.parlist[np.where(~np.isclose(jacobian,dResiddX,rtol))[1]]))
		self.assertTrue(np.allclose(jacobian,dResiddX,rtol))

	def test_specresid_jacobian_vary_uncertainty(self):
		"""Check that lognorm gradient and residuals' jacobian are correctly calculated with fixUncertainty=False"""
		print("Checking spectral resid derivatives")
		dx=1e-8
		rtol=1e-2
		atol=1e-2
		sn=self.resids.datadict.keys().__iter__().__next__()
		components = self.resids.SALTModel(self.resids.guess)
		componentderivs = self.resids.SALTModelDeriv(self.resids.guess,1,0,self.resids.phase,self.resids.wave)
		salterr = self.resids.ErrModel(self.resids.guess)
		saltcorr=self.resids.CorrelationModel(self.resids.guess)
		colorLaw = SALT2ColorLaw(self.resids.colorwaverange, self.resids.guess[self.resids.parlist == 'cl'])
		
		specresidsdict=self.resids.ResidsForSN(self.guess,sn,components,componentderivs,colorLaw,salterr,saltcorr,True,True)[1]
		grad=specresidsdict['lognorm_grad']
		jacobian=specresidsdict['resid_jacobian']

		specresidsdict=self.resids.ResidsForSN(self.guess,sn,components,componentderivs,colorLaw,salterr,saltcorr,False,False)[1]

		residuals = specresidsdict['resid']
		lognorm= specresidsdict['lognorm']

		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			components=self.resids.SALTModel(guess)
			componentderivs = self.resids.SALTModelDeriv(guess,1,0,self.resids.phase,self.resids.wave)
			colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])
			salterr=self.resids.ErrModel(guess)

			saltcorr=self.resids.CorrelationModel(guess)
			return self.resids.ResidsForSN(guess,sn,components,componentderivs,colorLaw,salterr,saltcorr,False)[1]

		dResiddX=np.zeros((residuals.size,self.parlist.size))
		dLognormdX=np.zeros(self.parlist.size)
		for i in range(self.guess.size):
			residsdict=incrementOneParam(i,dx)
			dResiddX[:,i]=(residsdict['resid']-residuals)/dx
			dLognormdX[i]=(residsdict['lognorm']-lognorm)/dx
		if not np.allclose(jacobian,dResiddX,rtol,atol): print('Problems with residual derivatives: ',np.unique(self.parlist[np.where(~np.isclose(jacobian,dResiddX,rtol,atol))[1]]))
		if not np.allclose(grad,dLognormdX,rtol,atol): print('Problems with lognorm derivatives: ',np.unique(self.parlist[np.where(~np.isclose(grad,dLognormdX,rtol,atol))[0]]))

		self.assertTrue(np.allclose(jacobian,dResiddX,rtol,atol))
		self.assertTrue(np.allclose(grad,dLognormdX,rtol,atol))
	
	def test_computeDerivatives(self):
		"""Checks that the computeDerivatives parameter of the ResidsForSN function is not affecting the residuals"""
		sn=self.resids.datadict.keys().__iter__().__next__()
		components = self.resids.SALTModel(self.resids.guess)
		componentderivs = self.resids.SALTModelDeriv(self.resids.guess,1,0,self.resids.phase,self.resids.wave)
		salterr = self.resids.ErrModel(self.resids.guess)
		saltcorr=self.resids.CorrelationModel(self.resids.guess)
		if self.resids.n_colorpars:
			colorLaw = SALT2ColorLaw(self.resids.colorwaverange, self.resids.guess[self.resids.parlist == 'cl'])
		else: colorLaw = None
		if self.resids.n_colorscatpars:
			colorScat = True
		else: colorScat = None
		combinations= [(True,True),(True,False),(False,False)]

		results=[self.resids.ResidsForSN(self.guess,sn,components,componentderivs,colorLaw,salterr,saltcorr,*x) for x in combinations]

		first=results[0]
		
		#Check photometric residuals are not affected by computeDerivatives
		self.assertTrue(all([np.allclose(first[0]['resid'],result[0]['resid']) for result in results[1:]]))
		#Check spectroscopic residuals are not affected by computeDerivatives
		self.assertTrue(all([np.allclose(first[1]['resid'],result[1]['resid']) for result in results[1:]]))

	def test_regularization_scale(self):
		dx=1e-6
		rtol=1e-2
		for k in range(2):
			def incrementOneParam(i):
				guess=self.guess.copy()
				guess[i]+=dx
				return self.resids.regularizationScale(self.resids.SALTModel(guess),self.resids.SALTModel(guess,self.resids.phaseRegularizationPoints,self.resids.waveRegularizationPoints))[0][k]
			scale,jac=self.resids.regularizationScale(self.resids.SALTModel(self.guess),self.resids.SALTModel(self.guess,self.resids.phaseRegularizationPoints,self.resids.waveRegularizationPoints))
			scale,jac=scale[k],jac[k]
			dScaledX=np.zeros(self.resids.im0.size) 
			for i,j in enumerate([self.resids.im0,self.resids.im1][k]):
				result=incrementOneParam(j)
				dScaledX[i]=(result-scale)/dx
			self.assertTrue(np.allclose(jac,dScaledX,rtol))
		
	def test_regularization_jacobian(self):
		"""Checks that the the jacobian of the regularization terms is being correctly calculated to within 1%"""
		dx=1e-6
		rtol=1e-2
		for regularization, name in [(self.resids.dyadicRegularization,'dyadic'),(self.resids.phaseGradientRegularization, 'phase gradient'),(self.resids.waveGradientRegularization,'wave gradient' )]:
			for component in range(2):
			
				def incrementOneParam(i):
					guess=self.guess.copy()
					guess[i]+=dx
					return regularization(guess,False)[0][component]

				print('Checking jacobian of {} regularization, {} component'.format(name,component))
				#Only checking the first component, since they're calculated using the same code
				residuals,jacobian=[x[component] for x in regularization(self.guess,True)]
				dResiddX=np.zeros((residuals.size,self.resids.im0.size))
				for i,j in enumerate([self.resids.im0,self.resids.im1][component]):
					dResiddX[:,i]=(incrementOneParam(j)-residuals)/dx
				self.assertTrue(np.allclose(dResiddX,jacobian,rtol))
	
if __name__ == "__main__":
    unittest.main()
