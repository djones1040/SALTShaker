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
		datadict = readutils.rdAllData(ts.options.snlists,ts.options.estimate_tpk,kcordict,
									   ts.addwarning,dospec=ts.options.dospec)
		ts.kcordict=kcordict
		self.parlist,self.guess,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc = ts.initialParameters(datadict)
		saltfitkwargs = ts.get_saltkw(phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)
				 
		self.resids = saltresids.SALTResids(self.guess,datadict,self.parlist,**saltfitkwargs)	
		
		sn=self.resids.datadict.keys().__iter__().__next__()
# 		self.guess[self.parlist == 'x0_%s'%sn]=1.31e-4
# 		self.guess[self.parlist == 'x1_%s'%sn]=0.7707
# 		self.guess[self.parlist == 'c_%s'%sn]=0.0514
# 		self.guess[self.parlist == 'tpkoff_%s'%sn] = 0

		
	def defineStoredResults(self,guess,sn,defineInterpolations=True):
		
		components = self.resids.SALTModel(guess)
		componentderivs = self.resids.SALTModelDeriv(guess,1,0,self.resids.phase,self.resids.wave)
		salterr = self.resids.ErrModel(guess)
		colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.resids.parlist == 'cl'])
		
		colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])
		colorlaw = -0.4 * colorLaw(self.resids.wave)

		
		saltErr=self.resids.ErrModel(guess)
		saltCorr=self.resids.CorrelationModel(guess)

		storedResults={}
		storedResults['components']=components
		storedResults['componentderivs']=componentderivs
		storedResults['saltErr']=saltErr
		storedResults['saltCorr']=saltCorr
		storedResults['colorLaw']=colorlaw
		storedResults['colorLawInterp']= interp1d(self.resids.wave,storedResults['colorLaw'],kind=self.resids.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
		temporaryResults={}
		if defineInterpolations:
			z = self.resids.datadict[sn]['zHelio']
			obsphase = self.resids.datadict[sn]['obsphase'] #self.phase*(1+z)
			x0,x1,c,tpkoff = guess[self.parlist == 'x0_%s'%sn],guess[self.parlist == 'x1_%s'%sn],\
							 guess[self.parlist == 'c_%s'%sn],guess[self.parlist == 'tpkoff_%s'%sn]
			colorexp = 10. ** (colorlaw * c)
			M0,M1=components
			M0deriv,M1deriv=componentderivs

			#Apply MW extinction
			M0 *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
			M1 *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]

			M0deriv *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
			M1deriv *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
	
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
		
			prefactor=(self.resids.datadict[sn]['mwextcurve'] *colorexp*  _SCALE_FACTOR/(1+z))
			interr1d = [interp1d(obsphase,err * prefactor ,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True) for err in saltErr ]
			intcorr1d= [interp1d(obsphase,corr ,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True) for corr in saltCorr ]

			modelUncertainty= x0**2 * prefactor**2 *(saltErr[0]**2  + 2*x1* saltCorr[0]*saltErr[0]*saltErr[1] + x1**2 *saltErr[1]**2)
			uncertaintyInterp=interp1d(obsphase,modelUncertainty ,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)

			temporaryResults['fluxInterp']=int1d
			temporaryResults['componentsInterp']=int1dM0,int1dM1
			temporaryResults['phaseDerivInterp']=int1dderiv
			temporaryResults['colorexp']=colorexp
			temporaryResults['uncertaintyComponentsInterp']=interr1d
			temporaryResults['uncertaintyCorrelationsInterp']=intcorr1d
			temporaryResults['modelUncertaintyInterp']=uncertaintyInterp
	
		return storedResults,temporaryResults
	
	def test_prior_jacobian(self):
		"""Checks that all designated priors are properly calculating the jacobian of their residuals to within 1%"""
				#Define simple models for m0,m1

		for prior in self.resids.priors.priors:
			print('Testing prior', prior)
			components=self.resids.SALTModel(self.guess)
			resid,val,jacobian=self.resids.priors.priors[prior](0.3145,self.guess,components)
			dx=1e-8
			rtol=1e-2
			atol=1e-6
			def incrementOneParam(i,dx):
				guess=self.guess.copy()
				guess[i]+=dx
				components=self.resids.SALTModel(guess)
				return self.resids.priors.priors[prior](0.3145,guess,components)[0]
			dPriordX=np.zeros((resid.size,self.guess.size))
			for i in range(self.guess.size):
				dPriordX[:,i]=(incrementOneParam(i,dx/2)-incrementOneParam(i,-dx/2))/dx
			#Check that all derivatives that should be 0 are zero
			if  not np.allclose(jacobian,dPriordX,rtol,atol): print('Problems with derivatives for prior {} : '.format(prior),np.unique(self.parlist[np.where(~np.isclose(jacobian,dPriordX,rtol,atol))[1]]))
			self.assertTrue(np.allclose(jacobian,dPriordX,rtol,atol))

	def test_photresid_jacobian(self):
		"""Checks that the the jacobian of the photometric residuals is being correctly calculated to within 1%"""
		print("Checking photometric derivatives")
		dx=1e-3
		rtol=1e-2
		atol=1e-8
		sn=self.resids.datadict.keys().__iter__().__next__()
		
		storedResults=self.defineStoredResults(self.guess,sn,defineInterpolations=False)[0]
		#import pdb;pdb.set_trace()
		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,storedResults,varyParams=np.zeros(self.resids.npar,dtype=bool),fixUncertainty=True)
		residuals = photresidsdict['resid']
		
		uncertainties={key:storedResults[key] for key in storedResults if key.startswith('photvariances_') or key.startswith('specvariances_') or key.startswith('photCholesky_') }
		storedResults=self.defineStoredResults(self.guess,sn,defineInterpolations=False)[0]
		storedResults.update(uncertainties)

		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,storedResults,varyParams=np.ones(self.resids.npar,dtype=bool),fixUncertainty=True)
		jacobian=photresidsdict['resid_jacobian']
		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			storedResults=self.defineStoredResults(guess,sn,defineInterpolations=False)[0]
			storedResults.update(uncertainties)
			return self.resids.ResidsForSN(guess,sn,storedResults ,np.zeros(self.resids.npar,dtype=bool),fixUncertainty=True)[0]['resid']

		dResiddX=np.zeros((residuals.size,self.parlist.size))
		for i in range(self.guess.size):
			dResiddX[:,i]=(incrementOneParam(i,dx/2)-incrementOneParam(i,-dx/2))/dx

		if not np.allclose(jacobian,dResiddX,rtol): print('Problems with derivatives: ',np.unique(self.parlist[np.where(~np.isclose(jacobian,dResiddX,rtol))[1]]))
		self.assertTrue((np.isclose(jacobian,dResiddX,rtol,atol).all(axis=0)|((self.parlist==f'tpkoff_{sn}'))).all())

	def test_photresid_jacobian_vary_uncertainty(self):
		"""Check that lognorm gradient and residuals' jacobian are correctly calculated with fixUncertainty=False"""
		print("Checking photometric derivatives including uncertainties")
		dxs=np.ones(self.parlist.size)*1e-6
		dxs[self.resids.iModelParam]=1e-7
		rtol=1e-2
		atol=1e-4
		sn=self.resids.datadict.keys().__iter__().__next__()
		self.guess[self.resids.iclscat[-1]]=-15
		storedResults=self.defineStoredResults(self.guess,sn,defineInterpolations=False)[0]
		#import pdb;pdb.set_trace()
		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,storedResults,varyParams=np.zeros(self.resids.npar,dtype=bool))
		residuals = photresidsdict['resid']
		lognorm=photresidsdict['lognorm']
		storedResults=self.defineStoredResults(self.guess,sn,defineInterpolations=False)[0]
		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,storedResults,varyParams=np.ones(self.resids.npar,dtype=bool))
		jacobian=photresidsdict['resid_jacobian']
		grad=photresidsdict['lognorm_grad']
		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			storedResults=self.defineStoredResults(guess,sn,defineInterpolations=False)[0]
			return self.resids.ResidsForSN(guess,sn,storedResults ,np.zeros(self.resids.npar,dtype=bool))[0]

		dResiddX=np.zeros((residuals.size,self.parlist.size))
		dLognormdX=np.zeros(self.parlist.size)
		for i,dx in enumerate(dxs):
			upper=incrementOneParam(i,dx/2)
			lower=incrementOneParam(i,-dx/2)
			dResiddX[:,i]=(upper['resid']-lower['resid'])/dx
			dLognormdX[i]=(upper['lognorm']-lower['lognorm'])/dx

		if not np.allclose(jacobian,dResiddX,rtol,atol): print('Problems with residual derivatives: ',np.unique(self.parlist[np.where(~np.isclose(jacobian,dResiddX,rtol,atol))[1]]))
		if not np.allclose(grad,dLognormdX,rtol,atol): print('Problems with lognorm derivatives: ',np.unique(self.parlist[np.where(~np.isclose(grad,dLognormdX,rtol,atol))[0]]))
		#import pdb;pdb.set_trace()
		self.assertTrue((np.isclose(jacobian,dResiddX,rtol,atol).all(axis=0)|((self.parlist==f'tpkoff_{sn}'))).all())
		self.assertTrue((np.isclose(grad,dLognormdX,rtol,atol).all(axis=0)|((self.parlist==f'tpkoff_{sn}'))).all())

	def test_photvals_jacobian(self):
		"""Check that the model is correctly calculating the jacobian of the photometry"""
		print("Checking photometric value derivatives")
		dx=1e-3
		rtol=1e-2
		atol=1e-8
		sn=self.resids.datadict.keys().__iter__().__next__()
		storedResults=self.defineStoredResults(self.guess,sn)
		
		valsdict=self.resids.photValsForSN(self.guess,sn, *storedResults,np.ones(self.guess.size,dtype=bool))
		jacobian=valsdict['modelflux_jacobian']

		valsdict=self.resids.photValsForSN(self.guess,sn, *storedResults,np.zeros(self.guess.size,dtype=bool))
		vals = valsdict['modelflux']
		
		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			storedResults=self.defineStoredResults(guess,sn)		
			return self.resids.photValsForSN(guess,sn, *storedResults,np.zeros(self.guess.size,dtype=bool))
			
		dValdX=np.zeros((vals.size,self.parlist.size))
		for i in range(self.guess.size):
			valsdict=incrementOneParam(i)
			dValdX[:,i]=(valsdict['modelflux']-vals)/dx
		
		if not np.allclose(dValdX,jacobian,rtol,atol): print('Problems with model value derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dValdX,jacobian,rtol,atol))[1]]))

		self.assertTrue((np.isclose(dValdX,jacobian,rtol,atol).all(axis=0)|((self.parlist==f'tpkoff_{sn}'))).all())

	def test_photuncertainty_jacobian(self):
		"""Check that the model is correctly calculating the jacobian of the photometric model uncertainties"""
		print("Checking photometric uncertainty derivatives")
		dx=1e-8
		rtol=1e-2
		atol=1e-10
		sn=self.resids.datadict.keys().__iter__().__next__()
		storedResults=self.defineStoredResults(self.guess,sn)
		
		uncertaintydict=self.resids.photVarianceForSN(self.guess,sn,*storedResults,np.ones(self.guess.size,dtype=bool))
		uncJac=	uncertaintydict['modelvariance_jacobian']

		uncertaintydict=self.resids.photVarianceForSN(self.guess,sn,*storedResults,np.zeros(self.guess.size,dtype=bool))
		uncertainty = uncertaintydict['modelvariance']
		
		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			newResults=self.defineStoredResults(guess,sn)
			return self.resids.photVarianceForSN(guess,sn,*newResults,np.zeros(self.guess.size,dtype=bool))
		
		
		dUncertaintydX=np.zeros((uncertainty.size,self.parlist.size))
		for i in range(self.guess.size):
			uncertaintydict=incrementOneParam(i,dx)
			dUncertaintydX[:,i]=(incrementOneParam(i,dx/2)['modelvariance']-incrementOneParam(i,-dx/2)['modelvariance'])/dx
		if not np.allclose(uncJac,dUncertaintydX,rtol,atol): print('Problems with model uncertainty derivatives: ',np.unique(self.parlist[np.where(~np.isclose(uncJac,dUncertaintydX,rtol,atol))[1]]))
		self.assertTrue((np.isclose(uncJac,dUncertaintydX,rtol,atol).all(axis=0)|((self.parlist==f'tpkoff_{sn}'))).all())

	def test_specuncertainty_jacobian(self):
		"""Check that jacobian of spectral variance is correct"""
		print("Checking spectral variance derivatives")
		dx=1e-8
		rtol=1e-2
		atol=1e-48
		sn=self.resids.datadict.keys().__iter__().__next__()
		storedResults=self.defineStoredResults(self.guess,sn)
		
		uncertaintydict=self.resids.specVarianceForSN(self.guess,sn,*storedResults,np.ones(self.guess.size,dtype=bool))
		uncJac=	uncertaintydict['modelvariance_jacobian']

		uncertaintydict=self.resids.specVarianceForSN(self.guess,sn,*storedResults,np.zeros(self.guess.size,dtype=bool))
		uncertainty = uncertaintydict['modelvariance']
		
		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			newResults=self.defineStoredResults(guess,sn)
			return self.resids.specVarianceForSN(guess,sn,*newResults,np.zeros(self.guess.size,dtype=bool))
		

		dUncertaintydX=np.zeros((uncertainty.size,self.parlist.size))
		for i in range(self.guess.size):
			uncertaintydict=incrementOneParam(i,dx)
			dUncertaintydX[:,i]=(uncertaintydict['modelvariance']-uncertainty)/dx
		if not np.allclose(uncJac,dUncertaintydX,rtol,atol): print('Problems with model variance derivatives: ',np.unique(self.parlist[np.where(~np.isclose(uncJac,dUncertaintydX,rtol,atol))[1]]))

		self.assertTrue((np.isclose(uncJac,dUncertaintydX,rtol,atol).all(axis=0)|((self.parlist==f'tpkoff_{sn}'))).all())


	def test_specvals_jacobian(self):
		"""Check that lognorm gradient and residuals' jacobian are correctly calculated with fixUncertainty=False"""
		print("Checking spectral derivatives")
		dx=1e-8
		rtol=1e-2
		atol=1e-20
		sn=self.resids.datadict.keys().__iter__().__next__()

		storedResults=self.defineStoredResults(self.guess,sn)
		
		valsdict=self.resids.specValsForSN(self.resids.guess,sn, *storedResults,np.ones(self.guess.size,dtype=bool))
		jacobian=valsdict['modelflux_jacobian']

		valsdict=self.resids.specValsForSN(self.resids.guess,sn, *storedResults,np.zeros(self.guess.size,dtype=bool))
		vals = valsdict['modelflux']
		
		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			storedResults=self.defineStoredResults(guess,sn)
			return self.resids.specValsForSN(guess,sn, *storedResults,np.zeros(self.guess.size,dtype=bool))
		dValdX=np.zeros((vals.size,self.parlist.size))
		for i in range(self.guess.size):
			valsdict=incrementOneParam(i)
			dValdX[:,i]=(valsdict['modelflux']-vals)/dx

		if not np.allclose(jacobian,dValdX,rtol,atol): print('Problems with model value derivatives: ',np.unique(self.parlist[np.where(~np.isclose(jacobian,dValdX,rtol,atol))[1]]))

		self.assertTrue((np.isclose(jacobian,dValdX,rtol,atol).all(axis=0)|((self.parlist==f'tpkoff_{sn}'))).all())

	def test_storePCDerivs(self):
		
		sn=self.resids.datadict.keys().__iter__().__next__()
		guess=self.guess
		resids=self.resids
		storedResults,temporaryResults=self.defineStoredResults(guess,sn)
		print("Testing that storing principal component deriviatives doesn't affect  photometric derivatives")
		jac=resids.photValsForSN(guess,sn, storedResults,temporaryResults,np.ones(resids.npar,dtype=bool))['modelflux_jacobian']
		newJac=resids.photValsForSN(guess,sn, storedResults,temporaryResults,np.ones(resids.npar,dtype=bool))['modelflux_jacobian']
		self.assertTrue(np.allclose(jac,newJac))
		print("Testing that storing principal component deriviatives doesn't affect  spectral derivatives")
		jac=resids.photValsForSN(guess,sn, storedResults,temporaryResults,np.ones(resids.npar,dtype=bool))['modelflux_jacobian']
		newJac=resids.photValsForSN(guess,sn, storedResults,temporaryResults,np.ones(resids.npar,dtype=bool))['modelflux_jacobian']
		self.assertTrue(np.allclose(jac,newJac))
		
	def test_specresid_jacobian(self):
		"""Checks that the the jacobian of the spectroscopic residuals is being correctly calculated to within 1%"""
		print("Checking spectral derivatives")
		dx=1e-8
		rtol=1e-2
		atol=1e-4
		sn=self.resids.datadict.keys().__iter__().__next__()
		
		storedResults=self.defineStoredResults(self.guess,sn,defineInterpolations=False)[0]
		#import pdb;pdb.set_trace()
		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,storedResults,varyParams=np.zeros(self.resids.npar,dtype=bool),fixUncertainty=True)
		residuals = specresidsdict['resid']
		
		uncertainties={key:storedResults[key] for key in storedResults if key.startswith('photvariances_') or key.startswith('specvariances_') or key.startswith('photCholesky_') }
		storedResults=self.defineStoredResults(self.guess,sn,defineInterpolations=False)[0]
		storedResults.update(uncertainties)

		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,storedResults,varyParams=np.ones(self.resids.npar,dtype=bool),fixUncertainty=True)
		jacobian=specresidsdict['resid_jacobian']
		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			storedResults=self.defineStoredResults(guess,sn,defineInterpolations=False)[0]
			storedResults.update(uncertainties)
			return self.resids.ResidsForSN(guess,sn,storedResults ,np.zeros(self.resids.npar,dtype=bool),fixUncertainty=True)[1]['resid']

		dResiddX=np.zeros((residuals.size,self.parlist.size))
		for i in range(self.guess.size):
			dResiddX[:,i]=(incrementOneParam(i,dx/2)-incrementOneParam(i,-dx/2))/dx

		if not np.allclose(jacobian,dResiddX,rtol): print('Problems with derivatives: ',np.unique(self.parlist[np.where(~np.isclose(jacobian,dResiddX,rtol,atol))[1]]))
		self.assertTrue((np.isclose(jacobian,dResiddX,rtol,atol).all(axis=0)|((self.parlist==f'tpkoff_{sn}'))).all())

	def test_specresid_jacobian_vary_uncertainty(self):
		"""Check that lognorm gradient and residuals' jacobian are correctly calculated with fixUncertainty=False"""
		print("Checking spectral resid derivatives")
		dx=1e-8
		rtol=1e-2
		atol=1e-2
		sn=self.resids.datadict.keys().__iter__().__next__()
		storedResults=self.defineStoredResults(self.guess,sn,defineInterpolations=False)[0]
		
		specresidsdict=self.resids.ResidsForSN(self.guess,sn,storedResults,varyParams=np.ones(self.resids.npar,dtype=bool))[1]
		grad=specresidsdict['lognorm_grad']
		jacobian=specresidsdict['resid_jacobian']
		
		specresidsdict=self.resids.ResidsForSN(self.guess,sn,storedResults,varyParams=np.zeros(self.resids.npar,dtype=bool))[1]

		residuals = specresidsdict['resid']
		lognorm= specresidsdict['lognorm']

		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			storedResults=self.defineStoredResults(guess,sn,defineInterpolations=False)[0]
			return self.resids.ResidsForSN(guess,sn,storedResults ,np.zeros(self.resids.npar,dtype=bool))[1]
			
		dResiddX=np.zeros((residuals.size,self.parlist.size))
		dLognormdX=np.zeros(self.parlist.size)
		for i in range(self.guess.size):
			residsdict=incrementOneParam(i,dx)
			dResiddX[:,i]=(residsdict['resid']-residuals)/dx
			dLognormdX[i]=(residsdict['lognorm']-lognorm)/dx
		if not np.allclose(jacobian,dResiddX,rtol,atol): print('Problems with residual derivatives: ',np.unique(self.parlist[np.where(~np.isclose(jacobian,dResiddX,rtol,atol))[1]]))
		if not np.allclose(grad,dLognormdX,rtol,atol): print('Problems with lognorm derivatives: ',np.unique(self.parlist[np.where(~np.isclose(grad,dLognormdX,rtol,atol))[0]]))

		self.assertTrue((np.isclose(jacobian,dResiddX,rtol,atol).all(axis=0)|((self.parlist==f'tpkoff_{sn}'))).all())
		self.assertTrue((np.isclose(grad,dLognormdX,rtol,atol).all(axis=0)|((self.parlist==f'tpkoff_{sn}'))).all())
	
	def test_computeDerivatives(self):
		"""Checks that the computeDerivatives parameter of the ResidsForSN function is not affecting the residuals"""
		sn=self.resids.datadict.keys().__iter__().__next__()

		combinations= [np.zeros(self.parlist.size,dtype=bool),np.ones(self.parlist.size,dtype=bool)]
		results=[self.resids.ResidsForSN(self.guess,sn,self.defineStoredResults(self.guess,sn,defineInterpolations=False)[0],x) for x in combinations]

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
		atol=1e-7
		for regularization, name in [(self.resids.dyadicRegularization,'dyadic'),
		(self.resids.phaseGradientRegularization, 'phase gradient'),
		(self.resids.waveGradientRegularization,'wave gradient' )]:
			for component in range(2):
				
				def incrementOneParam(i):
					guess=self.guess.copy()
					guess[i]+=dx
					storedResults=self.defineStoredResults(guess,None,defineInterpolations=False)[0]
					return regularization(guess,storedResults,np.zeros(self.parlist.size,dtype=bool))[0][component]
					
				storedResults=self.defineStoredResults(self.guess,None,defineInterpolations=False)[0]
				print('Checking jacobian of {} regularization, {} component'.format(name,component))

				residuals,jacobian=[x[component] for x in regularization(self.guess,storedResults,np.ones(self.parlist.size,dtype=bool))]
				dResiddX=np.zeros((residuals.size,self.parlist.size))
				for i in range(self.parlist.size):
					dResiddX[:,i]=(incrementOneParam(i)-residuals)/dx
				self.assertTrue(np.allclose(dResiddX,jacobian,rtol,atol))
	
if __name__ == "__main__":
    unittest.main()
