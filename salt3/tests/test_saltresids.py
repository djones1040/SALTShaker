import unittest
from salt3.training import saltresids
from salt3.training.TrainSALT import TrainSALT
import numpy as np
import pickle, argparse, configparser,warnings
from salt3.util import snana,readutils
from sncosmo.salt2utils import SALT2ColorLaw

class TRAINING_Test(unittest.TestCase):
	
	def setUp(self):
		ts=TrainSALT()
		config = configparser.ConfigParser()
		config.read('testdata/test.conf')
		parser = argparse.ArgumentParser()
		parser = ts.add_options(usage='',config=config)
		options = parser.parse_args([])
		ts.options=options
		kcordict=readutils.rdkcor(ts.surveylist,options,addwarning=ts.addwarning)
		ts.kcordict=kcordict

		# TODO: ASCII filter files
		# read the data
		datadict = readutils.rdAllData(options.snlist,options.estimate_tpk,kcordict,
									   ts.addwarning,dospec=options.dospec)
		self.parlist,self.guess,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc = ts.initialParameters(datadict)
		saltfitkwargs = ts.get_saltkw(phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)

		self.resids = saltresids.SALTResids(self.guess,datadict,self.parlist,**saltfitkwargs)	
		
	def test_prior_jacobian(self):
		"""Checks that all designated priors are properly calculating the jacobian of their residuals to within 1%"""
				#Define simple models for m0,m1
		for prior in self.resids.priors:
			print('Testing prior', prior)
			components=self.resids.SALTModel(self.guess)
			resid,val,jacobian=self.resids.priors[prior](0.3145,self.guess,components)
			self.assertTrue(self.resids.priors[prior].numResids==resid.size)
			dx=1e-3
			rtol=1e-2
			def incrementOneParam(i):
				guess=self.guess.copy()
				guess[i]+=dx
				components=self.resids.SALTModel(guess)
				return self.resids.priors[prior](0.3145,guess,components)[0]
			dPriordX=np.zeros((resid.size,self.guess.size))
			for i in range(self.guess.size):
				dPriordX[:,i]=(incrementOneParam(i)-resid)/dx
		
			#import pdb;pdb.set_trace()
			#Check that all derivatives that should be 0 are zero
			if  not np.allclose(dPriordX,jacobian,rtol): print('Problems with derivatives for prior {} : '.format(prior),np.unique(self.parlist[np.where(~np.isclose(dPriordX,jacobian,rtol))]))
			self.assertTrue(np.all((dPriordX==0)==(jacobian==0)))
			self.assertTrue(np.allclose(jacobian,dPriordX,rtol))

	def test_photresid_jacobian(self):
		"""Checks that the the jacobian of the photometric residuals is being correctly calculated to within 1%"""
		print("Checking photometric derivatives")
		dx=1e-3
		rtol=1e-2
		sn=self.resids.datadict.keys().__iter__().__next__()
		components = self.resids.SALTModel(self.resids.guess)
		salterr = self.resids.ErrModel(self.resids.guess)
		if self.resids.n_colorpars:
			colorLaw = SALT2ColorLaw(self.resids.colorwaverange, self.resids.guess[self.resids.parlist == 'cl'])
		else: colorLaw = None
		if self.resids.n_colorscatpars:
			colorScat = True
		else: colorScat = None
		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,components,colorLaw,salterr,True,True)
		
		residuals = photresidsdict['photresid']
		
		jacobian=photresidsdict['photresid_jacobian']
		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			components=self.resids.SALTModel(guess)
			if self.resids.n_colorpars:
				colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])
			else: colorLaw = None
			return self.resids.ResidsForSN(guess,sn,components,colorLaw,salterr,False)[0]['photresid']
		dResiddX=np.zeros((residuals.size,self.parlist.size))
		for i in range(self.guess.size):
			dResiddX[:,i]=(incrementOneParam(i)-residuals)/dx

		if not np.allclose(dResiddX,jacobian,rtol): print('Problems with derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dResiddX,jacobian,rtol))[1]]))
		self.assertTrue(np.allclose(dResiddX,jacobian,rtol))

	def test_computeDerivatives(self):
		"""Checks that the computeDerivatives parameter of the ResidsForSN function is not affecting the residuals"""
		sn=self.resids.datadict.keys().__iter__().__next__()
		components = self.resids.SALTModel(self.resids.guess)
		salterr = self.resids.ErrModel(self.resids.guess)
		if self.resids.n_colorpars:
			colorLaw = SALT2ColorLaw(self.resids.colorwaverange, self.resids.guess[self.resids.parlist == 'cl'])
		else: colorLaw = None
		if self.resids.n_colorscatpars:
			colorScat = True
		else: colorScat = None
		combinations= [(True,True),(True,False),(False,False)]
		results=[self.resids.ResidsForSN(self.guess,sn,components,colorLaw,salterr,*x) for x in combinations]
		first=results[0]
		self.assertTrue([np.allclose(first[0]['photresid'],result[0]['photresid']) for result in results[1:]])
		self.assertTrue([np.allclose(first[1]['specresid'],result[1]['specresid']) for result in results[1:]])

	def test_specresid_jacobian(self):
		"""Checks that the the jacobian of the spectroscopic residuals is being correctly calculated to within 1%"""
		print("Checking spectral derivatives")
		dx=1e-3
		rtol=1e-2
		sn=self.resids.datadict.keys().__iter__().__next__()
		components = self.resids.SALTModel(self.resids.guess)
		salterr = self.resids.ErrModel(self.resids.guess)
		if self.resids.n_colorpars:
			colorLaw = SALT2ColorLaw(self.resids.colorwaverange, self.resids.guess[self.resids.parlist == 'cl'])
		else: colorLaw = None
		if self.resids.n_colorscatpars:
			colorScat = True
		else: colorScat = None
		
		specresidsdict=self.resids.ResidsForSN(self.guess,sn,components,colorLaw,salterr,True,True)[1]
		
		residuals = specresidsdict['specresid']
		jacobian=specresidsdict['specresid_jacobian']
		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			components=self.resids.SALTModel(guess)
			if self.resids.n_colorpars:
				colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])
			else: colorLaw = None
			return self.resids.ResidsForSN(guess,sn,components,colorLaw,salterr,False)[1]['specresid']
		dResiddX=np.zeros((residuals.size,self.parlist.size))
		for i in range(self.guess.size):
			dResiddX[:,i]=(incrementOneParam(i)-residuals)/dx
		if not np.allclose(dResiddX,jacobian,rtol): print('Problems with derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dResiddX,jacobian,rtol))[1]]))
		self.assertTrue(np.allclose(dResiddX,jacobian,rtol))

	def test_regularization_jacobian(self):
		"""Checks that the the jacobian of the spectroscopic residuals is being correctly calculated to within 1%"""
		dx=1e-8
		rtol=1e-2
		for regularization, name in [(self.resids.dyadicRegularization,'Dyadic'),(self.resids.phaseGradientRegularization, 'Phase gradient'),(self.resids.waveGradientRegularization,'Wave gradient' )]:
			
			def incrementOneParam(i):
				guess=self.guess.copy()
				guess[i]+=dx
				return regularization(guess,False)[0][0]
			
			print('Checking jacobian of {} regularization'.format(name))
			#Only checking the first component, since they're calculated using the same code
			residuals,jacobian=[x[0] for x in regularization(self.guess,True)]
			dResiddX=np.zeros((residuals.size,self.resids.im0.size))
			for i,j in enumerate(self.resids.im0):
				dResiddX[:,i]=(incrementOneParam(j)-residuals)/dx
			self.assertTrue(np.allclose(dResiddX,jacobian,rtol))
	
if __name__ == "__main__":
    unittest.main()
