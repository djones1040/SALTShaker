import unittest
from salt3.training import saltresids
from salt3.training.saltresids import _SCALE_FACTOR

from salt3.training.TrainSALT import TrainSALT
import numpy as np
import pickle, argparse, configparser,warnings
from salt3.util import snana,readutils
from sncosmo.salt2utils import SALT2ColorLaw
from scipy.interpolate import interp1d


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
		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,components,componentderivs,colorLaw,salterr,True,True)
		
		residuals = photresidsdict['resid']
		
		jacobian=photresidsdict['resid_jacobian']
		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			components=self.resids.SALTModel(guess)
			if self.resids.n_colorpars:
				colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])
			else: colorLaw = None
			return self.resids.ResidsForSN(guess,sn,components,componentderivs,colorLaw,salterr,False)[0]['resid']
		dResiddX=np.zeros((residuals.size,self.parlist.size))
		for i in range(self.guess.size):
			dResiddX[:,i]=(incrementOneParam(i)-residuals)/dx

		if not np.allclose(dResiddX,jacobian,rtol): print('Problems with derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dResiddX,jacobian,rtol))[1]]))
		self.assertTrue(np.allclose(dResiddX,jacobian,rtol))

	def test_photresid_jacobian_vary_uncertainty(self):
		"""Check that lognorm gradient and residuals' jacobian are correctly calculated with fixUncertainty=False"""
		print("Checking photometric derivatives including uncertainties")
		dx=1e-8
		rtol=1e-2
		atol=1e-2
		sn=self.resids.datadict.keys().__iter__().__next__()
		components = self.resids.SALTModel(self.resids.guess)
		salterr = self.resids.ErrModel(self.resids.guess)
		colorLaw = SALT2ColorLaw(self.resids.colorwaverange, self.resids.guess[self.resids.parlist == 'cl'])
		
		residsdict=self.resids.ResidsForSN(self.guess,sn,components,componentderivs,colorLaw,salterr,True,True,fixUncertainty=False)[0]
		grad=residsdict['lognorm_grad']
		jacobian=residsdict['resid_jacobian']

		residsdict=self.resids.ResidsForSN(self.guess,sn,components,componentderivs,colorLaw,salterr,False,False,fixUncertainty=False)[0]
		residuals = residsdict['resid']
		lognorm= residsdict['lognorm']

		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			components=self.resids.SALTModel(guess)
			colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])
			salterr=self.resids.ErrModel(guess)
			
			return self.resids.ResidsForSN(guess,sn,components,componentderivs,colorLaw,salterr,False,fixUncertainty=False)[0]
		dResiddX=np.zeros((residuals.size,self.parlist.size))
		dLognormdX=np.zeros(self.parlist.size)
		for i in range(self.guess.size):
			residsdict=incrementOneParam(i)
			dResiddX[:,i]=(residsdict['resid']-residuals)/dx
			dLognormdX[i]=(residsdict['lognorm']-lognorm)/dx
		if not np.allclose(dResiddX,jacobian,rtol):
			print('Problems with residual derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dResiddX,jacobian,rtol,atol))[1]]))
		if not np.allclose(dLognormdX,grad,rtol):
			print('Problems with lognorm derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dLognormdX,grad,rtol,atol))[0]]))
		self.assertTrue(np.allclose(dResiddX,jacobian,rtol,atol))
		self.assertTrue(np.allclose(dLognormdX,grad,rtol,atol))

	def test_specvals(self):
		"""Check that lognorm gradient and residuals' jacobian are correctly calculated with fixUncertainty=False"""
		print("Checking spectral derivatives")
		dx=1e-8
		rtol=1e-2
		atol=1e-20
		sn=self.resids.datadict.keys().__iter__().__next__()
		components = self.resids.SALTModel(self.resids.guess)
		salterr = self.resids.ErrModel(self.resids.guess)
		colorLaw = SALT2ColorLaw(self.resids.colorwaverange, self.guess[self.resids.parlist == 'cl'])

		M0,M1=self.resids.SALTModel(self.guess)
		colorLaw = SALT2ColorLaw(self.resids.colorwaverange, self.guess[self.parlist == 'cl'])
		saltErr=self.resids.ErrModel(self.guess)
		z = self.resids.datadict[sn]['zHelio']
		obsphase = self.resids.datadict[sn]['obsphase'] #self.phase*(1+z)
		x0,x1,c,tpkoff = self.guess[self.parlist == 'x0_%s'%sn],self.guess[self.parlist == 'x1_%s'%sn],\
						 self.guess[self.parlist == 'c_%s'%sn],self.guess[self.parlist == 'tpkoff_%s'%sn]

		#Apply MW extinction
		M0 *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
		M1 *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
	
		colorlaw = -0.4 * colorLaw(self.resids.wave)
		colorexp = 10. ** (colorlaw * c)
		M0 *= colorexp; M1 *= colorexp
		M0 *= _SCALE_FACTOR/(1+z); M1 *= _SCALE_FACTOR/(1+z)
		int1dM0 = interp1d(obsphase,M0,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)
		int1dM1 = interp1d(obsphase,M1,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)

		interr1d = [interp1d(obsphase,err * (self.resids.datadict[sn]['mwextcurve'] *colorexp*  _SCALE_FACTOR/(1+z))**2 ,axis=0,
							 kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True) for err in saltErr]

		mod = x0*(M0 + x1*M1)
		int1d = interp1d(obsphase,mod,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)

		valsdict=self.resids.specValsForSN(self.resids.guess,sn, (int1dM0,int1dM1),colorlaw,colorexp,True,True)
		uncertaintydict=self.resids.specUncertaintyForSN(self.guess,sn,(int1dM0,int1dM1),colorlaw,colorexp,interr1d,True)
		jacobian=valsdict['modelflux_jacobian']
		uncJac=	uncertaintydict['modeluncertainty_jacobian']

		valsdict=self.resids.specValsForSN(self.resids.guess,sn, int1d,colorlaw,colorexp,False,False)
		uncertaintydict=self.resids.specUncertaintyForSN(self.guess,sn,int1d,colorlaw,colorexp,interr1d,False)
		vals = valsdict['modelflux']
		uncertainty = uncertaintydict['modeluncertainty']
		
		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			M0,M1=self.resids.SALTModel(guess)
			colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])
			saltErr=self.resids.ErrModel(guess)
			z = self.resids.datadict[sn]['zHelio']
			obsphase = self.resids.datadict[sn]['obsphase'] #self.phase*(1+z)
			x0,x1,c,tpkoff = guess[self.parlist == 'x0_%s'%sn],guess[self.parlist == 'x1_%s'%sn],\
							 guess[self.parlist == 'c_%s'%sn],guess[self.parlist == 'tpkoff_%s'%sn]

			#Apply MW extinction
			M0 *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
			M1 *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
	
			colorlaw = -0.4 * colorLaw(self.resids.wave)
			colorexp = 10. ** (colorlaw * c)
	
			interr1d = [interp1d(obsphase,err * (self.resids.datadict[sn]['mwextcurve'] *colorexp*  _SCALE_FACTOR/(1+z))**2 ,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True) for err in saltErr]

			mod = x0*(M0 + x1*M1)*colorexp
			mod *= _SCALE_FACTOR/(1+z)
			int1d = interp1d(obsphase,mod,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)

			return self.resids.specValsForSN(guess,sn, int1d,colorlaw,colorexp,False,False),self.resids.specUncertaintyForSN(guess,sn,int1d,colorlaw,colorexp,interr1d,False)
		dValdX=np.zeros((vals.size,self.parlist.size))
		dUncertaintydX=np.zeros((vals.size,self.parlist.size))
		for i in range(self.guess.size):
			valsdict,uncertaintydict=incrementOneParam(i)
			dValdX[:,i]=(valsdict['modelflux']-vals)/dx
			dUncertaintydX[:,i]=(uncertaintydict['modeluncertainty']-uncertainty)/dx

		if not np.allclose(dValdX,jacobian,rtol,atol): print('Problems with model value derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dValdX,jacobian,rtol,atol))[1]]))
		if not np.allclose(dUncertaintydX,uncJac,rtol,atol): print('Problems with model value derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dUncertaintydX,uncJac,rtol,atol))[1]]))

		self.assertTrue(np.allclose(dValdX,jacobian,rtol,atol))
		self.assertTrue(np.allclose(dUncertaintydX,uncJac,rtol,atol))


	def test_specresid_jacobian_vary_uncertainty(self):
		"""Check that lognorm gradient and residuals' jacobian are correctly calculated with fixUncertainty=False"""
		print("Checking spectral resid derivatives")
		dx=1e-8
		rtol=1e-2
		atol=1e-2
		sn=self.resids.datadict.keys().__iter__().__next__()
		components = self.resids.SALTModel(self.resids.guess)
		salterr = self.resids.ErrModel(self.resids.guess)
		colorLaw = SALT2ColorLaw(self.resids.colorwaverange, self.resids.guess[self.resids.parlist == 'cl'])
		
		specresidsdict=self.resids.ResidsForSN(self.guess,sn,components,componentderivs,colorLaw,salterr,True,True,fixUncertainty=False)[1]
		grad=specresidsdict['lognorm_grad']
		jacobian=specresidsdict['resid_jacobian']

		specresidsdict=self.resids.ResidsForSN(self.guess,sn,components,componentderivs,colorLaw,salterr,False,False,fixUncertainty=False)[1]
		residuals = specresidsdict['resid']
		lognorm= specresidsdict['lognorm']

		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			components=self.resids.SALTModel(guess)
			colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])
			salterr=self.resids.ErrModel(guess)
			return self.resids.ResidsForSN(guess,sn,components,componentderivs,colorLaw,salterr,False,fixUncertainty=False)[1]
		dResiddX=np.zeros((residuals.size,self.parlist.size))
		dLognormdX=np.zeros(self.parlist.size)
		for i in range(self.guess.size):
			residsdict=incrementOneParam(i,dx)
			dResiddX[:,i]=(residsdict['resid']-residuals)/dx
			dLognormdX[i]=(residsdict['lognorm']-lognorm)/dx
		if not np.allclose(dResiddX,jacobian,rtol,atol): print('Problems with residual derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dResiddX,jacobian,rtol,atol))[1]]))
		if not np.allclose(dLognormdX,grad,rtol,atol): print('Problems with lognorm derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dLognormdX,grad,rtol,atol))[0]]))
		self.assertTrue(np.allclose(dResiddX,jacobian,rtol,atol))
		self.assertTrue(np.allclose(dLognormdX,grad,rtol,atol))

	
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
		results=[self.resids.ResidsForSN(self.guess,sn,components,componentderivs,colorLaw,salterr,*x) for x in combinations]
		first=results[0]
		self.assertTrue([np.allclose(first[0]['resid'],result[0]['resid']) for result in results[1:]])
		self.assertTrue([np.allclose(first[1]['resid'],result[1]['resid']) for result in results[1:]])

	def test_specresid_jacobian(self):
		"""Checks that the the jacobian of the spectroscopic residuals is being correctly calculated to within 1%"""
		print("Checking spectral derivatives")
		dx=1e-8
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
		
		componentderivs = self.resids.SALTModelDeriv(self.resids.guess,1,0,self.resids.phase,self.resids.wave)
		
		specresidsdict=self.resids.ResidsForSN(self.guess,sn,components,componentderivs,colorLaw,salterr,True,True)[1]
		
		residuals = specresidsdict['resid']
		jacobian=specresidsdict['resid_jacobian']
		import pdb; pdb.set_trace()
		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			components=self.resids.SALTModel(guess)
			componentderivs = self.resids.SALTModelDeriv(guess,1,0,self.resids.phase,self.resids.wave)
			colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])
			return self.resids.ResidsForSN(guess,sn,components,componentderivs,colorLaw,salterr,False)[1]['resid']
		dResiddX=np.zeros((residuals.size,self.parlist.size))
		for i in range(self.guess.size):
			dResiddX[:,i]=(incrementOneParam(i,dx)-residuals)/dx
			if self.parlist[i].startswith('tpkoff'):
				dxarr = np.arange(0.001,1.0,0.01)
				dxout = [np.sum((incrementOneParam(i,dx)-residuals)/dx) for dx in dxarr]
				import pdb;pdb.set_trace()
		import pdb; pdb.set_trace()
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
