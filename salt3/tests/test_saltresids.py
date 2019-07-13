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
		datadict = ts.mkcuts(datadict)
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
			dx=1e-3
			rtol=1e-2
			def incrementOneParam(i):
				guess=self.guess.copy()
				guess[i]+=dx
				components=self.resids.SALTModel(guess)
				return self.resids.priors[prior](0.3145,guess,components)[0]
			dPriordX=(np.vectorize(incrementOneParam)(np.arange(self.guess.size))-resid)/dx
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
		photresidsdict,specresidsdict=self.resids.ResidsForSN(self.guess,sn,components,colorLaw,True,True)
		
		residuals = photresidsdict['photresid']
		
		jacobian=np.zeros((residuals.size,self.parlist.size))
		jacobian[:,self.parlist=='cl'] = photresidsdict['dphotresid_dcl']
		jacobian[:,self.parlist=='m0'] = photresidsdict['dphotresid_dM0']
		jacobian[:,self.parlist=='m1'] = photresidsdict['dphotresid_dM1']
		for snparam in ('x0','x1','c'): #tpkoff should go here
			jacobian[:,self.parlist == '{}_{}'.format(snparam,sn)] = photresidsdict['dphotresid_d{}'.format(snparam)]
		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			components=self.resids.SALTModel(guess)
			if self.resids.n_colorpars:
				colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])
			else: colorLaw = None
			return self.resids.ResidsForSN(guess,sn,components,colorLaw,False)[0]['photresid']
		dResiddX=np.zeros((residuals.size,self.parlist.size))
		for i in range(self.guess.size):
			dResiddX[:,i]=(incrementOneParam(i)-residuals)/dx
		if not np.allclose(dResiddX,jacobian,rtol): print('Problems with derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dResiddX,jacobian,rtol))[1]]))
		self.assertTrue(np.allclose(dResiddX,jacobian,rtol))

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
		specresidsdict=self.resids.ResidsForSN(self.guess,sn,components,colorLaw,True,True)[1]
		
		residuals = specresidsdict['specresid']
		
		jacobian=np.zeros((residuals.size,self.parlist.size))
		jacobian[:,self.parlist=='cl'] = specresidsdict['dspecresid_dcl']
		jacobian[:,self.parlist=='m0'] = specresidsdict['dspecresid_dM0']
		jacobian[:,self.parlist=='m1'] = specresidsdict['dspecresid_dM1']
		for snparam in ('x0','x1','c'): #tpkoff should go here
			jacobian[:,self.parlist == '{}_{}'.format(snparam,sn)] = specresidsdict['dspecresid_d{}'.format(snparam)]
		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			components=self.resids.SALTModel(guess)
			if self.resids.n_colorpars:
				colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])
			else: colorLaw = None
			return self.resids.ResidsForSN(guess,sn,components,colorLaw,False)[1]['specresid']
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
	
# 	def test_init_chi2func(self):
# 		self.assertTrue(True)
# 		waverange = [2000,9200]
# 		colorwaverange = [2800,7000]
# 		phaserange = [-14,50]
# 		phasesplineres = 3.2
# 		wavesplineres = 72
# 		phaseoutres = 2
# 		waveoutres = 2
# 		
# 		n_phaseknots = int((phaserange[1]-phaserange[0])/phasesplineres)-4
# 		n_waveknots = int((waverange[1]-waverange[0])/wavesplineres)-4
# 
# 		snlist = 'examples/exampledata/photdata/PHOTFILES.LIST'
# 		ts = TrainSALT()
# 		datadict = ts.rdAllData(snlist)
# 
# 
# 		parlist = ['m0']*(n_phaseknots*n_waveknots)
# 		for k in datadict.keys():
# 			parlist += ['x0_%s'%k,'x1_%s'%k,'c_%s'%k]*len(datadict.keys())
# 		parlist = np.array(parlist)
# 		
# 		guess = np.ones(len(parlist))*1e-10
# 		
# 		kcorfile = 'examples/kcor/kcor_PS1_PS1MD.fits'
# 		survey = 'PS1MD'
# 		kcorpath = ('%s,%s'%(survey,kcorfile),)
# 		ts = TrainSALT()
# 		ts.rdkcor(kcorpath)
# 
# 		saltfitter = saltfit.chi2(guess,datadict,parlist,phaserange,
# 								  waverange,phasesplineres,wavesplineres,phaseoutres,waveoutres,
# 								  colorwaverange,ts.kcordict)
# 
# 		#saltfitter.chi2fit(guess)
# 		self.assertTrue('stdmag' in saltfitter.__dict__.keys())
# 	def test_chi2forSN(self):
# 		self.assertTrue(True)
# 		waverange = [2000,9200]
# 		colorwaverange = [2800,7000]
# 		phaserange = [-14,50]
# 		phasesplineres = 3.2
# 		wavesplineres = 72
# 		phaseoutres = 2
# 		waveoutres = 2
# 		n_colorpars=4
# 		n_components=2
# 		sn='SN2017lc'
# 		
# 		n_phaseknots = int((phaserange[1]-phaserange[0])/phasesplineres)-4
# 		n_waveknots = int((waverange[1]-waverange[0])/wavesplineres)-4
# 
# 		snlist = 'examples/exampledata/photdata/PHOTFILES.LIST'
# 		ts = TrainSALT()
# 		datadict = ts.rdAllData(snlist)
# 
# 		parlist = ['m0']*(n_phaseknots*n_waveknots)
# 		if n_components == 2:
# 			parlist += ['m1']*(n_phaseknots*n_waveknots)
# 		if n_colorpars:
# 			parlist += ['cl']*n_colorpars
# 
# 		parlist += ['x0_%s'%sn,'x1_%s'%sn ,'c_%s'%sn,'tpkoff_%s'%sn]
# 		parlist = np.array(parlist)		
# 		#This is a basic example of some SALT model parameters with the SN parameters for 2017
# 		guess = np.load('examples/testfiles/testparams.npy')
# 
# 		kcorfile = 'examples/kcor/kcor_PS1_PS1MD.fits'
# 		survey = 'PS1MD'
# 		kcorpath = ('%s,%s'%(survey,kcorfile),)
# 		ts = TrainSALT()
# 		ts.rdkcor(kcorpath)
		
# 	def test_synphot(self):
# 		self.assertTrue(True)
# 		waverange = [2000,9200]
# 		colorwaverange = [2800,7000]
# 		phaserange = [-19,55]
# 		phasesplineres = 3.2
# 		wavesplineres = 72
# 		phaseoutres = 2
# 		waveoutres = 2
# 		
# 		n_phaseknots = int((phaserange[1]-phaserange[0])/phasesplineres)-4
# 		n_waveknots = int((waverange[1]-waverange[0])/wavesplineres)-4
# 
# 		snlist = 'examples/exampledata/photdata/PHOTFILES.LIST'
# 		ts = TrainSALT()
# 		datadict = ts.rdAllData(snlist)
# 
# 		parlist = ['m0']*(n_phaseknots*n_waveknots)
# 		for k in datadict.keys():
# 			parlist += ['x0_%s'%k,'x1_%s'%k,'c_%s'%k,'tpkoff_%s'%k]*len(datadict.keys())
# 		parlist = np.array(parlist)
# 		guess = np.ones(len(parlist))
# 		
# 		kcorfile = 'examples/kcor/kcor_PS1_PS1MD.fits'
# 		survey = 'PS1MD'
# 		kcorpath = ('%s,%s'%(survey,kcorfile),)
# 		ts = TrainSALT()
# 		ts.rdkcor(kcorpath)
# 		
# 		saltfitter = saltfit.chi2(guess,datadict,parlist,phaserange,
# 								  waverange,phasesplineres,wavesplineres,phaseoutres,waveoutres,
# 								  colorwaverange,ts.kcordict)
# 
# 		flatnuwave,flatnuflux = np.loadtxt('salt3/initfiles/flatnu.dat',unpack=True)
# 		vegawave,vegaflux = np.loadtxt('salt3/initfiles/vegased_2004_stis.txt',unpack=True)
# 		flatnufluxinterp = np.interp(saltfitter.wave,flatnuwave,flatnuflux)
# 		vegafluxinterp = np.interp(saltfitter.wave,vegawave,vegaflux)
# 
# 		ab_offsets = [-0.08,0.16,0.37,0.54]
# 		for flt,abo in zip('gri',ab_offsets):
# 			abmag = saltfitter.synflux(flatnufluxinterp,ts.kcordict['PS1MD'][flt]['zpoff'],survey='PS1MD',flt=flt)
# 			vegamag = saltfitter.synflux(vegafluxinterp,ts.kcordict['PS1MD'][flt]['zpoff'],survey='PS1MD',flt=flt)
# 
# 			# make sure my two synthetic phot methods agree
# 			# both self-consistent and agree better than 3% (?!?!) with griz AB offsets from
# 			# possibly reliable source (http://www.astronomy.ohio-state.edu/~martini/usefuldata.html)
# 			self.assertTrue(np.abs(abmag-ts.kcordict['PS1MD'][flt]['zpoff']-saltfitter.stdmag['PS1MD'][flt]) < 0.001)
# 			self.assertTrue(np.abs(vegamag-abmag-abo) < 0.015)
# 
if __name__ == "__main__":
    unittest.main()
