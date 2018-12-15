import unittest
from salt3.training import saltfit
from salt3.training.TrainSALT import TrainSALT
import numpy as np
import pickle
class TRAINING_Test(unittest.TestCase):

	def test_init_chi2func(self):
		
		waverange = [2000,9200]
		colorwaverange = [2800,7000]
		phaserange = [-14,50]
		phasesplineres = 3.2
		wavesplineres = 72
		phaseoutres = 2
		waveoutres = 2
		
		n_phaseknots = int((phaserange[1]-phaserange[0])/phasesplineres)-4
		n_waveknots = int((waverange[1]-waverange[0])/wavesplineres)-4

		snlist = 'examples/exampledata/photdata/PHOTFILES.LIST'
		ts = TrainSALT()
		datadict = ts.rdAllData(snlist)


		parlist = ['m0']*(n_phaseknots*n_waveknots)
		for k in datadict.keys():
			parlist += ['x0_%s'%k,'x1_%s'%k,'c_%s'%k]*len(datadict.keys())
		parlist = np.array(parlist)
		
		guess = np.ones(len(parlist))*1e-10
		
		kcorfile = 'examples/kcor/kcor_PS1_PS1MD.fits'
		survey = 'PS1MD'
		kcorpath = ('%s,%s'%(survey,kcorfile),)
		ts = TrainSALT()
		ts.rdkcor(kcorpath)

		saltfitter = saltfit.chi2(guess,datadict,parlist,phaserange,
								  waverange,phasesplineres,wavesplineres,phaseoutres,waveoutres,
								  colorwaverange,ts.kcordict)

		#saltfitter.chi2fit(guess)
		self.assertTrue('stdmag' in saltfitter.__dict__.keys())
	def test_chi2forSN(self):
		waverange = [2000,9200]
		colorwaverange = [2800,7000]
		phaserange = [-14,50]
		phasesplineres = 3.2
		wavesplineres = 72
		phaseoutres = 2
		waveoutres = 2
		n_colorpars=4
		n_components=2
		sn='SN2017lc'
		
		n_phaseknots = int((phaserange[1]-phaserange[0])/phasesplineres)-4
		n_waveknots = int((waverange[1]-waverange[0])/wavesplineres)-4

		snlist = 'examples/exampledata/photdata/PHOTFILES.LIST'
		ts = TrainSALT()
		datadict = ts.rdAllData(snlist)

		parlist = ['m0']*(n_phaseknots*n_waveknots)
		if n_components == 2:
			parlist += ['m1']*(n_phaseknots*n_waveknots)
		if n_colorpars:
			parlist += ['cl']*n_colorpars

		parlist += ['x0_%s'%sn,'x1_%s'%sn ,'c_%s'%sn,'tpkoff_%s'%sn]
		parlist = np.array(parlist)		
		#This is a basic example of some SALT model parameters with the SN parameters for 2017
		guess = np.load('examples/testfiles/testparams.npy')

		kcorfile = 'examples/kcor/kcor_PS1_PS1MD.fits'
		survey = 'PS1MD'
		kcorpath = ('%s,%s'%(survey,kcorfile),)
		ts = TrainSALT()
		ts.rdkcor(kcorpath)
		
		saltfitter = saltfit.chi2(guess,datadict,parlist,phaserange,
								  waverange,phasesplineres,wavesplineres,phaseoutres,waveoutres,
								  colorwaverange,ts.kcordict)
		#Change this if necessary
		expectedChi=167439.4242906828
		print(saltfitter.chi2forSN('SN2017lc',guess),expectedChi)
		assert np.abs(saltfitter.chi2forSN('SN2017lc',guess)-expectedChi)<1e-5
	def test_synphot(self):
		
		waverange = [2000,9200]
		colorwaverange = [2800,7000]
		phaserange = [-19,55]
		phasesplineres = 3.2
		wavesplineres = 72
		phaseoutres = 2
		waveoutres = 2
		
		n_phaseknots = int((phaserange[1]-phaserange[0])/phasesplineres)-4
		n_waveknots = int((waverange[1]-waverange[0])/wavesplineres)-4

		snlist = 'examples/exampledata/photdata/PHOTFILES.LIST'
		ts = TrainSALT()
		datadict = ts.rdAllData(snlist)

		parlist = ['m0']*(n_phaseknots*n_waveknots)
		for k in datadict.keys():
			parlist += ['x0_%s'%k,'x1_%s'%k,'c_%s'%k,'tpkoff_%s'%k]*len(datadict.keys())
		parlist = np.array(parlist)
		guess = np.ones(len(parlist))
		
		kcorfile = 'examples/kcor/kcor_PS1_PS1MD.fits'
		survey = 'PS1MD'
		kcorpath = ('%s,%s'%(survey,kcorfile),)
		ts = TrainSALT()
		ts.rdkcor(kcorpath)
		
		saltfitter = saltfit.chi2(guess,datadict,parlist,phaserange,
								  waverange,phasesplineres,wavesplineres,phaseoutres,waveoutres,
								  colorwaverange,ts.kcordict)

		flatnuwave,flatnuflux = np.loadtxt('salt3/initfiles/flatnu.dat',unpack=True)
		vegawave,vegaflux = np.loadtxt('salt3/initfiles/vegased_2004_stis.txt',unpack=True)
		flatnufluxinterp = np.interp(saltfitter.wave,flatnuwave,flatnuflux)
		vegafluxinterp = np.interp(saltfitter.wave,vegawave,vegaflux)

		ab_offsets = [-0.08,0.16,0.37,0.54]
		for flt,abo in zip('gri',ab_offsets):
			abmag = saltfitter.synflux(flatnufluxinterp,ts.kcordict['PS1MD'][flt]['zpoff'],survey='PS1MD',flt=flt)
			vegamag = saltfitter.synflux(vegafluxinterp,ts.kcordict['PS1MD'][flt]['zpoff'],survey='PS1MD',flt=flt)

			# make sure my two synthetic phot methods agree
			# both self-consistent and agree better than 3% (?!?!) with griz AB offsets from
			# possibly reliable source (http://www.astronomy.ohio-state.edu/~martini/usefuldata.html)
			self.assertTrue(np.abs(abmag-ts.kcordict['PS1MD'][flt]['zpoff']-saltfitter.stdmag['PS1MD'][flt]) < 0.001)
			self.assertTrue(np.abs(vegamag-abmag-abo) < 0.015)

