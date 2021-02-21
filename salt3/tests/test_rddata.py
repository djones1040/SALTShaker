import unittest
from salt3.util.estimate_tpk_bazin import estimate_tpk_bazin
from salt3.util import snana,readutils
from salt3.training.TrainSALT import TrainSALT
from salt3.training.init_hsiao import init_hsiao
import numpy as np

class RdTest(unittest.TestCase):

	def test_t0_measure(self):

		sn = snana.SuperNova('testdata/TEST.DAT')
		t0,msg = estimate_tpk_bazin(sn.MJD,sn.FLUXCAL,sn.FLUXCALERR)

		self.assertTrue('termination condition is satisfied' in msg)
		self.assertTrue(np.abs(t0-float(sn.SIM_PEAKMJD.replace('days',''))) < 5)

	def test_rddata(self):

		snlist = 'testdata/SALT3TEST_SIMPLE.LIST'
		nsn = (np.loadtxt(snlist,dtype='str')).size
		ts = TrainSALT()
		options= type('test', (), {})()
		options.PS1_LOWZ_COMBINED_kcorfile='kcor_PS1_LOWZ_COMBINED.fits'
		options.PS1_LOWZ_COMBINED_subsurveylist='CFA3S,CFA3K,CFA4p1,CFA4p2,CSP,CFA1,CFA2'
		kcordict=readutils.rdkcor(['PS1_LOWZ_COMBINED'],options)
		datadict = readutils.rdAllData(snlist,False,kcordict)

		self.assertTrue(len(datadict.keys()) == nsn)

	def test_rdspecdata(self):

		snlist = 'testdata/SALT3TEST_SIMPLE.LIST'
		speclist = 'testdata/SALT3TEST_SIMPLE.LIST'
		nsn = np.loadtxt(snlist,dtype='str').size
		ts = TrainSALT()
		options= type('test', (), {})()
		options.PS1_LOWZ_COMBINED_kcorfile='kcor_PS1_LOWZ_COMBINED.fits'
		options.PS1_LOWZ_COMBINED_subsurveylist='CFA3S,CFA3K,CFA4p1,CFA4p2,CSP,CFA1,CFA2'
		
		kcordict=readutils.rdkcor(['PS1_LOWZ_COMBINED'],options)
		datadict = readutils.rdAllData(snlist,False,kcordict,speclist)

		self.assertTrue(len(datadict.keys()) == nsn)
		
	def test_rdkcor(self):
		survey = 'PS1MD(PS1MD)'
		ts = TrainSALT()
		options= type('test', (), {})()
		options.PS1MD_kcorfile='kcor_PS1_PS1MD.fits'
		options.PS1MD_subsurveylist='PS1MD'
		ts.kcordict=readutils.rdkcor(['PS1MD'],options)

		self.assertTrue('kcordict' in ts.__dict__.keys())
		self.assertTrue('g' in ts.kcordict[survey].keys())
		self.assertTrue('r' in ts.kcordict[survey].keys())
		self.assertTrue('i' in ts.kcordict[survey].keys())
		self.assertTrue('z' in ts.kcordict[survey].keys())
		
		self.assertTrue(len(ts.kcordict[survey]['g']['filttrans']) == len(ts.kcordict[survey]['filtwave']))
		self.assertTrue(len(ts.kcordict[survey]['primarywave']) == len(ts.kcordict[survey]['AB']))

		
	def test_read_hsiao(self):
		initmodelfile = 'salt3/initfiles/Hsiao07.dat'
		waverange = [2000,9200]
		phaserange = [-14,50]
		phasesplineres = 3.2
		wavesplineres = 72
		phaseoutres = 2
		waveoutres = 2

		n_phaseknots = int((phaserange[1]-phaserange[0])/phasesplineres)-3
		n_waveknots = int((waverange[1]-waverange[0])/wavesplineres)-3
		
		phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_hsiao(initmodelfile,
			 Bfilt='salt3/initfiles/Bessell90_B.dat',
			   flatnu='salt3/initfiles/flatnu.dat',phaserange=phaserange,waverange=waverange,
			phasesplineres=phasesplineres,wavesplineres=wavesplineres,
			phaseinterpres=phaseoutres,waveinterpres=waveoutres)

		self.assertTrue(np.abs(wave[1]-wave[0] - waveoutres) < 0.001)
		self.assertTrue(np.abs(phase[1]-phase[0] - phaseoutres) < 0.1)
		self.assertTrue(len(m0knots) == n_phaseknots*n_waveknots)
		self.assertTrue(len(m1knots) == n_phaseknots*n_waveknots)
		
if __name__ == "__main__":
	unittest.main()
