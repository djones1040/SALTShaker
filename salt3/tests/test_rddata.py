import unittest
from salt3.util.estimate_tpk_bazin import estimate_tpk_bazin
from salt3.util import snana
from salt3.training.TrainSALT import TrainSALT
from salt3.training.init_hsiao import init_hsiao
import numpy as np

class RdTest(unittest.TestCase):

	def test_t0_measure(self):

		sn = snana.SuperNova('examples/exampledata/photdata/ASASSN-16bc.snana.dat')
		t0,msg = estimate_tpk_bazin(sn.MJD,sn.FLUXCAL,sn.FLUXCALERR)

		self.assertTrue('termination condition is satisfied' in msg)
		print(t0)
		self.assertTrue(np.abs(t0-57425) < 5)

	def test_rddata(self):

		snlist = 'examples/exampledata/photdata/PHOTFILES.LIST'
		nsn = len(np.loadtxt(snlist,dtype='str'))
		ts = TrainSALT()
		datadict = ts.rdAllData(snlist)

		self.assertTrue(len(datadict.keys()) == nsn)

	def test_rdkcor(self):

		kcorfile = 'examples/kcor/kcor_PS1_PS1MD.fits'
		survey = 'PS1MD'
		kcorpath = ('%s,%s'%(survey,kcorfile),)
		ts = TrainSALT()
		ts.rdkcor(kcorpath)

		self.assertTrue('kcordict' in ts.__dict__.keys())
		self.assertTrue('g' in ts.kcordict['PS1MD'].keys())
		self.assertTrue('r' in ts.kcordict['PS1MD'].keys())
		self.assertTrue('i' in ts.kcordict['PS1MD'].keys())
		self.assertTrue('z' in ts.kcordict['PS1MD'].keys())
		
		self.assertTrue(len(ts.kcordict['PS1MD']['g']['filttrans']) == len(ts.kcordict['PS1MD']['filtwave']))
		self.assertTrue(len(ts.kcordict['PS1MD']['primarywave']) == len(ts.kcordict['PS1MD']['AB']))

		
	def test_read_hsiao(self):
		initmodelfile = 'salt3/initfiles/Hsiao07.dat'
		waverange = [2000,9200]
		phaserange = [-14,50]
		phasesplineres = 3.2
		wavesplineres = 72
		phaseoutres = 2
		waveoutres = 2

		n_phaseknots = int((phaserange[1]-phaserange[0])/phasesplineres)-4
		n_waveknots = int((waverange[1]-waverange[0])/wavesplineres)-4
		
		phase,wave,m0,m1,m0knots,m1knots = init_hsiao(
			initmodelfile,phaserange=phaserange,waverange=waverange,
			phasesplineres=phasesplineres,wavesplineres=wavesplineres,
			phaseinterpres=phaseoutres,waveinterpres=waveoutres)

		self.assertTrue(np.abs(wave[1]-wave[0] - waveoutres) < 0.001)
		self.assertTrue(np.abs(phase[1]-phase[0] - phaseoutres) < 0.1)
		self.assertTrue(len(m0knots) == n_phaseknots*n_waveknots)
		self.assertTrue(len(m1knots) == n_phaseknots*n_waveknots)
		
if __name__ == "__main__":
	unittest.main()
