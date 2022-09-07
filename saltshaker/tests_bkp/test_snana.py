import unittest
from saltshaker.util import snana

class SNANA_Test(unittest.TestCase):

	def test_read_ascii(self):
		
		sn = snana.SuperNova('testdata/TEST.DAT')
		self.assertTrue('FLUXCAL' in sn.__dict__.keys())
		self.assertTrue(len(sn.FLUXCAL) > 1)
		self.assertTrue('FLUXCALERR' in sn.__dict__.keys())
		self.assertTrue(len(sn.FLUXCALERR) > 1)

	def test_read_spectra(self):
		sn = snana.SuperNova('testdata/TEST.DAT')
		self.assertTrue('SPECTRA' in sn.__dict__.keys())
		count = 0
		for k in sn.SPECTRA.keys():
			count += 1
		self.assertTrue(count == sn.NSPECTRA)
		
if __name__ == "__main__":
	unittest.main()
