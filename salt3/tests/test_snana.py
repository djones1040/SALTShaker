import unittest
from salt3.util import snana

class SNANA_Test(unittest.TestCase):

	def test_read_ascii(self):
		
		sn = snana.SuperNova('salt3/exampledata/ASASSN-16bc.snana.dat')
		self.assertTrue('FLUXCAL' in sn.__dict__.keys())
		self.assertTrue(len(sn.FLUXCAL) > 1)
		self.assertTrue('FLUXCALERR' in sn.__dict__.keys())
		self.assertTrue(len(sn.FLUXCALERR) > 1)
		
if __name__ == "__main__":
	unittest.main()
