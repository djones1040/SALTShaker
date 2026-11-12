import unittest
from types import SimpleNamespace

import astropy.table as at
import numpy as np

from saltshaker.util import readutils
from saltshaker.util.readutils import readhostlogmass


class test_hostmass(unittest.TestCase):

    def test_reads_testdata_headers(self):
        """ASCII headers written as 'value +- error' give the mass and its uncertainty for every SN."""
        datadict = readutils.rdAllData(
            'testdata/data/SALT3TRAIN_K21_CSPDR3/SALT3TRAIN_K21_CSPDR3.LIST', False,
            peakmjdlist='testdata/test_pkmjd.LIST')
        self.assertEqual((datadict['2004ef'].hostlogmass, datadict['2004ef'].hostlogmasserr), (10.992, 0.049))
        self.assertEqual((datadict['2008gp'].hostlogmass, datadict['2008gp'].hostlogmasserr), (10.663, 3.998))
        self.assertTrue(all(np.isfinite(sn.hostlogmass) for sn in datadict.values()))

    def test_header_formats_and_sentinels(self):
        """Numeric and commented headers parse, and negative sentinels or missing keys give NaN."""
        cases = [
            (dict(HOSTGAL_LOGMASS=10.5, HOSTGAL_LOGMASS_ERR=0.1), (10.5, 0.1)),
            (dict(HOSTGAL_LOGMASS='9.8 +- 0.2 # from SED fit'), (9.8, 0.2)),
            (dict(HOSTGAL_LOGMASS=9.8, HOSTGAL_LOGMASS_ERR=-9), (9.8, np.nan)),
            (dict(HOSTGAL_LOGMASS=-9), (np.nan, np.nan)),
            (dict(HOSTGAL_LOGMASS='-999.0 +- -999.0'), (np.nan, np.nan)),
            (dict(HOSTGAL_LOGMASS='NULL'), (np.nan, np.nan)),
            (dict(), (np.nan, np.nan)),
        ]
        for header, expected in cases:
            np.testing.assert_array_equal(readhostlogmass(SimpleNamespace(SNID='sn1', **header)), expected)

    def test_snparlist_takes_precedence(self):
        """HOST_LOGMASS columns in the SN parameter list override the header; a missing entry gives NaN."""
        snpar = at.Table(dict(SNID=['sn1', 'sn2'], HOST_LOGMASS=[11.2, 0.0], HOST_LOGMASS_ERR=[0.3, 0.0]),
                         masked=True)
        snpar['HOST_LOGMASS'].mask[1] = True
        header = dict(HOSTGAL_LOGMASS='9.8 +- 0.2')
        np.testing.assert_array_equal(readhostlogmass(SimpleNamespace(SNID='sn1', **header), snpar), (11.2, 0.3))
        np.testing.assert_array_equal(readhostlogmass(SimpleNamespace(SNID='sn2', **header), snpar),
                                      (np.nan, np.nan))
        np.testing.assert_array_equal(readhostlogmass(SimpleNamespace(SNID='sn3', **header), snpar), (9.8, 0.2))


if __name__ == "__main__":
    unittest.main()
