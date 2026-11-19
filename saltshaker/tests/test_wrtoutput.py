import gzip
import os
import tempfile
import unittest

import numpy as np

from saltshaker.training import saltresids
from saltshaker.training.TrainSALT import RunTraining, TrainSALT
from saltshaker.util import readutils

DATADIR = 'testdata/data/SALT3TRAIN_K21_CSPDR3'
# 2004ef is in both lists; the training keeps the copy from the first
LISTS = [['2004ef', '2005M'], ['2006et', '2008gp', '2004ef']]


def simvalues(i, j):
    """Distinct SIM_SALT2x0, SIM_SALT2x1, SIM_SALT2c and SIM_PEAKMJD for SN j of list i."""
    n = 10 * i + j
    return (n + 1) * 1e-3, n / 10 - 1, n / 100 - 0.1, 53300.25 + n


class test_wrtoutput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.expected = {}
        snlists = []
        for i, snids in enumerate(LISTS):
            # Each list in its own directory, so light-curve paths resolve against their own list
            listdir = os.path.join(cls.tmpdir.name, f'list{i}')
            os.makedirs(listdir)
            for j, snid in enumerate(snids):
                x0, x1, c, pkmjd = simvalues(i, j)
                cls.expected.setdefault(snid, (x0, x1, c, pkmjd))
                with gzip.open(f'{DATADIR}/CSPDR3_{snid}.dat.gz', 'rt') as fin:
                    lines = fin.readlines()
                isnid = [l.startswith('SNID:') for l in lines].index(True)
                lines[isnid + 1:isnid + 1] = [f'SIM_SALT2x0: {x0}\n', f'SIM_SALT2x1: {x1}\n',
                                              f'SIM_SALT2c: {c}\n', f'SIM_PEAKMJD: {pkmjd}  days\n']
                with open(os.path.join(listdir, f'CSPDR3_{snid}.dat'), 'w') as fout:
                    fout.writelines(lines)
            snlists += [os.path.join(listdir, 'SN.LIST')]
            with open(snlists[-1], 'w') as fout:
                fout.write(''.join(f'CSPDR3_{snid}.dat\n' for snid in snids))

        salt = TrainSALT()
        RunTraining().get_config_options(salt, 'testdata/test.conf', [])
        salt.options.snlists = ','.join(snlists)
        salt.kcordict = readutils.rdkcor(['CSP'], salt.options)
        datadict = readutils.rdAllData(
            salt.options.snlists, False, dospec=True, peakmjdlist='testdata/test_pkmjd.LIST',
            binspecres=None, snparlist='testdata/test_pars.LIST', maxsn=None,
            specrecallist='testdata/test_spcrl.LIST')
        x, config = salt.initialParameters(datadict)
        res = saltresids.SALTResids(datadict, salt.kcordict, config, salt.options)
        cls.trainingresult = res.processoptimizedparametersforoutput(x, x)
        salt.wrtoutput(cls.tmpdir.name, cls.trainingresult, None, None, datadict)

        with open(os.path.join(cls.tmpdir.name, 'salt3train_snparams.txt')) as fin:
            cls.rows = [l.split() for l in fin if not l.startswith('#')]

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_snparams_one_row_per_sn(self):
        """salt3train_snparams.txt has one row per trained SN, with the SIM values from the file the training read."""
        snids = [row[0] for row in self.rows]
        self.assertEqual(sorted(snids), sorted(self.trainingresult.snparams))
        self.assertEqual(sorted(snids), sorted(self.expected))
        for row in self.rows:
            sim = [float(v) for v in row[5:9]]
            np.testing.assert_allclose(sim[:3], self.expected[row[0]][:3], rtol=1e-9, atol=0, err_msg=row[0])
            # SIM_t0 is written to two decimals
            self.assertAlmostEqual(sim[3], self.expected[row[0]][3], delta=0.005, msg=row[0])
            np.testing.assert_allclose(float(row[1]), self.trainingresult.snparams[row[0]]['x0'], rtol=1e-9)


if __name__ == '__main__':
    unittest.main()
