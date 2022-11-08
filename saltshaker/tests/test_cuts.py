# Created Nov 8 2022 by D.Jones and R.Kessler

import unittest
from saltshaker.util import readutils
from saltshaker.training.TrainSALT import TrainSALT,RunTraining

class test_cuts(unittest.TestCase):

    def __get_options__(self):
        
        salt = TrainSALT()
        configfile = 'testdata/test.conf'
        rt = RunTraining()
        rt.get_config_options(salt,configfile,None)

        self.options = salt.options
        
    def test_cuts(self):

        snlist = 'testdata/data/SALT3TRAIN_K21_CSPDR3/SALT3TRAIN_K21_CSPDR3.LIST'
        tmaxlist = 'testdata/test_pkmjd.LIST'
        snparlist = 'testdata/test_pars.LIST'
        
        datadict = readutils.rdAllData(
            snlist,False,
            dospec=True,
            peakmjdlist=tmaxlist,
            binspecres=None,snparlist=snparlist,
            maxsn=None)


        salt     = TrainSALT()
        self.__get_options__()
        salt.options = self.options
        kcordict = readutils.rdkcor(['CSP'],salt.options)
        salt.kcordict = kcordict
        salt.options.filtercen_obs_waverange = [ 4000, 25000 ]
        datadict = salt.mkcuts(datadict)[0]

        band_test = 'CSP-u'
        msgerr    = f"Should not accept {band_test}"  # warning; msg runs off screeen
        for snid in datadict:
            sn=datadict[snid]
            for flt in sn.filt:
                success = flt.split('/')[0] != band_test  # avoid checking filtname after slash
                assert success, msgerr
                # assert flt != 'CSP-u/t'

        return
        # end test_cuts

