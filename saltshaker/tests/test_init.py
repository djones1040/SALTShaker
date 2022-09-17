import unittest
from saltshaker.util import readutils
from saltshaker.training.TrainSALT import TrainSALT,RunTraining

class test_init(unittest.TestCase):

    def __get_options__(self):
        
        salt = TrainSALT()
        configfile = 'testdata/test.conf'
        rt = RunTraining()
        rt.get_config_options(salt,configfile,None)

        self.options = salt.options
        
    def test_rddata(self):

        snlist = 'testdata/data/SALT3TRAIN_K21_CSPDR3/SALT3TRAIN_K21_CSPDR3.LIST'
        tmaxlist = 'testdata/test_pkmjd.LIST'
        snparlist = 'testdata/test_pars.LIST'
        
        datadict = readutils.rdAllData(
            snlist,False,
            dospec=True,
            peakmjdlist=tmaxlist,
            binspecres=None,snparlist=snparlist,
            maxsn=None)

    def test_rdkcor(self):
        self.__get_options__()
        kcordict=readutils.rdkcor(['CSP'],self.options)
