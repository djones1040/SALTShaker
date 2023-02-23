import unittest
from saltshaker.training.TrainSALT import RunTraining
from saltshaker.training.TrainSALT import TrainSALT

class config_test(unittest.TestCase):
    """test that all config options exist in the example .conf"""

    def test_config_options(self):

        salt = TrainSALT()
        configfile = 'testdata/test.conf'
        rt = RunTraining()
        rt.get_config_options(salt,configfile,None)
