import unittest
from saltshaker.training.TrainSALT import TrainSALT,RunTraining
from saltshaker.util import readutils
#from saltshaker.training.saltfit import fitting
from saltshaker.training import optimizers
from saltshaker.training import saltresids
import numpy as np
import time

class test_likelihood(unittest.TestCase):

    def test_likelihood(self):
        """make sure the code can go through and return a likelihood.
Not really a unit test but hopefully gets simpler as we clean up the 
code."""

        tstart = time.time()
        
        snlist = 'testdata/data/SALT3TRAIN_K21_CSPDR3/SALT3TRAIN_K21_CSPDR3.LIST'
        tmaxlist = 'testdata/test_pkmjd.LIST'
        snparlist = 'testdata/test_pars.LIST'

        salt = TrainSALT()
        configfile = 'testdata/test.conf'
        rt = RunTraining()
        rt.get_config_options(salt,configfile,None)

        # get the data
        datadict = readutils.rdAllData(
            snlist,False,
            dospec=True,
            peakmjdlist=tmaxlist,
            binspecres=None,snparlist=snparlist,
            maxsn=None)

        # read the kcor file
        kcordict=readutils.rdkcor(['CSP'],salt.options)
        salt.kcordict = kcordict
        
        x_modelpars,modelconfiguration = salt.initialParameters(datadict)
        saltres = saltresids.SALTResids(datadict,kcordict,modelconfiguration,salt.options)
        
        #n_phaseknots,n_waveknots = len(phaseknotloc)-salt.options.interporder-1,len(waveknotloc)-salt.options.interporder-1
        
        #fitter = fitting(salt.options.n_components,salt.options.n_colorpars,
        #                 n_phaseknots,n_waveknots,
        #                 datadict)
        
        #saltfitkwargs = salt.get_saltkw(
        #    phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)
        #saltfitkwargs['regularize'] = True
        #saltfitkwargs['fitting_sequence'] = 'all'

        optimizer=optimizers.getoptimizer('rpropwithbacktracking')
        saltfitter = optimizer(x_modelpars,saltres,salt.options.outputdir,salt.options)
        #saltfitter = saltfit.GaussNewton(
        #    x_modelpars,datadict,parlist,**saltfitkwargs)
        
        #trainingresult,message = fitter.gaussnewton(
        #    saltfitter,
        #    x_modelpars,
        #    0,
        #    getdatauncertainties=False)

        maxlike = saltres.maxlikefit(x_modelpars,dospec=True)
        maxlike_phot = saltres.maxlikefit(x_modelpars,dospec=False)

        self.assertTrue(np.isclose(maxlike,-3.67e9,1e7))
        self.assertTrue(np.isclose(maxlike_phot,-3.67e9,1e7))
        
        tfinish = time.time()
        
        # this shouldn't take more than 3 minutes
        self.assertTrue(tfinish - tstart < 180)
