import unittest
from saltshaker.training.TrainSALT import TrainSALT,RunTraining
from saltshaker.util import readutils
#from saltshaker.training.saltfit import fitting
from saltshaker.training import optimizers
from saltshaker.training import saltresids
from saltshaker.training.datamodels import SALTparameters
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
        specrecallist = 'testdata/test_spcrl.LIST'
        
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
            maxsn=None,specrecallist=specrecallist)

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
        #import pdb; pdb.set_trace()
        maxlike = saltres.maxlikefit(x_modelpars,dospec=True)
        maxlike_phot = saltres.maxlikefit(x_modelpars,dospec=False)

        self.assertTrue(np.isclose(maxlike,-3.67e9,1e7))
        self.assertTrue(np.isclose(maxlike_phot,-3.67e9,1e7))
        
        tfinish = time.time()
        
        # this is slower than it used to be
        # but still shouldn't take more than 5 minutes
        self.assertTrue(tfinish - tstart < 300)

    def test_likelihood_passband_not_preintegrated(self):
        """Light curves with different wavelength coverage batch together, and each
applies the color law at its own wavelengths."""

        salt = TrainSALT()
        rt = RunTraining()
        rt.get_config_options(salt,'testdata/test.conf',None)
        salt.options.preintegrate_photometric_passband = False

        datadict = readutils.rdAllData(
            'testdata/data/SALT3TRAIN_K21_CSPDR3/SALT3TRAIN_K21_CSPDR3.LIST',False,
            dospec=True,
            peakmjdlist='testdata/test_pkmjd.LIST',
            binspecres=None,snparlist='testdata/test_pars.LIST',
            maxsn=None,specrecallist='testdata/test_spcrl.LIST')
        kcordict=readutils.rdkcor(['CSP'],salt.options)
        salt.kcordict = kcordict

        x_modelpars,modelconfiguration = salt.initialParameters(datadict)
        saltres = saltresids.SALTResids(datadict,kcordict,modelconfiguration,salt.options)

        # Nonzero colors, so the flux depends on where the color law is evaluated
        x_modelpars = x_modelpars.copy()
        x_modelpars[saltres.ic] = np.linspace(-0.3,0.5,np.size(saltres.ic))
        self.assertTrue(np.isfinite(saltres.maxlikefit(x_modelpars,dospec=True)))

        lcs = [(sn,flt,lc) for sn in saltres.datadict.values() for flt,lc in sn.photdata.items()]
        self.assertGreater(len({lc.pcderivsparse.shape[1] for _,_,lc in lcs}),1)

        batched = saltres.batchedphotlikelihood(x_modelpars,None,False,False)
        unbatched = sum(lc.modelloglikelihood(x_modelpars) for _,_,lc in lcs)
        np.testing.assert_allclose(batched,unbatched,rtol=1e-12)

        for sn,flt,lc in lcs:
            pars = SALTparameters(lc,x_modelpars)
            kcor = kcordict[sn.survey][flt]
            wave = saltres.wave[(sn.obswave >= kcor['minlam']) & (sn.obswave <= kcor['maxlam'])]
            colorlaw = sum(fun(c,cl,wave) for fun,c,cl in zip(lc.colorlawfunction,pars.c,pars.CL))
            fluxcoeffs = np.concatenate(([1.],pars.coordinates)) @ pars.components * pars.x0
            expected = np.clip((lc.pcderivsparse @ fluxcoeffs) @ 10**(-0.4*colorlaw),0,None)
            np.testing.assert_allclose(lc.modelflux(x_modelpars),expected,rtol=1e-12,err_msg=lc.uniqueid)
