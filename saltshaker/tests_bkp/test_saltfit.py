import unittest
from saltshaker.training import saltfit
from saltshaker.training.TrainSALT import TrainSALT
import numpy as np
import pickle, argparse, configparser,warnings
from saltshaker.util import snana,readutils
from sncosmo.salt2utils import SALT2ColorLaw
from tqdm import trange
import sys
class TRAINING_Test(unittest.TestCase):
	
	def setUp(self):
		ts=TrainSALT()
		config = configparser.ConfigParser()
		config.read('testdata/test.conf')
		user_parser = ts.add_user_options(usage='',config=config)
		user_options = user_parser.parse_known_args()[0]
		
		trainingconfig = configparser.ConfigParser()
		trainingconfig.read(user_options.trainingconfig)

		training_parser = ts.add_training_options(
			usage='',config=trainingconfig)
		training_options = training_parser.parse_known_args(namespace=user_options)[0]

		ts.options=training_options
		
		kcordict=readutils.rdkcor(ts.surveylist,ts.options)
		ts.kcordict=kcordict

		# TODO: ASCII filter files
		# read the data
		datadict = readutils.rdAllData(ts.options.snlists,ts.options.estimate_tpk,kcordict,dospec=ts.options.dospec)

		self.parlist,self.guess,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc = ts.initialParameters(datadict)
		saltfitkwargs = ts.get_saltkw(phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)
		saltfitkwargs['regularize'] = ts.options.regularize
		saltfitkwargs['fitting_sequence'] = ts.options.fitting_sequence
		saltfitkwargs['fix_salt2modelpars'] = ts.options.fix_salt2modelpars

		self.fitter = saltfit.GaussNewton(self.guess,datadict,self.parlist,**saltfitkwargs)	

	def test_lsqwrap_jacobian(self):
		dx=1e-8
		rtol,atol=0.1,1e-2
		residuals,jacobian=self.fitter.lsqwrap(self.guess,{},self.fitter.iModelParam)
		jacobian=jacobian.toarray()
		import pdb;pdb.set_trace()
		#Other test already makes sure these are close enough for normal purposes, but small differences make a big difference in these results
		storedResults={}
		residuals=self.fitter.lsqwrap(self.guess,storedResults,np.zeros(self.parlist.size,dtype=bool))
		self.uncertaintyKeys={key for key in storedResults if key.startswith('photvariances_') or key.startswith('specvariances_') or key.startswith('photCholesky_') }
		uncertainties={key:storedResults[key] for key in self.uncertaintyKeys}

		def incrementOneParam(i,dx):
			guess=self.guess.copy()
			guess[i]+=dx
			return self.fitter.lsqwrap(guess,uncertainties.copy(),np.zeros(self.parlist.size,dtype=bool))

		dResiddX=np.zeros((residuals.size,self.guess.size))
		for i in (trange if sys.stdout.isatty() else np.arange)(self.guess.size):
			dResiddX[:,i]=(incrementOneParam(i,dx/2)-incrementOneParam(i,-dx/2))/dx
		dResiddX=dResiddX[:,self.fitter.iModelParam]
		
		np.save('testdata/testdresiddx.npy',dResiddX)
		np.save('testdata/jacobian.npy',jacobian)
		num=(~(np.isclose(dResiddX,jacobian,rtol,atol) | (self.parlist[self.fitter.iModelParam]=='tpkoff_5999390')[np.newaxis,:] )).sum()
		if num>0: 
			print('Problems with derivatives: ',np.unique(self.parlist[self.fitter.iModelParam][np.where(~np.isclose(dResiddX,jacobian,rtol,atol))[1]]))
			if num<5:
				print('Passing with {} problematic derivatives'.format(num))
		#Set these tolerances pretty broadly considering the small step size and the effects of numerical errors
		self.assertTrue(num<5)

	def test_lsqwrap_computeDerivatives(self):

		self.assertTrue(np.allclose(self.fitter.lsqwrap(self.guess,{}, np.zeros(self.parlist.size,dtype=bool)),self.fitter.lsqwrap(self.guess,{}, self.fitter.iModelParam)[0]))
