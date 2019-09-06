import unittest
from salt3.training import saltfit
from salt3.training.TrainSALT import TrainSALT
import numpy as np
import pickle, argparse, configparser,warnings
from salt3.util import snana,readutils
from sncosmo.salt2utils import SALT2ColorLaw
from tqdm import trange
import sys
class TRAINING_Test(unittest.TestCase):
	
	def setUp(self):
		ts=TrainSALT()
		config = configparser.ConfigParser()
		config.read('testdata/test.conf')
		parser = argparse.ArgumentParser()
		parser = ts.add_options(usage='',config=config)
		options = parser.parse_args([])
		ts.options=options
		kcordict=readutils.rdkcor(ts.surveylist,options,addwarning=ts.addwarning)
		ts.kcordict=kcordict

		# TODO: ASCII filter files
		# read the data
		datadict = readutils.rdAllData(options.snlist,options.estimate_tpk,kcordict,
									   ts.addwarning,dospec=options.dospec)

		self.parlist,self.guess,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc = ts.initialParameters(datadict)
		saltfitkwargs = ts.get_saltkw(phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)

		self.fitter = saltfit.GaussNewton(self.guess,datadict,self.parlist,**saltfitkwargs)	

	def test_lsqwrap_jacobian(self):
		dx=1e-8
		rtol,atol=0.1,1e-2
		residuals,jacobian=self.fitter.lsqwrap(self.guess,True,True,True)
		#Other test already makes sure these are close enough for normal purposes, but small differences make a big difference in these results
		residuals=self.fitter.lsqwrap(self.guess,False,False)
		def incrementOneParam(i):
			guess=self.guess.copy()
			guess[i]+=dx
			return self.fitter.lsqwrap(guess,False,False)
		dResiddX=np.zeros((residuals.size,self.guess.size))
		
		for i in (trange if sys.stdout.isatty() else np.arange)(self.guess.size):
			dResiddX[:,i]=(incrementOneParam(i)-residuals)/dx
		num=(~np.isclose(dResiddX,jacobian,rtol,atol) ).sum()
		if num>0: 
			print('Problems with derivatives: ',np.unique(self.parlist[np.where(~np.isclose(dResiddX,jacobian,rtol,atol))[1]]))
			if num<5:
				print('Passing with {} problematic derivatives'.format(num))
		#Set these tolerances pretty broadly considering the small step size and the effects of numerical errors
		self.assertTrue(num<5)

	def test_lsqwrap_computeDerivatives(self):
		combinations= [(True,True),(True,False),(False,False)]
		results=[self.fitter.lsqwrap(self.guess,*combinations)[0] if combinations[0] else self.fitter.lsqwrap(self.guess,*combinations) for x in combinations]
		first=results[0]
		self.assertTrue([np.allclose(first,result) for result in results[1:]])
