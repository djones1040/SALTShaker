import unittest,os
from salt3.training import saltresids
from salt3.training.saltresids import _SCALE_FACTOR

from salt3.training.TrainSALT import TrainSALT
import numpy as np
import pickle, argparse, configparser,warnings
from salt3.util import snana,readutils
from sncosmo.salt2utils import SALT2ColorLaw
from scipy.interpolate import interp1d
import os
from salt3.config import config_rootdir

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
		kcordict=readutils.rdkcor(ts.surveylist,ts.options,addwarning=ts.addwarning)

		#if not os.path.exists(user_options.trainingconfig):
		#	print('warning : training config file %s doesn\'t exist.  Trying package directory'%user_options.trainingconfig)
		#	user_options.trainingconfig = '%s/%s'%(config_rootdir,user_options.trainingconfig)
		#if not os.path.exists(user_options.trainingconfig):
		#	raise RuntimeError('can\'t find training config file!  Checked %s'%user_options.trainingconfig)
	
		#ts.kcordict=kcordict

		# TODO: ASCII filter files
		# read the data
		datadict = readutils.rdAllData(ts.options.snlists,ts.options.estimate_tpk,kcordict,
									   ts.addwarning,dospec=ts.options.dospec)
		ts.kcordict=kcordict
		self.parlist,self.guess,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc = ts.initialParameters(datadict)
		saltfitkwargs = ts.get_saltkw(phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)
				 
		self.resids = saltresids.SALTResids(self.guess,datadict,self.parlist,**saltfitkwargs)	
		
		
		
	def defineStoredResults(self,guess,defineInterpolations=True):
		sn=self.resids.datadict.keys().__iter__().__next__()
		
		components = self.resids.SALTModel(guess)
		componentderivs = self.resids.SALTModelDeriv(guess,1,0,self.resids.phase,self.resids.wave)
		salterr = self.resids.ErrModel(guess)
		colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.resids.parlist == 'cl'])
		
		M0,M1=self.resids.SALTModel(guess)
		M0deriv,M1deriv=self.resids.SALTModelDeriv(guess,1,0,self.resids.phase,self.resids.wave)
		colorLaw = SALT2ColorLaw(self.resids.colorwaverange, guess[self.parlist == 'cl'])

		z = self.resids.datadict[sn]['zHelio']
		obsphase = self.resids.datadict[sn]['obsphase'] #self.phase*(1+z)
		x0,x1,c,tpkoff = guess[self.parlist == 'x0_%s'%sn],guess[self.parlist == 'x1_%s'%sn],\
						 guess[self.parlist == 'c_%s'%sn],guess[self.parlist == 'tpkoff_%s'%sn]

		#Apply MW extinction
		M0 *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
		M1 *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]

		M0deriv *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
		M1deriv *= self.resids.datadict[sn]['mwextcurve'][np.newaxis,:]
	
		colorlaw = -0.4 * colorLaw(self.resids.wave)
		colorexp = 10. ** (colorlaw * c)
		M0 *= colorexp; M1 *= colorexp
		M0 *= _SCALE_FACTOR/(1+z); M1 *= _SCALE_FACTOR/(1+z)

		int1dM0 = interp1d(obsphase,M0,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)
		int1dM1 = interp1d(obsphase,M1,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)

		# derivs
		M0deriv *= colorexp; M1deriv *= colorexp
		M0deriv *= _SCALE_FACTOR/(1+z); M1deriv *= _SCALE_FACTOR/(1+z)
		
		mod = x0*(M0 + x1*M1)
		int1d = interp1d(obsphase,mod,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)
		
		modderiv = x0*(M0deriv + x1*M1deriv)
		int1dderiv = interp1d(obsphase,modderiv,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)
		
		saltErr=self.resids.ErrModel(guess)
		saltCorr=self.resids.CorrelationModel(guess)
		
		prefactor=(self.resids.datadict[sn]['mwextcurve'] *colorexp*  _SCALE_FACTOR/(1+z))
		interr1d = [interp1d(obsphase,err * prefactor ,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True) for err in saltErr ]
		intcorr1d= [interp1d(obsphase,corr ,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True) for corr in saltCorr ]

		modelUncertainty= x0**2 * prefactor**2 *(saltErr[0]**2  + 2*x1* saltCorr[0]*saltErr[0]*saltErr[1] + x1**2 *saltErr[1]**2)
		uncertaintyInterp=interp1d(obsphase,modelUncertainty ,axis=0,kind=self.resids.interpMethod,bounds_error=True,assume_sorted=True)

		storedResults={}
		storedResults['components']=components
		storedResults['componentderivs']=componentderivs
		storedResults['saltErr']=saltErr
		storedResults['saltCorr']=saltCorr
		storedResults['colorLaw']=colorlaw
		storedResults['colorLawInterp']= interp1d(self.resids.wave,storedResults['colorLaw'],kind=self.resids.interpMethod,bounds_error=False,fill_value=0,assume_sorted=True)
		temporaryResults={}
		if defineInterpolations:
			temporaryResults['fluxInterp']=int1d
			temporaryResults['componentsInterp']=int1dM0,int1dM1
			temporaryResults['phaseDerivInterp']=int1dderiv
			temporaryResults['colorexp']=colorexp
			temporaryResults['uncertaintyComponentsInterp']=interr1d
			temporaryResults['uncertaintyCorrelationsInterp']=intcorr1d
			temporaryResults['modelUncertaintyInterp']=uncertaintyInterp
	
		return storedResults,temporaryResults
	
	def test_satisfyDefinitions(self):
		"""Checks that all designated priors are properly calculating the jacobian of their residuals to within 1%"""
				#Define simple models for m0,m1
		X=self.guess.copy()
		Xnew=self.resids.priors.satisfyDefinitions(X,self.resids.SALTModel(X))
		
		storedResults=self.defineStoredResults(X,defineInterpolations=False)[0]
		residuals={}
		uncertainties={}
		for sn in self.resids.datadict.keys():
			tempResults=storedResults.copy()
			results=self.resids.ResidsForSN(X,sn,tempResults,varyParams=np.zeros(self.resids.npar,dtype=bool),fixUncertainty=True)
			residuals[sn]=[x['resid'] for x in results]
			uncertainties.update({key:tempResults[key] for key in tempResults if key.startswith('photvariances_') or key.startswith('specvariances_') or key.startswith('photCholesky_') })

		storedResults=self.defineStoredResults(Xnew,defineInterpolations=False)[0]		
		for sn in self.resids.datadict.keys():
			tempResults=storedResults.copy()
			tempResults.update(uncertainties)
			results=self.resids.ResidsForSN(Xnew,sn,tempResults,varyParams=np.zeros(self.resids.npar,dtype=bool),fixUncertainty=True)
			assert(all([np.allclose(newResult['resid'],oldResids) for newResult,oldResids in zip(results,residuals[sn])]))
		
		
if __name__ == "__main__":
    unittest.main()
