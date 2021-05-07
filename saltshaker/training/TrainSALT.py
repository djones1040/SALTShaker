#!/usr/bin/env python
# D. Jones, R. Kessler - 8/31/18
from __future__ import print_function

import argparse
import configparser
import numpy as np
import sys
import multiprocessing
import pickle

import os
from os import path

from scipy.linalg import lstsq
from scipy.optimize import minimize, least_squares, differential_evolution
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from sncosmo.constants import HC_ERG_AA
import astropy.table as at

from saltshaker.util import snana,readutils
from saltshaker.util.estimate_tpk_bazin import estimate_tpk_bazin
from saltshaker.util.txtobj import txtobj
from saltshaker.util.specSynPhot import getScaleForSN
from saltshaker.util.specrecal import SpecRecal

from saltshaker.training.init_hsiao import init_hsiao, init_kaepora, init_errs,init_errs_percent,init_custom,init_salt2
from saltshaker.training.base import TrainSALTBase
from saltshaker.training.saltfit import fitting
from saltshaker.training import saltfit as saltfit
from saltshaker.validation import ValidateParams

from saltshaker.data import data_rootdir
from saltshaker.initfiles import init_rootdir
from saltshaker.config import config_rootdir

import astropy.units as u
import sncosmo
import yaml

from astropy.table import Table
from saltshaker.initfiles import init_rootdir as salt2dir
_flatnu=f'{init_rootdir}/flatnu.dat'

# validation utils
import matplotlib as mpl
# mpl.use('MacOSX')
import pylab as plt
from saltshaker.validation import ValidateLightcurves
from saltshaker.validation import ValidateSpectra
from saltshaker.validation import ValidateModel
from saltshaker.validation import CheckSALTParams
from saltshaker.validation.figs import plotSALTModel
from saltshaker.util.synphot import synphot
from saltshaker.initfiles import init_rootdir as salt2dir
from saltshaker.validation import SynPhotPlot
from time import time
from sncosmo.salt2utils import SALT2ColorLaw
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.special import factorial
import extinction

import logging
log=logging.getLogger(__name__)

def RatioToSatisfyDefinitions(phase,wave,kcordict,components):
	"""Ensures that the definitions of M1,M0,x0,x1 are satisfied"""

	Bmag = synphot(
		kcordict['default']['primarywave'],kcordict['default']['AB'],
		filtwave=kcordict['default']['Bwave'],filttp=kcordict['default']['Btp'],
		zpoff=0)
	
	Bflux = 10**(0.4*(Bmag+27.5))

	filttrans = kcordict['default']['Btp']
	filtwave = kcordict['default']['Bwave']
			
	pbspl = np.interp(wave,filtwave,filttrans,left=0,right=0)
	
	pbspl *= wave
	denom = np.trapz(pbspl,wave)
	pbspl /= denom*HC_ERG_AA
	kcordict['default']['Bpbspl'] = pbspl
	
	int1d = interp1d(phase,components[0],axis=0,assume_sorted=True)
	m0Bflux = np.sum(kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
		(wave[1]-wave[0])*Bflux
	
	int1d = interp1d(phase,components[1],axis=0,assume_sorted=True)
	m1Bflux = np.sum(kcordict['default']['Bpbspl']*int1d([0]), axis=1)*\
		(wave[1]-wave[0])*Bflux
	ratio=m1Bflux/m0Bflux
	return ratio

def specflux(obsphase,obswave,m0phase,m0wave,m0flux,m1flux,colorlaw,z,x0,x1,c,mwebv):
	
	modelflux = x0*(m0flux + x1*m1flux)*1e-12/(1+z)

	m0interp = interp1d(np.unique(m0phase)*(1+z),m0flux*1e-12/(1+z),axis=0,
						kind='nearest',bounds_error=False,fill_value="extrapolate")
	m0phaseinterp = m0interp(obsphase)
	m0interp = np.interp(obswave,np.unique(m0wave)*(1+z),m0phaseinterp)

	m1interp = interp1d(np.unique(m0phase)*(1+z),m1flux*1e-12/(1+z),axis=0,
						kind='nearest',bounds_error=False,fill_value="extrapolate")
	m1phaseinterp = m1interp(obsphase)
	m1interp = np.interp(obswave,np.unique(m0wave)*(1+z),m1phaseinterp)

	
	intphase = interp1d(np.unique(m0phase)*(1+z),modelflux,axis=0,kind='nearest',bounds_error=False,fill_value="extrapolate")
	modelflux_phase = intphase(obsphase)
	intwave = interp1d(np.unique(m0wave)*(1+z),modelflux_phase,kind='nearest',bounds_error=False,fill_value="extrapolate")
	modelflux_wave = intwave(obswave)
	modelflux_wave = x0*(m0interp + x1*m1interp)
	mwextcurve = 10**(-0.4*extinction.fitzpatrick99(obswave,mwebv*3.1))
	modelflux_wave *= mwextcurve

	return modelflux_wave


class TrainSALT(TrainSALTBase):
	def __init__(self):
		self.warnings = []
		self.initializationtime=time()
	
	def initialParameters(self,datadict):
		from saltshaker.initfiles import init_rootdir
		self.options.inithsiaofile = f'{init_rootdir}/hsiao07.dat'
		self.options.initbfilt = f'{init_rootdir}/{self.options.initbfilt}'
		if self.options.initm0modelfile and not os.path.exists(self.options.initm0modelfile):
			if self.options.initm0modelfile:
				self.options.initm0modelfile = f'{init_rootdir}/{self.options.initm0modelfile}'
			if self.options.initm1modelfile:
				self.options.initm1modelfile = f'{init_rootdir}/{self.options.initm1modelfile}'
		
		if self.options.initm0modelfile and not os.path.exists(self.options.initm0modelfile):
			raise RuntimeError('model initialization file not found in local directory or %s'%init_rootdir)

		# initial guesses
		init_options = {'phaserange':self.options.phaserange,'waverange':self.options.waverange,
						'phasesplineres':self.options.phasesplineres,'wavesplineres':self.options.wavesplineres,
						'phaseinterpres':self.options.phaseinterpres,'waveinterpres':self.options.waveinterpres,
						'normalize':True,'order':self.options.interporder,'use_snpca_knots':self.options.use_snpca_knots}
				
		phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_hsiao(
			self.options.inithsiaofile,self.options.initbfilt,_flatnu,**init_options)

		if self.options.initsalt2model:
			if self.options.initm0modelfile =='':
				self.options.initm0modelfile=f'{init_rootdir}/salt2_template_0.dat'
			if self.options.initm1modelfile	 =='':
				self.options.initm1modelfile=f'{init_rootdir}/salt2_template_1.dat'

		if self.options.initm0modelfile and self.options.initm1modelfile:
			if self.options.initsalt2model:
				phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_salt2(
					m0file=self.options.initm0modelfile,m1file=self.options.initm1modelfile,
					Bfilt=self.options.initbfilt,flatnu=_flatnu,**init_options)
			else:
				phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_kaepora(
					self.options.initm0modelfile,self.options.initm1modelfile,
					Bfilt=self.options.initbfilt,flatnu=_flatnu,**init_options)

			
		init_options['phasesplineres'] = self.options.error_snake_phase_binsize
		init_options['wavesplineres'] = self.options.error_snake_wave_binsize
		init_options['order']=self.options.errinterporder
		init_options['n_colorscatpars']=self.options.n_colorscatpars

		del init_options['use_snpca_knots']
		if self.options.initsalt2var:
			errphaseknotloc,errwaveknotloc,m0varknots,m1varknots,m0m1corrknots,clscatcoeffs=init_errs(
				 *['%s/%s'%(init_rootdir,x) for x in ['salt2_lc_relative_variance_0.dat','salt2_lc_relative_covariance_01.dat','salt2_lc_relative_variance_1.dat','salt2_lc_dispersion_scaling.dat','salt2_color_dispersion.dat']],**init_options)
		else:
			#errphaseknotloc,errwaveknotloc,m0varknots,m1varknots,m0m1corrknots,clscatcoeffs=init_errs(**init_options)
			init_options['phase'] = phase
			init_options['wave'] = wave
			init_options['phaseknotloc'] = phaseknotloc
			init_options['waveknotloc'] = waveknotloc
			init_options['m0knots'] = m0knots
			init_options['m1knots'] = m1knots
			errphaseknotloc,errwaveknotloc,m0varknots,m1varknots,m0m1corrknots,clscatcoeffs=init_errs_percent(**init_options)
			
		# number of parameters
		n_phaseknots,n_waveknots = len(phaseknotloc)-self.options.interporder-1,len(waveknotloc)-self.options.interporder-1
		n_errphaseknots,n_errwaveknots = len(errphaseknotloc)-self.options.errinterporder-1,len(errwaveknotloc)-self.options.errinterporder-1
		n_sn = len(datadict.keys())

		# set up the list of parameters
		parlist = np.array(['m0']*(n_phaseknots*n_waveknots))
		if self.options.n_components == 2:
			parlist = np.append(parlist,['m1']*(n_phaseknots*n_waveknots))
		if self.options.n_colorpars:
			parlist = np.append(parlist,['cl']*self.options.n_colorpars)
		if self.options.error_snake_phase_binsize and self.options.error_snake_wave_binsize:
			for i in range(self.options.n_components): parlist = np.append(parlist,['modelerr_{}'.format(i)]*n_errphaseknots*n_errwaveknots)
			if self.options.n_components == 2:
				parlist = np.append(parlist,['modelcorr_01']*n_errphaseknots*n_errwaveknots)
		
		if self.options.n_colorscatpars:
			parlist = np.append(parlist,['clscat']*(self.options.n_colorscatpars))

		# SN parameters
		for k in datadict.keys():
			parlist = np.append(parlist,['x0_%s'%k,'x1_%s'%k,'c_%s'%k,'tpkoff_%s'%k])

		if self.options.specrecallist:
			spcrcldata = at.Table.read(self.options.specrecallist,format='ascii')
			
		# spectral params
		for sn in datadict.keys():
			specdata=datadict[sn].specdata
			photdata=datadict[sn].photdata
			for k in specdata.keys():
				if not self.options.specrecallist:
					order=self.options.n_min_specrecal+int(np.log((specdata[k].wavelength.max() - \
						specdata[k].wavelength.min())/self.options.specrange_wavescale_specrecal) + \
						len(datadict[sn].filt)* self.options.n_specrecal_per_lightcurve)
				else:
					spcrclcopy = spcrcldata[spcrcldata['SNID'] == sn]
					order = int(spcrclcopy['ncalib'][spcrclcopy['N'] == k+1])
				recalParams=[f'specx0_{sn}_{k}']+[f'specrecal_{sn}_{k}']*(order-1)
				parlist=np.append(parlist,recalParams)
		# initial guesses
		n_params=parlist.size
		guess = np.zeros(parlist.size)
		if self.options.resume_from_outputdir:
			log.info(f"resuming from output directory {self.options.outputdir}")
			
			names=None
			for possibleDir in [self.options.resume_from_outputdir,self.options.outputdir]:
				for possibleFile in ['salt3_parameters_unscaled.dat','salt3_parameters.dat']:	
					if names is None:
						try:
							names,pars = np.loadtxt(path.join(possibleDir,possibleFile),unpack=True,skiprows=1,dtype="U30,f8")
							break
						except:
							continue
			if self.options.resume_from_gnhistory:
				with open(f"{self.options.resume_from_gnhistory}/gaussnewtonhistory.pickle",'rb') as fin:
					data = pickle.load(fin)
					pars = data[-1][0]

			for key in np.unique(parlist):
				try:
					guess[parlist == key] = pars[names == key]
				except:
					log.critical(f'Problem while initializing parameter {key} from previous training')
					sys.exit(1)
					

		else:
			m0knots[m0knots == 0] = 1e-4
			guess[parlist == 'm0'] = m0knots
			for i in range(3): guess[parlist == 'modelerr_{}'.format(i)] = 1e-6 
			if self.options.n_components == 2:
				guess[parlist == 'm1'] = m1knots
			if self.options.n_colorpars:
				if self.options.initsalt2model:
					if self.options.n_colorpars == 4:
						guess[parlist == 'cl'] = [-0.504294,0.787691,-0.461715,0.0815619]
					else:
						clwave = np.linspace(self.options.waverange[0],self.options.waverange[1],1000)
						salt2cl = SALT2ColorLaw([2800.,7000.], [-0.504294,0.787691,-0.461715,0.0815619])(clwave)
						def bestfit(p):
							cl_init = SALT2ColorLaw(self.options.colorwaverange, p)(clwave)
							return cl_init-salt2cl

						md = least_squares(bestfit,[0,0,0,0,0])
						if 'termination conditions are satisfied' not in md.message:
							raise RuntimeError('problem initializing color law!')
						guess[parlist == 'cl'] = md.x
				else:
					guess[parlist == 'cl'] =[0.]*self.options.n_colorpars 
			if self.options.n_colorscatpars:

				guess[parlist == 'clscat'] = clscatcoeffs

			guess[(parlist == 'm0') & (guess < 0)] = 1e-4
			
			guess[parlist=='modelerr_0']=m0varknots
			guess[parlist=='modelerr_1']=m1varknots
			guess[parlist=='modelcorr_01']=m0m1corrknots

			# if SN param list is provided, initialize with these params
			if self.options.snparlist:
				snpar = Table.read(self.options.snparlist,format='ascii')
				snpar['SNID'] = snpar['SNID'].astype(str)
				
			for sn in datadict.keys():
				if self.options.snparlist:
					# hacky matching, but SN names are a mess as usual
					iSN = ((sn == snpar['SNID']) | ('sn'+sn == snpar['SNID']) |
						   ('sn'+sn.lower() == snpar['SNID']) | (sn+'.0' == snpar['SNID']))
					if len(snpar['SNID'][iSN]) > 1:
						raise RuntimeError(f"found duplicate in parameter list for SN {snpar['SNID'][iSN][0]}")
					if len(snpar[iSN]):
						guess[parlist == 'x0_%s'%sn] = snpar['x0'][iSN]
						guess[parlist == 'x1_%s'%sn] = snpar['x1'][iSN]
						guess[parlist == 'c_%s'%sn] = snpar['c'][iSN]
					else:
						log.warning(f'SN {sn} not found in SN par list {self.options.snparlist}')
						guess[parlist == 'x0_%s'%sn] = 10**(-0.4*(cosmo.distmod(datadict[sn].zHelio).value-19.36-10.635))

				else:
					guess[parlist == 'x0_%s'%sn] = 10**(-0.4*(cosmo.distmod(datadict[sn].zHelio).value-19.36-10.635))
				
				for k in datadict[sn].specdata : 
					guess[parlist==f'specx0_{sn}_{k}']= guess[parlist == 'x0_%s'%sn]

			# let's redefine x1 before we start
			ratio = RatioToSatisfyDefinitions(phase,wave,self.kcordict,[m0,m1])
			ix1 = np.array([i for i, si in enumerate(parlist) if si.startswith('x1')],dtype=int)
			guess[ix1]/=1+ratio*guess[ix1]
			guess[ix1]-=np.mean(guess[ix1])
			x1std = np.std(guess[ix1])
			if x1std == x1std and x1std != 0.0:
				guess[ix1]/= x1std
				

			# spectral params
			for sn in datadict.keys():
				specdata=datadict[sn].specdata
				photdata=datadict[sn].photdata
				for k in specdata.keys():
					order=(parlist == 'specrecal_{}_{}'.format(sn,k)).sum()
					
					pow=(order)-np.arange(order)
					recalCoord=(specdata[k].wavelength-np.mean(specdata[k].wavelength))/2500
					drecaltermdrecal=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]

					zHel,x0,x1,c = datadict[sn].zHelio,guess[parlist == f'x0_{sn}'],guess[parlist == f'x1_{sn}'],guess[parlist == f'c_{sn}']
					mwebv = datadict[sn].MWEBV
					colorlaw = SALT2ColorLaw(self.options.colorwaverange,guess[parlist == 'cl'])
					
					uncalledModel = specflux(specdata[k].tobs,specdata[k].wavelength,phase,wave,
											 m0,m1,colorlaw,zHel,x0,x1,c,mwebv=mwebv)
		
					def recalpars(x):
						recalexp=np.exp((drecaltermdrecal*x[1:][np.newaxis,:]).sum(axis=1))
						return (x[0]*uncalledModel*recalexp - specdata[k].flux)/specdata[k].fluxerr

					md = least_squares(recalpars,[np.median(specdata[k].flux)/np.median(uncalledModel)]+list(guess[parlist == 'specrecal_{}_{}'.format(sn,k)]))

					guess[parlist == f'specx0_{sn}_{k}' ]= md.x[0]*x0
					guess[parlist == f'specrecal_{sn}_{k}'] = md.x[1:]

		return parlist,guess,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc
	
	def fitSALTModel(self,datadict):

		parlist,x_modelpars,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc = self.initialParameters(datadict)

		saltfitkwargs = self.get_saltkw(phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)
		n_phaseknots,n_waveknots = len(phaseknotloc)-4,len(waveknotloc)-4
		n_errphaseknots,n_errwaveknots = len(errphaseknotloc)-4,len(errwaveknotloc)-4

		fitter = fitting(self.options.n_components,self.options.n_colorpars,
						 n_phaseknots,n_waveknots,
						 datadict)
		log.info('training on %i SNe!'%len(datadict.keys()))
		for i in range(self.options.n_repeat):
			if i == 0: laststepsize = None
			
			saltfitkwargs['regularize'] = self.options.regularize
			saltfitkwargs['fitting_sequence'] = self.options.fitting_sequence
			saltfitkwargs['fix_salt2modelpars'] = self.options.fix_salt2modelpars
			saltfitter = saltfit.GaussNewton(x_modelpars,datadict,parlist,**saltfitkwargs)

			# do the fitting
			trainingresult,message = fitter.gaussnewton(
					saltfitter,x_modelpars,
					self.options.gaussnewton_maxiter,getdatauncertainties=False)
			del saltfitter # free up memory before the uncertainty estimation

			# get the errors
			log.info('beginning error estimation loop')
			self.fit_model_err = False
			saltfitter_errs = saltfit.GaussNewton(trainingresult.X_raw,datadict,parlist,**saltfitkwargs)
			trainingresult,message = fitter.gaussnewton(
					saltfitter_errs,trainingresult.X_raw,
					0,getdatauncertainties=True)

			for k in datadict.keys():
				trainingresult.SNParams[k]['t0'] = -trainingresult.SNParams[k]['tpkoff'] + datadict[k].tpk_guess
		
		log.info('message: %s'%message)
		log.info('Final loglike'); saltfitter_errs.maxlikefit(trainingresult.X_raw)

		log.info(trainingresult.X.size)



		if 'chain' in saltfitter_errs.__dict__.keys():
			chain = saltfitter_errs.chain
			loglikes = saltfitter_errs.loglikes
		else: chain,loglikes = None,None

		return trainingresult,chain,loglikes

	def wrtoutput(self,outdir,trainingresult,chain,
				  loglikes,datadict):
		if not os.path.exists(outdir):
			raise RuntimeError('desired output directory %s doesn\'t exist'%outdir)

		#Save final model parameters
		
		with  open('{}/salt3_parameters.dat'.format(outdir),'w') as foutpars:
			foutpars.write('{: <30} {}\n'.format('Parameter Name','Value'))
			for name,par in zip(trainingresult.parlist,trainingresult.X):

				foutpars.write('{: <30} {:.15e}\n'.format(name,par))

		with  open('{}/salt3_parameters_unscaled.dat'.format(outdir),'w') as foutpars:
			foutpars.write('{: <30} {}\n'.format('Parameter Name','Value'))
			for name,par in zip(trainingresult.parlist,trainingresult.X_raw):

				foutpars.write('{: <30} {:.15e}\n'.format(name,par))
		
		#Save mcmc chain and log_likelihoods
		
		np.save('{}/salt3_mcmcchain.npy'.format(outdir),chain)
		np.save('{}/salt3_loglikes.npy'.format(outdir),loglikes)
		# principal components and color law
		with open(f'{outdir}/salt3_template_0.dat','w') as foutm0, open('%s/salt3_template_1.dat'%outdir,'w') as foutm1,\
			 open(f'{outdir}/salt3_lc_model_variance_0.dat','w') as foutm0modelerr,\
			 open(f'{outdir}/salt3_lc_model_variance_1.dat','w') as foutm1modelerr,\
			 open(f'{outdir}/salt3_lc_dispersion_scaling.dat','w') as fouterrmod,\
			 open(f'{outdir}/salt3_lc_model_covariance_01.dat','w') as foutmodelcov,\
			 open(f'{outdir}/salt3_lc_covariance_01.dat','w') as foutdatacov,\
			 open(f'{outdir}/salt3_lc_variance_0.dat','w') as foutm0dataerr,\
			 open(f'{outdir}/salt3_lc_variance_1.dat','w') as foutm1dataerr:
		
			for i,p in enumerate(trainingresult.phase):
				for j,w in enumerate(trainingresult.wave):
					print(f'{p:.1f} {w:.2f} {trainingresult.M0[i,j]:8.15e}',file=foutm0)
					print(f'{p:.1f} {w:.2f} {trainingresult.M1[i,j]:8.15e}',file=foutm1)
					print(f'{p:.1f} {w:.2f} {trainingresult.M0modelerr[i,j]**2.:8.15e}',file=foutm0modelerr)
					print(f'{p:.1f} {w:.2f} {trainingresult.M1modelerr[i,j]**2.:8.15e}',file=foutm1modelerr)
					print(f'{p:.1f} {w:.2f} {trainingresult.cov_M0_M1_model[i,j]:8.15e}',file=foutmodelcov)
					print(f'{p:.1f} {w:.2f} {trainingresult.cov_M0_M1_data[i,j]+trainingresult.cov_M0_M1_model[i,j]:8.15e}',file=foutdatacov)
					print(f'{p:.1f} {w:.2f} {trainingresult.modelerr[i,j]:8.15e}',file=fouterrmod)
					print(f'{p:.1f} {w:.2f} {trainingresult.M0dataerr[i,j]**2.+trainingresult.M0modelerr[i,j]**2.:8.15e}',file=foutm0dataerr)
					print(f'{p:.1f} {w:.2f} {trainingresult.M1dataerr[i,j]**2.+trainingresult.M1modelerr[i,j]**2.:8.15e}',file=foutm1dataerr)
					
		with open(f'{outdir}/salt3_color_dispersion.dat','w') as foutclscat:
			for j,w in enumerate(trainingresult.wave):
				print(f'{w:.2f} {trainingresult.clscat[j]:8.15e}',file=foutclscat)

		foutinfotext = f"""RESTLAMBDA_RANGE: {self.options.colorwaverange[0]} {self.options.colorwaverange[1]}
COLORLAW_VERSION: 1
COLORCOR_PARAMS: {self.options.colorwaverange[0]:.0f} {self.options.colorwaverange[1]:.0f}	{len(trainingresult.clpars)}  {' '.join(['%8.10e'%cl for cl in trainingresult.clpars])}

COLOR_OFFSET:  0.0

MAG_OFFSET:	 0.27  # to get B-band mag from cosmology fit (Nov 23, 2011)

SEDFLUX_INTERP_OPT: 2  # 1=>linear,	   2=>spline
ERRMAP_INTERP_OPT:	1  # 1	# 0=snake off;	1=>linear  2=>spline
ERRMAP_KCOR_OPT:	1  # 1/0 => on/off

MAGERR_FLOOR:	0.005			 # don;t allow smaller error than this
MAGERR_LAMOBS:	0.0	 2000  4000	 # magerr minlam maxlam
MAGERR_LAMREST: 0.1	  100	200	 # magerr minlam maxlam

SIGMA_INT: 0.106  # used in simulation"""
		with open(f'{outdir}/SALT3.INFO','w') as foutinfo:
			print(foutinfotext,file=foutinfo)
		

		with open(f'{outdir}/salt3_color_correction.dat','w') as foutcl:
			print(f'{len(trainingresult.clpars):.0f}',file=foutcl)
			for c in trainingresult.clpars:
				print(f'{c:8.10e}',file=foutcl)
			print(f"""Salt2ExtinctionLaw.version 1
			Salt2ExtinctionLaw.min_lambda {self.options.colorwaverange[0]:.0f}
			Salt2ExtinctionLaw.max_lambda {self.options.colorwaverange[1]:.0f}""",file=foutcl)

		
		# best-fit and simulated SN params
		with open(f'{outdir}/salt3train_snparams.txt','w') as foutsn:
			print('# SN x0 x1 c t0 tpkoff SIM_x0 SIM_x1 SIM_c SIM_t0 SALT2_x0 SALT2_x1 SALT2_c SALT2_t0',file=foutsn)
			for snlist in self.options.snlists.split(','):
				snlist = os.path.expandvars(snlist)
				if not os.path.exists(snlist):
					log.warning(f'SN list file {snlist} does not exist. Checking {data_rootdir}/trainingdata/{snlist}')
					snlist = f'{data_rootdir}/trainingdata/{snlist}'
					if not os.path.exists(snlist):
						raise RuntimeError(f'SN list file {snlist} does not exist')


				snfiles = np.genfromtxt(snlist,dtype='str')
				snfiles = np.atleast_1d(snfiles)

				for k in trainingresult.SNParams.keys():
					foundfile = False
					for l in snfiles:
						if '.fits' in l.lower(): continue
						if str(k) not in l: continue
						foundfile = True
						if '/' not in l:
							l = f"{os.path.dirname(snlist)}/{l}"
						sn = snana.SuperNova(l)
						if str(k) != str(sn.SNID): continue

						sn.SNID = str(sn.SNID)
						if 'SIM_SALT2x0' in sn.__dict__.keys(): SIM_x0 = sn.SIM_SALT2x0
						else: SIM_x0 = -99
						if 'SIM_SALT2x1' in sn.__dict__.keys(): SIM_x1 = sn.SIM_SALT2x1
						else: SIM_x1 = -99
						if 'SIM_SALT2c' in sn.__dict__.keys(): SIM_c = sn.SIM_SALT2c
						else: SIM_c = -99
						if 'SIM_PEAKMJD' in sn.__dict__.keys(): SIM_PEAKMJD = float(sn.SIM_PEAKMJD.split()[0])
						else: SIM_PEAKMJD = -99
						break
					if not foundfile:
						SIM_x0,SIM_x1,SIM_c,SIM_PEAKMJD,salt2x0,salt2x1,salt2c,salt2t0 = -99,-99,-99,-99,-99,-99,-99,-99
					elif self.options.fitsalt2:
						salt2x0,salt2x1,salt2c,salt2t0 = self.salt2fit(sn,datadict)
					else:
						salt2x0,salt2x1,salt2c,salt2t0 = -99,-99,-99,-99

					if 't0' not in trainingresult.SNParams[k].keys():
						trainingresult.SNParams[k]['t0'] = 0.0

					print(f"{k} {trainingresult.SNParams[k]['x0']:8.10e} {trainingresult.SNParams[k]['x1']:.10f} {trainingresult.SNParams[k]['c']:.10f} {trainingresult.SNParams[k]['t0']:.10f} {trainingresult.SNParams[k]['tpkoff']:.10f} {SIM_x0:8.10e} {SIM_x1:.10f} {SIM_c:.10f} {SIM_PEAKMJD:.2f} {salt2x0:8.10e} {salt2x1:.10f} {salt2c:.10f} {salt2t0:.10f}",file=foutsn)
		
		keys=['num_lightcurves','num_spectra','num_sne']
		yamloutputdict={key.upper():trainingresult.__dict__[key] for key in keys}
		yamloutputdict['CPU_MINUTES']=(time()-self.initializationtime)/60
		yamloutputdict['ABORT_IF_ZERO']=1
		with open(f'{self.options.yamloutputfile}','w') as file: yaml.dump(yamloutputdict,file)
		
		return

	def salt2fit(self,sn,datadict):

		if 'FLT' not in sn.__dict__.keys():
			sn.FLT = sn.BAND[:]
		for flt in np.unique(sn.FLT):
			filtwave = self.kcordict[sn.SURVEY][flt]['filtwave']
			filttrans = self.kcordict[sn.SURVEY][flt]['filttrans']

			band = sncosmo.Bandpass(
				filtwave,
				filttrans,
				wave_unit=u.angstrom,name=flt)
			sncosmo.register(band, force=True)

		data = Table(rows=None,names=['mjd','band','flux','fluxerr','zp','zpsys'],
					 dtype=('f8','S1','f8','f8','f8','U5'),
					 meta={'t0':sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]})

		sysdict = {}
		for m,flt,flx,flxe in zip(sn.MJD,sn.FLT,sn.FLUXCAL,sn.FLUXCALERR):
			if self.kcordict[sn.SURVEY][flt]['magsys'] == 'BD17': sys = 'bd17'
			elif self.kcordict[sn.SURVEY][flt]['magsys'] == 'AB': sys = 'ab'
			else: sys = 'vega'
			if self.kcordict[sn.SURVEY][flt]['lambdaeff']/(1+float(sn.REDSHIFT_HELIO.split('+-')[0])) > 2000 and \
			   self.kcordict[sn.SURVEY][flt]['lambdaeff']/(1+float(sn.REDSHIFT_HELIO.split('+-')[0])) < 9200 and\
			   '-u' not in self.kcordict[sn.SURVEY][flt]['fullname']:
				data.add_row((m,flt,flx,flxe,
							  27.5+self.kcordict[sn.SURVEY][flt]['zpoff'],sys))
			sysdict[flt] = sys
		
		flux = sn.FLUXCAL
		salt2model = sncosmo.Model(source='salt2')
		salt2model.set(z=float(sn.REDSHIFT_HELIO.split()[0]))
		fitparams = ['t0', 'x0', 'x1', 'c']

		result, fitted_model = sncosmo.fit_lc(
			data, salt2model, fitparams,
			bounds={'t0':(sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)][0]-10, sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)][0]+10),
					'z':(0.0,0.7),'x1':(-3,3),'c':(-0.3,0.3)})

		return result['parameters'][2],result['parameters'][3],result['parameters'][4],result['parameters'][1]
	
	def validate(self,outputdir,datadict,modelonly=False):

		# prelims
		plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=0.025, hspace=0)
		x0,x1,c,t0 = np.loadtxt(f'{outputdir}/salt3train_snparams.txt',unpack=True,usecols=[1,2,3,4])
		snid = np.genfromtxt(f'{outputdir}/salt3train_snparams.txt',unpack=True,dtype='str',usecols=[0])


		if self.options.fitsalt2:
			ValidateParams.main(f'{outputdir}/salt3train_snparams.txt',f'{outputdir}/saltparcomp.png')
		
		plotSALTModel.mkModelErrPlot(outputdir,outfile=f'{outputdir}/SALTmodelerrcomp.pdf',
									 xlimits=[self.options.waverange[0],self.options.waverange[1]])

		plotSALTModel.mkModelPlot(outputdir,outfile=f'{outputdir}/SALTmodelcomp.png',
								  xlimits=[self.options.waverange[0],self.options.waverange[1]],
								  n_colorpars=self.options.n_colorpars)
		SynPhotPlot.plotSynthPhotOverStretchRange(
			'{}/synthphotrange.pdf'.format(outputdir),outputdir,'SDSS')
		SynPhotPlot.overPlotSynthPhotByComponent(
			'{}/synthphotoverplot.pdf'.format(outputdir),outputdir,'SDSS')

		snfiles_tot = np.array([])
		for j,snlist in enumerate(self.options.snlists.split(',')):
			snlist = os.path.expandvars(snlist)
			snfiles = np.genfromtxt(snlist,dtype='str')
			snfiles = np.atleast_1d(snfiles)
			snfiles_tot = np.append(snfiles_tot,snfiles)
			parlist,parameters = np.genfromtxt(
				f'{outputdir}/salt3_parameters.dat',unpack=True,dtype=str,skip_header=1)
			parameters = parameters.astype(float)
			CheckSALTParams.checkSALT(parameters,parlist,snfiles,snlist,outputdir,idx=j)

		# kcor files
		kcordict = {}
		for k in self.kcordict.keys():
			if k == 'default': continue
			for k2 in self.kcordict[k].keys():
				if k2 not in ['primarywave','snflux','BD17','filtwave','AB','Vega']:
					if self.kcordict[k][k2]['magsys'] == 'AB': primarykey = 'AB'
					elif self.kcordict[k][k2]['magsys'] == 'Vega': primarykey = 'Vega'
					elif self.kcordict[k][k2]['magsys'] == 'VEGA': primarykey = 'Vega'
					elif self.kcordict[k][k2]['magsys'] == 'BD17': primarykey = 'BD17'

					kcordict[k2] = self.kcordict[k][k2]
					kcordict[k2]['stdmag'] = synphot(
						self.kcordict[k]['primarywave'],
						self.kcordict[k][primarykey],
						filtwave=self.kcordict[k][k2]['filtwave'],
						filttp=self.kcordict[k][k2]['filttrans'],
						zpoff=0) - self.kcordict[k][k2]['primarymag']

		from matplotlib.backends.backend_pdf import PdfPages
		plt.close('all')

		if modelonly:
			return
		
		pdf_pages = PdfPages(f'{outputdir}/lcfits.pdf')
		import matplotlib.gridspec as gridspec
		gs1 = gridspec.GridSpec(3, 5)
		gs1.update(wspace=0.0)
		i = 0
		
		# read in and save SALT2 files
		m0file='salt3_template_0.dat'
		m1file='salt3_template_1.dat'
		salt3phase,salt3wave,salt3flux = np.genfromtxt(f'{outputdir}/{m0file}',unpack=True)
		salt3m1phase,salt3m1wave,salt3m1flux = np.genfromtxt(f'{outputdir}/{m1file}',unpack=True)
		salt2phase,salt2wave,salt2flux = np.genfromtxt(f'{salt2dir}/salt2_template_0.dat',unpack=True)
		salt2m1phase,salt2m1wave,salt2m1flux = np.genfromtxt(f'{salt2dir}/salt2_template_1.dat',unpack=True)
		salt3phase = np.unique(salt3phase)
		salt3wave = np.unique(salt3wave)
		salt3flux = salt3flux.reshape([len(salt3phase),len(salt3wave)])
		salt3m1flux = salt3m1flux.reshape([len(salt3phase),len(salt3wave)])
		salt2phase = np.unique(salt2phase)
		salt2wave = np.unique(salt2wave)
		salt2m0flux = salt2flux.reshape([len(salt2phase),len(salt2wave)])
		salt2flux = salt2flux.reshape([len(salt2phase),len(salt2wave)])
		salt2m1flux = salt2m1flux.reshape([len(salt2phase),len(salt2wave)])

		saltdict = {'salt3phase':salt3phase,'salt3wave':salt3wave,'salt3flux':salt3flux,
					'salt3m1phase':salt3m1phase,'salt3m1wave':salt3m1wave,'salt3m1flux':salt3m1flux,
					'salt2phase':salt2phase,'salt2wave':salt2wave,'salt2m0flux':salt2m0flux,
					'salt2m1phase':salt2m1phase,'salt2m1wave':salt2m1wave,'salt2m1flux':salt2m1flux}

			
		for j,snlist in enumerate(self.options.snlists.split(',')):
			snlist = os.path.expandvars(snlist)
			if not os.path.exists(snlist):
				print(f'SN list file {snlist} does not exist.  Checking {data_rootdir}/trainingdata/{snlist}')
				snlist = f'{data_rootdir}/trainingdata/{snlist}'%(data_rootdir,snlist)
				if not os.path.exists(snlist):
					raise RuntimeError(f'SN list file {snlist} does not exist')

			tspec = time()
			if self.options.dospec:
				if self.options.binspec:
					binspecres = self.options.binspecres
				else:
					binspecres = None


				ValidateSpectra.compareSpectra(
					snlist,self.options.outputdir,specfile=f'{self.options.outputdir}/speccomp_{j:.0f}.pdf',
					maxspec=2000,base=self,verbose=self.verbose,datadict=datadict,binspecres=binspecres)
			log.info(f'plotting spectra took {time()-tspec:.1f}')
				
			snfiles = np.genfromtxt(snlist,dtype='str')
			snfiles = np.atleast_1d(snfiles)
			fitx1,fitc = False,False
			if self.options.n_components == 2:
				fitx1 = True
			if self.options.n_colorpars > 0:
				fitc = True

			if self.options.binspec:
				binspecres = self.options.binspecres
			else:
				binspecres = None
			
			datadict = readutils.rdAllData(snlist,self.options.estimate_tpk,
										   dospec=self.options.dospec,
										   peakmjdlist=self.options.tmaxlist,
										   binspecres=binspecres,snparlist=self.options.snparlist,
										   maxsn=self.options.maxsn)
				
			tlc = time()
			count = 0
			salt2_chi2tot,salt3_chi2tot = 0,0
			plotsnlist = []
			snfilelist = []
			for l in snfiles:
				if l.lower().endswith('.fits') or l.lower().endswith('.fits.gz'):

					if '/' not in l:
						l = '%s/%s'%(os.path.dirname(snlist),l)
					if l.lower().endswith('.fits') and not os.path.exists(l) and os.path.exists('{}.gz'.format(l)):
						l = '{}.gz'.format(l)
					# get list of SNIDs
					hdata = fits.getdata( l, ext=1 )
					survey = fits.getval( l, 'SURVEY')
					Nsn = fits.getval( l, 'NAXIS2', ext=1 )
					snidlist = np.array([ int( hdata[isn]['SNID'] ) for isn in range(Nsn) ])

					for sniditer in snidlist:
						sn = snana.SuperNova(
							snid=sniditer,headfitsfile=l,photfitsfile=l.replace('_HEAD.FITS','_PHOT.FITS'),
							specfitsfile=None,readspec=False)
						sn.SNID = str(sn.SNID)
						plotsnlist.append(sn)
						snfilelist.append(l)
				
				else:
					if '/' not in l:
						l = f'{os.path.dirname(snlist)}/{l}'
					sn = snana.SuperNova(l)
					sn.SNID = str(sn.SNID)
					if not sn.SNID in datadict:
						continue
					plotsnlist.append(sn)
					snfilelist.append(l)

			for sn,l in zip(plotsnlist,snfilelist):

				if not i % 12:
					fig = plt.figure()
				try:
					ax1 = plt.subplot(gs1[i % 15]); ax2 = plt.subplot(gs1[(i+1) % 15]); ax3 = plt.subplot(gs1[(i+2) % 15]); ax4 = plt.subplot(gs1[(i+3) % 15]); ax5 = plt.subplot(gs1[(i+4) % 15])
				except:
					import pdb; pdb.set_trace()


				if sn.SNID not in snid:
					log.warning(f'sn {sn.SNID} not in output files')
					continue
				x0sn,x1sn,csn,t0sn = \
					x0[snid == sn.SNID][0],x1[snid == sn.SNID][0],\
					c[snid == sn.SNID][0],t0[snid == sn.SNID][0]
				if not fitc: csn = 0
				if not fitx1: x1sn = 0

				if '.fits' in l.lower():
					snidval = int(sn.SNID)
				else:
					snidval = None
				salt2chi2,salt3chi2 = ValidateLightcurves.customfilt(
					f'{outputdir}/lccomp_{sn.SNID}.png',l,outputdir,
					t0=t0sn,x0=x0sn,x1=x1sn,c=csn,fitx1=fitx1,fitc=fitc,
					bandpassdict=self.kcordict,n_components=self.options.n_components,
					ax1=ax1,ax2=ax2,ax3=ax3,ax4=ax4,ax5=ax5,saltdict=saltdict,n_colorpars=self.options.n_colorpars,
					snid=snidval)
				salt2_chi2tot += salt2chi2
				salt3_chi2tot += salt3chi2
				if i % 12 == 8:
					pdf_pages.savefig()
					plt.close('all')
				else:
					for ax in [ax1,ax2,ax3,ax4]:
						ax.xaxis.set_ticklabels([])
						ax.set_xlabel(None)
				i += 4
				count += 1
			log.info(f'plotted light curves for {count} SNe')
			log.info(f'total chi^2 is {salt2_chi2tot:.1f} for SALT2 and {salt3_chi2tot:.1f} for SALT3')
		if not i %12 ==0:
			pdf_pages.savefig()
		pdf_pages.close()
		log.info(f'plotting light curves took {time()-tlc:.1f}')
		
	def main(self):
		try:
			stage='initialization'
			if not len(self.surveylist):
				raise RuntimeError('surveys are not defined - see documentation')
			tkstart = time()
			self.kcordict=readutils.rdkcor(self.surveylist,self.options)
			log.info(f'took {time()-tkstart:.3f} to read in kcor files')
			# TODO: ASCII filter files
				
			if not os.path.exists(self.options.outputdir):
				os.makedirs(self.options.outputdir)
			if self.options.binspec:
				binspecres = self.options.binspecres
			else:
				binspecres = None

			tdstart = time()
			datadict = readutils.rdAllData(self.options.snlists,self.options.estimate_tpk,
										   dospec=self.options.dospec,
										   peakmjdlist=self.options.tmaxlist,
										   binspecres=binspecres,snparlist=self.options.snparlist,maxsn=self.options.maxsn)
			log.info(f'took {time()-tdstart:.3f} to read in data files')
			tcstart = time()

			datadict = self.mkcuts(datadict)[0]
			log.info(f'took {time()-tcstart:.3f} to apply cuts')
			
			# fit the model - initial pass
			if self.options.stage == "all" or self.options.stage == "train":
				# read the data
				stage='training'

				trainingresult,chain,loglikes = self.fitSALTModel(datadict)
				stage='output'
				# write the output model - M0, M1, c
				self.wrtoutput(self.options.outputdir,trainingresult,chain,loglikes,datadict)
			log.info('successful SALT2 training!  Output files written to %s'%self.options.outputdir)
			if not self.options.skip_validation:
				if self.options.stage == "all" or self.options.stage == "validate":
					stage='validation'
					if self.options.validate_modelonly:
						self.validate(self.options.outputdir,datadict,modelonly=True)
					else:
						self.validate(self.options.outputdir,datadict,modelonly=False)
		except:
			log.exception(f'Exception raised during {stage}')
			if stage != 'validation':
				raise RuntimeError("Training exited unexpectedly")
		
