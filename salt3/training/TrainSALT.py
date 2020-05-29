#!/usr/bin/env python
# D. Jones, R. Kessler - 8/31/18
from __future__ import print_function

import argparse
import configparser
import numpy as np
import sys
import multiprocessing

import os
from os import path

from scipy.linalg import lstsq
from scipy.optimize import minimize, least_squares, differential_evolution
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from sncosmo.constants import HC_ERG_AA
import astropy.table as at

from salt3.util import snana,readutils
from salt3.util.estimate_tpk_bazin import estimate_tpk_bazin
from salt3.util.txtobj import txtobj
from salt3.util.specSynPhot import getScaleForSN
from salt3.util.specrecal import SpecRecal

from salt3.training.init_hsiao import init_hsiao, init_kaepora, init_errs,init_custom,init_salt2
from salt3.training.base import TrainSALTBase
from salt3.training.saltfit import fitting
from salt3.training import saltfit as saltfit

from salt3.data import data_rootdir
from salt3.initfiles import init_rootdir
from salt3.config import config_rootdir

import astropy.units as u
import sncosmo
from astropy.table import Table
from salt3.initfiles import init_rootdir as salt2dir
_flatnu=f'{init_rootdir}/flatnu.dat'

# validation utils
import matplotlib as mpl
mpl.use('agg')
import pylab as plt
from salt3.validation import ValidateLightcurves
from salt3.validation import ValidateSpectra
from salt3.validation import ValidateModel
from salt3.validation import CheckSALTParams
from salt3.validation.figs import plotSALTModel
from salt3.util.synphot import synphot
from salt3.initfiles import init_rootdir as salt2dir
from salt3.validation import SynPhotPlot
from time import time

import logging
log=logging.getLogger(__name__)


class TrainSALT(TrainSALTBase):
	def __init__(self):
		self.warnings = []
	
	def initialParameters(self,datadict):
		from salt3.initfiles import init_rootdir
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
						'phaseinterpres':self.options.phaseoutres,'waveinterpres':self.options.waveoutres,
						'normalize':True,'order':self.options.interporder}
				
		phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_hsiao(
			self.options.inithsiaofile,self.options.initbfilt,_flatnu,**init_options)

		if self.options.initsalt2model:
			if self.options.initm0modelfile =='':
				self.options.initm0modelfile=f'{init_rootdir}/salt2_template_0.dat'
			if self.options.initm1modelfile  =='':
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
		
		if self.options.initsalt2var:
			init_options['clscatfile'] = f'{init_rootdir}/salt2_color_dispersion.dat'
			errphaseknotloc,errwaveknotloc,m0varknots,m1varknots,m0m1corrknots,clscatcoeffs=init_errs(
				 *['%s/%s'%(init_rootdir,x) for x in ['salt2_lc_relative_variance_0.dat','salt2_lc_relative_covariance_01.dat','salt2_lc_relative_variance_1.dat','salt2_lc_dispersion_scaling.dat','salt2_color_dispersion.dat']],**init_options)
		else:
			errphaseknotloc,errwaveknotloc,m0varknots,m1varknots,m0m1corrknots,clscatcoeffs=init_errs(**init_options)

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
			specdata=datadict[sn]['specdata']
			photdata=datadict[sn]['photdata']
			for k in specdata.keys():
				if not self.options.specrecallist:
					try:
						order=self.options.n_min_specrecal+int(np.log((specdata[k]['wavelength'].max() - \
							specdata[k]['wavelength'].min())/self.options.specrange_wavescale_specrecal) + \
							np.unique(photdata['filt']).size* self.options.n_specrecal_per_lightcurve)
					except:
						import pdb; pdb.set_trace()
				else:
					spcrclcopy = spcrcldata[spcrcldata['SNID'] == sn]
					order = int(spcrclcopy['ncalib'][spcrclcopy['N'] == k+1])
					#print(spcrclcopy['phase'][spcrclcopy['N'] == k+1],specdata[k]['tobs'],float(spcrclcopy['phase'][spcrclcopy['N'] == k+1])-specdata[k]['tobs'])
					#if np.abs(float(spcrclcopy['phase'][spcrclcopy['N'] == k+1])-specdata[k]['tobs']) > 1:
					#	import pdb; pdb.set_trace()

				order-=1
				recalParams=[f'specx0_{sn}_{k}']+['specrecal_{}_{}'.format(sn,k)]*order
				parlist=np.append(parlist,recalParams)
		# initial guesses
		n_params=parlist.size
		guess = np.zeros(parlist.size)
		if self.options.resume_from_outputdir:
			names=None
			for possibleDir in [self.options.resume_from_outputdir,self.options.outputdir]:
				for possibleFile in ['salt3_parameters_unscaled.dat','salt3_parameters.dat']:	
					if names is None:
						try:
							names,pars = np.loadtxt(path.join(possibleDir,possibleFile),unpack=True,skiprows=1,dtype="U30,f8")
							break
						except:
							continue
			# HACK!
			#guess[parlist == 'm0'] = m0knots
			#guess[parlist == 'm1'] = m1knots
			for key in np.unique(parlist):
				try:
					#if key not in ['m0','m1']: 
					#	guess[parlist == key] = pars[names == key]
					#else:
					guess[parlist == key] = pars[names == key]
				except:
					log.critical('Problem while initializing parameter ',key,' from previous training')
					sys.exit(1)
					

		else:
			m0knots[m0knots == 0] = 1e-4
			guess[parlist == 'm0'] = m0knots
			for i in range(3): guess[parlist == 'modelerr_{}'.format(i)] = 1e-6 
			if self.options.n_components == 2:
				guess[parlist == 'm1'] = m1knots
			if self.options.n_colorpars:
				#log.warning('BAD CL HACK')
				if self.options.initsalt2model:
					guess[parlist == 'cl'] = [-0.504294,0.787691,-0.461715,0.0815619]
				else: 
					guess[parlist == 'cl'] =[0.]*self.options.n_colorpars 
			if self.options.n_colorscatpars:

				#guess[parlist == 'clscat'] = cdisp_coeffs #[1e-6]*self.options.n_colorscatpars
				#guess[np.where(parlist == 'clscat')[0][-1]]=-np.inf
				guess[parlist == 'clscat'] = clscatcoeffs

			guess[(parlist == 'm0') & (guess < 0)] = 1e-4
			
			guess[parlist=='modelerr_0']=m0varknots
			guess[parlist=='modelerr_1']=m1varknots
			guess[parlist=='modelcorr_01']=m0m1corrknots

			# if SN param list is provided, initialize with these params
			if self.options.snparlist:
				snpar = Table.read(self.options.snparlist,format='ascii')
				snpar['SNID'] = snpar['SNID'].astype(str)
				
			i=0
			for sn in datadict.keys():
				if self.options.snparlist:
					# hacky matching, but SN names are a mess as usual
					iSN = ((sn == snpar['SNID']) | ('sn'+sn == snpar['SNID']) |
						   ('sn'+sn.lower() == snpar['SNID']) | (sn+'.0' == snpar['SNID']))
					if len(snpar[iSN]):
						guess[parlist == 'x0_%s'%sn] = snpar['x0'][iSN]
						guess[parlist == 'x1_%s'%sn] = snpar['x1'][iSN]
						guess[parlist == 'c_%s'%sn] = snpar['c'][iSN]
					else:
						log.warning(f'SN {sn} not found in SN par list {self.options.snparlist}')
						guess[parlist == 'x0_%s'%sn] = 10**(-0.4*(cosmo.distmod(datadict[sn]['zHelio']).value-19.36-10.635))

				else:
					guess[parlist == 'x0_%s'%sn] = 10**(-0.4*(cosmo.distmod(datadict[sn]['zHelio']).value-19.36-10.635))
				
				for k in datadict[sn]['specdata'] : 
					guess[parlist==f'specx0_{sn}_{k}']= guess[parlist == 'x0_%s'%sn]
				i+=1


			# spectral params
			for sn in datadict.keys():
				specdata=datadict[sn]['specdata']
				photdata=datadict[sn]['photdata']
				for k in specdata.keys():
					if not self.options.specrecallist:
						order=self.options.n_min_specrecal+int(np.log((specdata[k]['wavelength'].max() - \
							specdata[k]['wavelength'].min())/self.options.specrange_wavescale_specrecal) + \
							np.unique(photdata['filt']).size* self.options.n_specrecal_per_lightcurve)
					else:
						spcrclcopy = spcrcldata[spcrcldata['SNID'] == sn]
						order = int(spcrclcopy['ncalib'][spcrclcopy['N'] == k+1])
					order-=1

					specpars_init = SpecRecal(datadict[sn]['photdata'],datadict[sn]['specdata'][k],self.kcordict,
											  datadict[sn]['survey'],self.options.specrange_wavescale_specrecal,
											  nrecalpars=order,sn=sn)
					if specpars_init[0] != 0:
						guess[parlist==f'specx0_{sn}_{k}']= guess[parlist == 'x0_%s'%sn]/specpars_init[0]
						if self.options.specrecal: guess[parlist == 'specrecal_{}_{}'.format(sn,k)] = specpars_init[1:]

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
			
			if self.options.do_mcmc:
				saltfitkwargs['regularize'] = False
				saltfitter = saltfit.mcmc(x_modelpars,datadict,parlist,**saltfitkwargs)
				log.info('initial loglike: %.1f'%saltfitter.maxlikefit(x_modelpars,{},np.zeros(len(x_modelpars),dtype=bool)))
				# do the fitting
				x_modelpars,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
					modelerr,clpars,clerr,clscat,SNParams,message = fitter.mcmc(
						saltfitter,x_modelpars,self.options.n_processes,
						self.options.n_steps_mcmc,self.options.n_burnin_mcmc)#,
						#stepsizes=laststepsize)

			if self.options.do_gaussnewton:
				saltfitkwargs['regularize'] = self.options.regularize
				saltfitkwargs['fitting_sequence'] = self.options.fitting_sequence
				saltfitkwargs['fix_salt2modelpars'] = self.options.fix_salt2modelpars
				saltfitter = saltfit.GaussNewton(x_modelpars,datadict,parlist,**saltfitkwargs)
				# do the fitting
				x_modelpars,x_unscaled,phase,wave,M0,M0modelerr,M0dataerr,M1,M1modelerr,M1dataerr,cov_M0_M1_model,cov_M0_M1_data,\
					modelerr,clpars,clerr,clscat,SNParams,laststepsize,message = fitter.gaussnewton(
						saltfitter,x_modelpars,
						self.options.gaussnewton_maxiter)

				#if not 'hi': #self.options.fit_model_err:
				#	M0dataerr,M1dataerr,cov_M0_M1_data = fitter.gaussnewton(
				#		saltfitter,x_modelpars,
				#		self.options.gaussnewton_maxiter,only_data_errs=True)
				#else:
				#	M0dataerr = M1dataerr = cov_M0_M1_data = np.zeros(len(M0))
					
			for k in datadict.keys():
				try:
					tpk_init = datadict[k]['photdata']['mjd'][0] - datadict[k]['photdata']['tobs'][0]
					SNParams[k]['t0'] = -SNParams[k]['tpkoff'] + tpk_init
				except:
					SNParams[k]['t0'] = -99
		
		log.info('message: %s'%message)
		#print('Final regularization chi^2 terms:', saltfitter.regularizationChi2(x_modelpars,1,0,0),
		#	  saltfitter.regularizationChi2(x_modelpars,0,1,0),saltfitter.regularizationChi2(x_modelpars,0,0,1))
		log.info('Final loglike'); saltfitter.maxlikefit(x_unscaled)

		log.info(x_modelpars.size)


		#print('Finishing...To write files, hit c')
		#import pdb; pdb.set_trace()

		if 'chain' in saltfitter.__dict__.keys():
			chain = saltfitter.chain
			loglikes = saltfitter.loglikes
		else: chain,loglikes = None,None

		return phase,wave,M0,M0modelerr,M0dataerr,M1,M1modelerr,M1dataerr,cov_M0_M1_model,cov_M0_M1_data,\
			modelerr,clpars,clerr,clscat,SNParams,x_modelpars,x_unscaled,parlist,chain,loglikes

	def wrtoutput(self,outdir,phase,wave,
				  M0,M0modelerr,M0dataerr,M1,M1modelerr,M1dataerr,cov_M0_M1_model,cov_M0_M1_data,
				  modelerr,clpars,
				  clerr,clscat,SNParams,pars,
				  pars_unscaled,parlist,chain,
				  loglikes,datadict):
		if not os.path.exists(outdir):
			raise RuntimeError('desired output directory %s doesn\'t exist'%outdir)

		#Save final model parameters
		
		with  open('{}/salt3_parameters.dat'.format(outdir),'w') as foutpars:
			foutpars.write('{: <30} {}\n'.format('Parameter Name','Value'))
			for name,par in zip(parlist,pars):

				foutpars.write('{: <30} {:.15e}\n'.format(name,par))

		with  open('{}/salt3_parameters_unscaled.dat'.format(outdir),'w') as foutpars:
			foutpars.write('{: <30} {}\n'.format('Parameter Name','Value'))
			for name,par in zip(parlist,pars_unscaled):

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
		
			for i,p in enumerate(phase):
				for j,w in enumerate(wave):
					print(f'{p:.1f} {w:.2f} {M0[i,j]:8.15e}',file=foutm0)
					print(f'{p:.1f} {w:.2f} {M1[i,j]:8.15e}',file=foutm1)
					print(f'{p:.1f} {w:.2f} {M0modelerr[i,j]**2.:8.15e}',file=foutm0modelerr)
					print(f'{p:.1f} {w:.2f} {M1modelerr[i,j]**2.:8.15e}',file=foutm1modelerr)
					print(f'{p:.1f} {w:.2f} {cov_M0_M1_model[i,j]:8.15e}',file=foutmodelcov)
					print(f'{p:.1f} {w:.2f} {cov_M0_M1_data[i,j]+cov_M0_M1_model[i,j]:8.15e}',file=foutdatacov)
					print(f'{p:.1f} {w:.2f} {modelerr[i,j]:8.15e}',file=fouterrmod)
					print(f'{p:.1f} {w:.2f} {M0dataerr[i,j]**2.+M0modelerr[i,j]**2.:8.15e}',file=foutm0dataerr)
					print(f'{p:.1f} {w:.2f} {M1dataerr[i,j]**2.+M1modelerr[i,j]**2.:8.15e}',file=foutm1dataerr)
					
		with open(f'{outdir}/salt3_color_dispersion.dat','w') as foutclscat:
			for j,w in enumerate(wave):
				print(f'{w:.2f} {clscat[j]:8.15e}',file=foutclscat)

		foutinfotext = f"""RESTLAMBDA_RANGE: {self.options.colorwaverange[0]} {self.options.colorwaverange[1]}
COLORLAW_VERSION: 1
COLORCOR_PARAMS: {self.options.colorwaverange[0]:.0f} {self.options.colorwaverange[1]:.0f}  {len(clpars)}  {' '.join(['%8.10e'%cl for cl in clpars])}

COLOR_OFFSET:  0.0

MAG_OFFSET:  0.27  # to get B-band mag from cosmology fit (Nov 23, 2011)

SEDFLUX_INTERP_OPT: 2  # 1=>linear,    2=>spline
ERRMAP_INTERP_OPT:  1  # 1  # 0=snake off;  1=>linear  2=>spline
ERRMAP_KCOR_OPT:    1  # 1/0 => on/off

MAGERR_FLOOR:   0.005            # don;t allow smaller error than this
MAGERR_LAMOBS:  0.0  2000  4000  # magerr minlam maxlam
MAGERR_LAMREST: 0.1   100   200  # magerr minlam maxlam

SIGMA_INT: 0.106  # used in simulation"""
		with open(f'{outdir}/SALT3.INFO','w') as foutinfo:
			print(foutinfotext,file=foutinfo)
		
		foutm0.close()
		foutm1.close()
		foutm0modelerr.close()
		foutm1modelerr.close()
		foutmodelcov.close()
		foutm0dataerr.close()
		foutm1dataerr.close()
		foutdatacov.close()
		fouterrmod.close()

		with open(f'{outdir}/salt3_color_correction.dat','w') as foutcl:
			print(f'{len(clpars):.0f}',file=foutcl)
			for c in clpars:
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

				for k in SNParams.keys():
					foundfile = False
					for l in snfiles:
						if str(k) not in l: continue
						foundfile = True
						if '/' not in l:
							l = '%s/%s'%(os.path.dirname(snlist),l)
						sn = snana.SuperNova(l)
						sn.SNID = str(sn.SNID)
						if 'SIM_SALT2x0' in sn.__dict__.keys(): SIM_x0 = sn.SIM_SALT2x0
						else: SIM_x0 = -99
						if 'SIM_SALT2x1' in sn.__dict__.keys(): SIM_x1 = sn.SIM_SALT2x1
						else: SIM_x1 = -99
						if 'SIM_SALT2c' in sn.__dict__.keys(): SIM_c = sn.SIM_SALT2c
						else: SIM_c = -99
						if 'SIM_PEAKMJD' in sn.__dict__.keys(): SIM_PEAKMJD = float(sn.SIM_PEAKMJD.split()[0])
						else: SIM_PEAKMJD = -99
					if not foundfile:
						SIM_x0,SIM_x1,SIM_c,SIM_PEAKMJD,salt2x0,salt2x1,salt2c,salt2t0 = -99,-99,-99,-99,-99,-99,-99,-99
					elif self.options.fitsalt2:
						salt2x0,salt2x1,salt2c,salt2t0 = self.salt2fit(sn,datadict)
					else:
						salt2x0,salt2x1,salt2c,salt2t0 = -99,-99,-99,-99

					if 't0' not in SNParams[k].keys():
						SNParams[k]['t0'] = 0.0
					print(f"{k} {SNParams[k]['x0'][0]:8.10e} {SNParams[k]['x1'][0]:.10f} {SNParams[k]['c'][0]:.10f} {SNParams[k]['t0'][0]:.10f} {SNParams[k]['tpkoff'][0]:.10f} {SIM_x0:8.10e} {SIM_x1:.10f} {SIM_c:.10f} {SIM_PEAKMJD:.2f} {salt2x0:8.10e} {salt2x1:.10f} {salt2c:.10f} {salt2t0:.10f}",file=foutsn)

		return

	def salt2fit(self,sn,datadict):

		for flt in np.unique(sn.FLT):
			filtwave = self.kcordict[sn.SURVEY]['filtwave']
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
	
	def validate(self,outputdir,datadict):

		# prelims
		plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=0.025, hspace=0)
		x0,x1,c,t0 = np.loadtxt(f'{outputdir}/salt3train_snparams.txt',unpack=True,usecols=[1,2,3,4])
		snid = np.genfromtxt(f'{outputdir}/salt3train_snparams.txt',unpack=True,dtype='str',usecols=[0])

		# have stopped really looking at these for now
		#ValidateModel.main(
		#	'%s/spectralcomp.png'%outputdir,
		#	outputdir)
		#ValidateModel.m0m1_chi2(
		#	'%s/spectralcomp_chi2.png'%outputdir,
		#	outputdir)

		if self.options.fitsalt2:
			from salt3.validation import ValidateParams
			ValidateParams.main(f'{outputdir}/salt3train_snparams.txt',f'{outputdir}/saltparcomp.png')
		
		plotSALTModel.mkModelErrPlot(outputdir,outfile=f'{outputdir}/SALTmodelerrcomp.pdf',
									 xlimits=[self.options.waverange[0],self.options.waverange[1]])

		plotSALTModel.mkModelPlot(outputdir,outfile=f'{outputdir}/SALTmodelcomp.pdf',
								  xlimits=[self.options.waverange[0],self.options.waverange[1]],
								  n_colorpars=self.options.n_colorpars)
		SynPhotPlot.plotSynthPhotOverStretchRange(
			'{}/synthphotrange.pdf'.format(outputdir),outputdir,'Bessell')
		SynPhotPlot.overPlotSynthPhotByComponent(
			'{}/synthphotoverplot.pdf'.format(outputdir),outputdir,'Bessell')		
		
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
					kcordict[k2]['filtwave'] = self.kcordict[k]['filtwave']
					kcordict[k2]['stdmag'] = synphot(
						self.kcordict[k]['primarywave'],
						self.kcordict[k][primarykey],
						filtwave=self.kcordict[k]['filtwave'],
						filttp=self.kcordict[k][k2]['filttrans'],
						zpoff=0) - self.kcordict[k][k2]['primarymag']

		from matplotlib.backends.backend_pdf import PdfPages
		plt.close('all')

		pdf_pages = PdfPages(f'{outputdir}/lcfits.pdf')
		import matplotlib.gridspec as gridspec
		gs1 = gridspec.GridSpec(3, 4)
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
					maxspec=50,base=self,verbose=self.verbose,datadict=datadict,binspecres=binspecres)
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
			datadict = readutils.rdAllData(snlist,self.options.estimate_tpk,self.kcordict,
										   dospec=self.options.dospec,KeepOnlySpec=self.options.keeponlyspec,
										   peakmjdlist=self.options.tmaxlist,waverange=self.options.waverange,
										   binspecres=binspecres)
			tlc = time()
			count = 0
			for l in snfiles:
				#if 'Foundation' not in l: continue
				if '/' not in l:
					l = f'{os.path.dirname(snlist)}/{l}'
				sn = snana.SuperNova(l)
				sn.SNID = str(sn.SNID)
				if not sn.SNID in datadict:
					continue

				if not i % 12:
					fig = plt.figure()
				try:
					ax1 = plt.subplot(gs1[i % 12]); ax2 = plt.subplot(gs1[(i+1) % 12]); ax3 = plt.subplot(gs1[(i+2) % 12]); ax4 = plt.subplot(gs1[(i+3) % 12])
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

				ValidateLightcurves.customfilt(
					f'{outputdir}/lccomp_{sn.SNID}.png',l,outputdir,
					t0=t0sn,x0=x0sn,x1=x1sn,c=csn,fitx1=fitx1,fitc=fitc,
					bandpassdict=self.kcordict,n_components=self.options.n_components,
					ax1=ax1,ax2=ax2,ax3=ax3,ax4=ax4,saltdict=saltdict,n_colorpars=self.options.n_colorpars)
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
			datadict = readutils.rdAllData(self.options.snlists,self.options.estimate_tpk,self.kcordict,
										   dospec=self.options.dospec,KeepOnlySpec=self.options.keeponlyspec,
										   peakmjdlist=self.options.tmaxlist,waverange=self.options.waverange,
										   binspecres=binspecres)
			log.info(f'took {time()-tdstart:.3f} to read in data files')
			tcstart = time()

			datadict = self.mkcuts(datadict,KeepOnlySpec=self.options.keeponlyspec)
			log.info(f'took {time()-tcstart:.3f} to apply cuts')

			# fit the model - initial pass
			if self.options.stage == "all" or self.options.stage == "train":
				# read the data
				stage='training'

				phase,wave,M0,M0modelerr,M0dataerr,M1,M1modelerr,M1dataerr,cov_M0_M1_model,cov_M0_M1_data,\
					modelerr,clpars,clerr,clscat,SNParams,pars,pars_unscaled,parlist,chain,loglikes = self.fitSALTModel(datadict)
				stage='output'
				# write the output model - M0, M1, c
				self.wrtoutput(self.options.outputdir,phase,wave,M0,M0modelerr,M0dataerr,M1,M1modelerr,M1dataerr,cov_M0_M1_model,cov_M0_M1_data,
							   modelerr,clpars,clerr,clscat,SNParams,
							   pars,pars_unscaled,parlist,chain,loglikes,datadict)
			log.info('successful SALT2 training!  Output files written to %s'%self.options.outputdir)
			if self.options.stage == "all" or self.options.stage == "validate":
				stage='validation'
				self.validate(self.options.outputdir,datadict)
		except:
			log.exception(f'Exception raised during {stage}')
		
	
