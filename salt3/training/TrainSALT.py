#!/usr/bin/env python
# D. Jones, R. Kessler - 8/31/18
from __future__ import print_function

import os
import argparse
import configparser
import numpy as np
import sys
import multiprocessing

from scipy.linalg import lstsq
from scipy.optimize import minimize, least_squares, differential_evolution
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from sncosmo.constants import HC_ERG_AA

from salt3.util import snana,readutils
from salt3.util.estimate_tpk_bazin import estimate_tpk_bazin
from salt3.util.txtobj import txtobj
from salt3.util.specSynPhot import getScaleForSN

from salt3.training.init_hsiao import init_hsiao, init_kaepora, init_errs, init_errs_fromfile
from salt3.training.base import TrainSALTBase
from salt3.training.saltfit import fitting
from salt3.training import saltfit as saltfit

from salt3.data import data_rootdir
from salt3.initfiles import init_rootdir
from salt3.config import config_rootdir

# validation utils
import pylab as plt
from salt3.validation import ValidateLightcurves
from salt3.validation import ValidateSpectra
from salt3.validation import ValidateModel
from salt3.validation.figs import plotSALTModel
from salt3.util.synphot import synphot
from salt3.initfiles import init_rootdir as salt2dir
from time import time

class TrainSALT(TrainSALTBase):
	def __init__(self):
		self.warnings = []
	
	def initialParameters(self,datadict):
		from salt3.initfiles import init_rootdir
		self.options.inithsiaofile = '%s/hsiao07.dat'%(init_rootdir)
		flatnu='%s/flatnu.dat'%(init_rootdir)
		self.options.initbfilt = '%s/%s'%(init_rootdir,self.options.initbfilt)
		if self.options.initm0modelfile and not os.path.exists(self.options.initm0modelfile):
			if self.options.initm0modelfile:
				self.options.initm0modelfile = '%s/%s'%(init_rootdir,self.options.initm0modelfile)
			if self.options.initm1modelfile:
				self.options.initm1modelfile = '%s/%s'%(init_rootdir,self.options.initm1modelfile)
		
		if self.options.initm0modelfile and not os.path.exists(self.options.initm0modelfile):
			raise RuntimeError('model initialization file not found in local directory or %s'%init_rootdir)

		# initial guesses
		init_options = {'phaserange':self.options.phaserange,'waverange':self.options.waverange,
						'phasesplineres':self.options.phasesplineres,'wavesplineres':self.options.wavesplineres,
						'phaseinterpres':self.options.phaseoutres,'waveinterpres':self.options.waveoutres,
						'normalize':True}
				
		phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_hsiao(
			self.options.inithsiaofile,self.options.initbfilt,flatnu,**init_options)
		if self.options.initm1modelfile:
			phase,wave,m1,mtemp,phaseknotloc,waveknotloc,m1knots,m1tmpknots = init_hsiao(
				self.options.initm1modelfile,
				self.options.initbfilt,flatnu,**init_options)
		if self.options.initm0modelfile:
			phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_kaepora(
				self.options.initm0modelfile,self.options.initm1modelfile,self.options.initbfilt,flatnu,**init_options)
		#import pdb; pdb.set_trace()
			
		init_options['phasesplineres'] = self.options.error_snake_phase_binsize
		init_options['wavesplineres'] = self.options.error_snake_wave_binsize
		
		errphaseknotloc = np.array([-20., -20., -20., -20.,-3.52941176, 0.58823529,   4.70588235,   8.82352941,
									12.94117647,  17.05882353,  21.17647059,  25.29411765,        29.41176471,
									85.        ,  85.        ,  85.        ,85.]) 
		errwaveknotloc = np.array([ 1000.,  1000.,  1000.,  1000.,  3600.,  4000.,  4400.,  4800.,
									5200.,  5600.,  6000.,  6400.,  6800.,  7200., 25000., 25000.,
									25000., 25000.])
		# number of parameters
		n_phaseknots,n_waveknots = len(phaseknotloc)-4,len(waveknotloc)-4
		n_errphaseknots,n_errwaveknots = len(errphaseknotloc)-4,len(errwaveknotloc)-4
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

		# spectral params
		for sn in datadict.keys():
			specdata=datadict[sn]['specdata']
			photdata=datadict[sn]['photdata']
			for k in specdata.keys():
				order=self.options.n_min_specrecal+int(np.log((specdata[k]['wavelength'].max() - \
					specdata[k]['wavelength'].min())/self.options.specrange_wavescale_specrecal) + \
					np.unique(photdata['filt']).size* self.options.n_specrecal_per_lightcurve)
				if self.options.specrecal:
					parlist=np.append(parlist,['specrecal_{}_{}'.format(sn,k)]*order)

		# initial guesses
		n_params=parlist.size
		guess = np.zeros(parlist.size)
		m0knots[m0knots == 0] = 1e-4
		guess[parlist == 'm0'] = m0knots
		for i in range(3): guess[parlist == 'modelerr_{}'.format(i)] = 1e-6 
		if self.options.n_components == 2:
			guess[parlist == 'm1'] = m1knots*1e3
		if self.options.n_colorpars:
			guess[parlist == 'cl'] = [-0.504294, 0.787691, -0.461715, 0.0815619] #[0.]*self.options.n_colorpars
		if self.options.n_colorscatpars:
			guess[parlist == 'clscat'] = [1e-6]*self.options.n_colorscatpars
			guess[np.where(parlist == 'clscat')[0][-1]]=-10
		guess[(parlist == 'm0') & (guess < 0)] = 1e-4

		if self.options.specrecal:
			for sn in datadict.keys():
				specdata=datadict[sn]['specdata']
				photdata=datadict[sn]['photdata']
				for k in specdata.keys():
					print(sn)
					init_scale,colordiffs = getScaleForSN(specdata[k],photdata,self.kcordict,datadict[sn]['survey'])
					guess[np.where(parlist == 'specrecal_{}_{}'.format(sn,k))[0][-1]] = init_scale
		i=0
		for k in datadict.keys():
			guess[parlist == 'x0_%s'%k] = 10**(-0.4*(cosmo.distmod(datadict[k]['zHelio']).value-19.36-10.635))
			i+=1

		if self.options.resume_from_outputdir:
			try:
				names,pars = np.loadtxt('%s/salt3_parameters.dat'%self.options.resume_from_outputdir,unpack=True,skiprows=1,dtype="U30,f8")
			except:
				names,pars = np.loadtxt('%s/salt3_parameters.dat'%self.options.outputdir,unpack=True,skiprows=1,dtype="U30,f8")
			for key in np.unique(parlist):
				try:
					#if 'specrecal' not in key: 
					guess[parlist == key] = pars[names == key]
					
				except:
					print ('Problem while initializing parameter ',key,' from previous training')
					import pdb;pdb.set_trace()
					sys.exit(1)
					
		return parlist,guess,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc
	
	def fitSALTModel(self,datadict):
		parlist,x_modelpars,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc = self.initialParameters(datadict)
		saltfitkwargs = self.get_saltkw(phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)
		n_phaseknots,n_waveknots = len(phaseknotloc)-4,len(waveknotloc)-4
		n_errphaseknots,n_errwaveknots = len(errphaseknotloc)-4,len(errwaveknotloc)-4

		fitter = fitting(self.options.n_components,self.options.n_colorpars,
						 n_phaseknots,n_waveknots,
						 datadict)
		print('training on %i SNe!'%len(datadict.keys()))
		for i in range(self.options.n_repeat):
			if i == 0: laststepsize = None
			
			if self.options.do_mcmc:
				saltfitkwargs['regularize'] = False
				saltfitter = saltfit.mcmc(x_modelpars,datadict,parlist,**saltfitkwargs)
				print('initial loglike: %.1f'%saltfitter.maxlikefit(x_modelpars,{},np.zeros(len(x_modelpars),dtype=bool)))
				# do the fitting
				x_modelpars,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
					modelerr,clpars,clerr,clscat,SNParams,message = fitter.mcmc(
						saltfitter,x_modelpars,self.options.n_processes,
						self.options.n_steps_mcmc,self.options.n_burnin_mcmc)#,
						#stepsizes=laststepsize)

			if self.options.do_gaussnewton:
				saltfitkwargs['regularize'] = self.options.regularize
				saltfitter = saltfit.GaussNewton(x_modelpars,datadict,parlist,**saltfitkwargs)			
				# do the fitting
				x_modelpars,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
					modelerr,clpars,clerr,clscat,SNParams,laststepsize,message = fitter.gaussnewton(
						saltfitter,x_modelpars,self.options.n_processes,
						self.options.n_steps_mcmc,self.options.n_burnin_mcmc,
						self.options.gaussnewton_maxiter)
			for k in datadict.keys():
				try:
					tpk_init = datadict[k]['photdata']['mjd'][0] - datadict[k]['photdata']['tobs'][0]
					SNParams[k]['t0'] = -SNParams[k]['tpkoff'] + tpk_init
				except:
					SNParams[k]['t0'] = -99
		
		print('MCMC message: %s'%message)
		#print('Final regularization chi^2 terms:', saltfitter.regularizationChi2(x_modelpars,1,0,0),
		#	  saltfitter.regularizationChi2(x_modelpars,0,1,0),saltfitter.regularizationChi2(x_modelpars,0,0,1))
		print('Final loglike'); saltfitter.maxlikefit(x_modelpars)

		print(x_modelpars.size)


		#print('Finishing...To write files, hit c')
		#import pdb; pdb.set_trace()

		if 'chain' in saltfitter.__dict__.keys():
			chain = saltfitter.chain
			loglikes = saltfitter.loglikes
		else: chain,loglikes = None,None

		return phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams,x_modelpars,parlist,chain,loglikes

	def wrtoutput(self,outdir,phase,wave,
				  M0,M0err,M1,M1err,cov_M0_M1,
				  modelerr,clpars,
				  clerr,clscat,SNParams,pars,parlist,chain,loglikes):
		if not os.path.exists(outdir):
			raise RuntimeError('desired output directory %s doesn\'t exist'%outdir)

		#Save final model parameters
		
		with  open('{}/salt3_parameters.dat'.format(outdir),'w') as foutpars:
			foutpars.write('{: <30} {}\n'.format('Parameter Name','Value'))
			for name,par in zip(parlist,pars):

				foutpars.write('{: <30} {:.15e}\n'.format(name,par))
		
		#Save mcmc chain and log_likelihoods
		
		np.save('{}/salt3_mcmcchain.npy'.format(outdir),chain)
		np.save('{}/salt3_loglikes.npy'.format(outdir),loglikes)
		
		# principal components and color law
		foutm0 = open('%s/salt3_template_0.dat'%outdir,'w')
		foutm1 = open('%s/salt3_template_1.dat'%outdir,'w')
		foutm0err = open('%s/salt3_lc_relative_variance_0.dat'%outdir,'w')
		foutm1err = open('%s/salt3_lc_relative_variance_1.dat'%outdir,'w')
		fouterrmod = open('%s/salt3_lc_dispersion_scaling.dat'%outdir,'w')
		foutcov = open('%s/salt3_lc_relative_covariance_01.dat'%outdir,'w')
		foutcl = open('%s/salt3_color_correction.dat'%outdir,'w')

		for p,i in zip(phase,range(len(phase))):
			for w,j in zip(wave,range(len(wave))):
				print('%.1f %.2f %8.15e'%(p,w,M0[i,j]),file=foutm0)
				print('%.1f %.2f %8.15e'%(p,w,M1[i,j]),file=foutm1)
				print('%.1f %.2f %8.15e'%(p,w,M0err[i,j]**2.),file=foutm0err)
				print('%.1f %.2f %8.15e'%(p,w,M1err[i,j]**2.),file=foutm1err)
				print('%.1f %.2f %8.15e'%(p,w,cov_M0_M1[i,j]),file=foutcov)
				print('%.1f %.2f %8.15e'%(p,w,modelerr[i,j]),file=fouterrmod)

		foutclscat = open('%s/salt3_color_dispersion.dat'%outdir,'w')
		for w,j in zip(wave,range(len(wave))):
			print('%.2f %8.15e'%(w,clscat[j]),file=foutclscat)
		foutclscat.close()
				
		foutm0.close()
		foutm1.close()
		foutm0err.close()
		foutm1err.close()
		foutcov.close()
		fouterrmod.close()
		
		print('%i'%len(clpars),file=foutcl)
		for c in clpars:
			print('%8.10e'%c,file=foutcl)
		print("""Salt2ExtinctionLaw.version 1
Salt2ExtinctionLaw.min_lambda %i
Salt2ExtinctionLaw.max_lambda %i"""%(
	self.options.colorwaverange[0],
	self.options.colorwaverange[1]),file=foutcl)
		foutcl.close()

		#for c in foutclscat
		foutclscat.close()
		
		# best-fit and simulated SN params
		foutsn = open('%s/salt3train_snparams.txt'%outdir,'w')
		print('# SN x0 x1 c t0 tpkoff SIM_x0 SIM_x1 SIM_c SIM_t0',file=foutsn)
		for snlist in self.options.snlists.split(','):
			snlist = os.path.expandvars(snlist)
			if not os.path.exists(snlist):
				print('SN list file %s does not exist.	Checking %s/trainingdata/%s'%(snlist,data_rootdir,snlist))
				snlist = '%s/trainingdata/%s'%(data_rootdir,snlist)
				if not os.path.exists(snlist):
					raise RuntimeError('SN list file %s does not exist'%snlist)


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
				if not foundfile: SIM_x0,SIM_x1,SIM_c,SIM_PEAKMJD = -99,-99,-99,-99

				if 't0' not in SNParams[k].keys():
					SNParams[k]['t0'] = 0.0
				print('%s %8.10e %.10f %.10f %.10f %.10f %8.10e %.10f %.10f %.2f'%(
					k,SNParams[k]['x0'],SNParams[k]['x1'],SNParams[k]['c'],SNParams[k]['t0'],
					SNParams[k]['tpkoff'],SIM_x0,SIM_x1,SIM_c,SIM_PEAKMJD),file=foutsn)
		foutsn.close()
			
		return

	def validate(self,outputdir):

		# prelims
		plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=0.025, hspace=0)
		x0,x1,c,t0 = np.loadtxt('%s/salt3train_snparams.txt'%outputdir,unpack=True,usecols=[1,2,3,4])
		snid = np.genfromtxt('%s/salt3train_snparams.txt'%outputdir,unpack=True,dtype='str',usecols=[0])

		# have stopped really looking at these for now
		#ValidateModel.main(
		#	'%s/spectralcomp.png'%outputdir,
		#	outputdir)
		#ValidateModel.m0m1_chi2(
		#	'%s/spectralcomp_chi2.png'%outputdir,
		#	outputdir)

		plotSALTModel.mkModelPlot(outputdir,outfile='%s/SALTmodelcomp.pdf'%outputdir,
								  xlimits=[self.options.waverange[0],self.options.waverange[1]])

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

		pdf_pages = PdfPages('%s/lcfits.pdf'%outputdir)
		import matplotlib.gridspec as gridspec
		gs1 = gridspec.GridSpec(3, 3)
		gs1.update(wspace=0.0)
		i = 0
			
		for j,snlist in enumerate(self.options.snlists.split(',')):
			snlist = os.path.expandvars(snlist)
			if not os.path.exists(snlist):
				print('SN list file %s does not exist.	Checking %s/trainingdata/%s'%(snlist,data_rootdir,snlist))
				snlist = '%s/trainingdata/%s'%(data_rootdir,snlist)
				if not os.path.exists(snlist):
					raise RuntimeError('SN list file %s does not exist'%snlist)

			tspec = time()
			if self.options.dospec:
				ValidateSpectra.compareSpectra(
					snlist,self.options.outputdir,specfile='%s/speccomp_%i.pdf'%(self.options.outputdir,j),
					maxspec=50,base=self,verbose=self.verbose)
			print('plotting spectra took %.1f'%(time()-tspec))
				
			snfiles = np.genfromtxt(snlist,dtype='str')
			snfiles = np.atleast_1d(snfiles)
			fitx1,fitc = False,False
			if self.options.n_components == 2:
				fitx1 = True
			if self.options.n_colorpars > 0:
				fitc = True


			# read in and save SALT2 files
			m0file='salt3_template_0.dat'
			m1file='salt3_template_1.dat'
			salt3phase,salt3wave,salt3flux = np.genfromtxt('%s/%s'%(outputdir,m0file),unpack=True)
			salt3m1phase,salt3m1wave,salt3m1flux = np.genfromtxt('%s/%s'%(outputdir,m1file),unpack=True)
			salt2phase,salt2wave,salt2flux = np.genfromtxt('{}/salt2_template_0.dat'.format(salt2dir),unpack=True)
			salt2m1phase,salt2m1wave,salt2m1flux = np.genfromtxt('{}/salt2_template_1.dat'.format(salt2dir),unpack=True)
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

			tlc = time()
			for l in snfiles:
				if not i % 9:
					fig = plt.figure()
				try:
					ax1 = plt.subplot(gs1[i % 9]); ax2 = plt.subplot(gs1[(i+1) % 9]); ax3 = plt.subplot(gs1[(i+2) % 9])
				except:
					import pdb; pdb.set_trace()

				if '/' not in l:
					l = '%s/%s'%(os.path.dirname(snlist),l)
				sn = snana.SuperNova(l)
				sn.SNID = str(sn.SNID)

				if sn.SNID not in snid:
					self.addwarning('sn %s not in output files'%sn.SNID)
					continue
				x0sn,x1sn,csn,t0sn = \
					x0[snid == sn.SNID][0],x1[snid == sn.SNID][0],\
					c[snid == sn.SNID][0],t0[snid == sn.SNID][0]
				if not fitc: csn = 0
				if not fitx1: x1sn = 0

				ValidateLightcurves.customfilt(
					'%s/lccomp_%s.png'%(outputdir,sn.SNID),l,outputdir,
					t0=t0sn,x0=x0sn,x1=x1sn,c=csn,fitx1=fitx1,fitc=fitc,
					bandpassdict=self.kcordict,n_components=self.options.n_components,
					ax1=ax1,ax2=ax2,ax3=ax3,saltdict=saltdict)
				if i % 9 == 6:
					pdf_pages.savefig()
					plt.close('all')
				else:
					for ax in [ax1,ax2,ax3]:
						ax.xaxis.set_ticklabels([])
						ax.set_xlabel(None)
				i += 3

		pdf_pages.savefig()
		pdf_pages.close()
		print('plotting light curves took %.1f'%(time()-tlc))
		
	def main(self):

		if not len(self.surveylist):
			raise RuntimeError('surveys are not defined - see documentation')
		self.kcordict=readutils.rdkcor(self.surveylist,self.options,addwarning=self.addwarning)
		# TODO: ASCII filter files
				
		if not os.path.exists(self.options.outputdir):
			os.makedirs(self.options.outputdir)

		
		# fit the model - initial pass
		if self.options.stage == "all" or self.options.stage == "train":
			# read the data
			if self.options.binspec:
				binspecres = self.options.binspecres
			else:
				binspecres = None
				
			datadict = readutils.rdAllData(self.options.snlists,self.options.estimate_tpk,self.kcordict,
										   self.addwarning,dospec=self.options.dospec,KeepOnlySpec=self.options.keeponlyspec,
										   peakmjdlist=self.options.tmaxlist,waverange=self.options.waverange,
										   binspecres=binspecres)
			datadict = self.mkcuts(datadict,KeepOnlySpec=self.options.keeponlyspec)

			phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
				modelerr,clpars,clerr,clscat,SNParams,pars,parlist,chain,loglikes = self.fitSALTModel(datadict)
		
			# write the output model - M0, M1, c
			self.wrtoutput(self.options.outputdir,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,
						   modelerr,clpars,clerr,clscat,SNParams,
						   pars,parlist,chain,loglikes)

		if self.options.stage == "all" or self.options.stage == "validate":
			self.validate(self.options.outputdir)
		
		print('successful SALT2 training!  Output files written to %s'%self.options.outputdir)
		
	
