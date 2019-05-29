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

from salt3.util import snana,readutils
from salt3.util.estimate_tpk_bazin import estimate_tpk_bazin
from scipy.optimize import minimize, least_squares, differential_evolution
from salt3.training.saltfit import fitting
from salt3.training.init_hsiao import init_hsiao, init_kaepora, init_errs, init_errs_fromfile
from salt3.training import saltfit
from salt3.training.base import TrainSALTBase
from salt3.initfiles import init_rootdir
from salt3.util.txtobj import txtobj
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from sncosmo.constants import HC_ERG_AA

class TrainSALT(TrainSALTBase):
	def __init__(self):
		self.warnings = []
		
	def fitSALTModel(self,datadict):
		
		if not os.path.exists(self.options.initmodelfile):
			from salt3.initfiles import init_rootdir
			self.options.inithsiaofile = '%s/hsiao07.dat'%(init_rootdir)
			self.options.initmodelfile = '%s/%s'%(init_rootdir,self.options.initmodelfile)
			self.options.initx1modelfile = '%s/%s'%(init_rootdir,self.options.initx1modelfile)
			flatnu='%s/flatnu.dat'%(init_rootdir)
			self.options.initbfilt = '%s/%s'%(init_rootdir,self.options.initbfilt)
			salt2file = '%s/salt2_template_0.dat'%init_rootdir
			salt2m1file = '%s/salt2_template_1.dat'%init_rootdir
		if not os.path.exists(self.options.initmodelfile):
			raise RuntimeError('model initialization file not found in local directory or %s'%init_rootdir)

		# initial guesses
		init_options = {'phaserange':self.options.phaserange,'waverange':self.options.waverange,
						'phasesplineres':self.options.phasesplineres,'wavesplineres':self.options.wavesplineres,
						'phaseinterpres':self.options.phaseoutres,'waveinterpres':self.options.waveoutres}
		
		#phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_kaepora(
		#	self.options.initmodelfile,self.options.initx1modelfile,salt2file,self.options.initbfilt,flatnu,**init_options)
		#phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_hsiao(
		#	self.options.inithsiaofile,salt2file,self.options.initbfilt,flatnu,**init_options)
		#print('hack: initial M1 params equal to salt2 M1')


		
		phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_hsiao(
			self.options.inithsiaofile,salt2file,self.options.initbfilt,flatnu,normalize=True,**init_options)
		#phase,wave,m1,mtemp,phaseknotloc,waveknotloc,m1knots,m1tmpknots = init_hsiao(
		#	salt2m1file,salt2file,self.options.initbfilt,flatnu,normalize=False,**init_options)
		#phase,wave,m0,m1,phaseknotloc,waveknotloc,m0knots,m1knots = init_kaepora(
		#	salt2file,salt2m1file,salt2file,self.options.initbfilt,flatnu,**init_options)

		init_options['phasesplineres'] = self.options.error_snake_phase_binsize
		init_options['wavesplineres'] = self.options.error_snake_wave_binsize
		errphaseknotloc,errwaveknotloc = init_errs(
			self.options.inithsiaofile,salt2file,self.options.initbfilt,**init_options)
		
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
			parlist = np.append(parlist,['modelerr']*n_errphaseknots*n_errwaveknots)
		if self.options.n_colorscatpars:
			# four knots for the end points
			parlist = np.append(parlist,['clscat']*(self.options.n_colorscatpars+8))

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
				if self.options.n_specrecal:
					parlist=np.append(parlist,['specrecal_{}_{}'.format(sn,k)]*order)


		# initial guesses
		n_params=parlist.size
		guess = np.zeros(parlist.size)
		m0knots[m0knots == 0] = 1e-4
		guess[parlist == 'm0'] = m0knots
		guess[parlist == 'modelerr'] = 1e-6 #*np.mean(m0knots)*HC_ERG_AA
		if self.options.n_components == 2:
			guess[parlist == 'm1'] = m1knots
		if self.options.n_colorpars:
			guess[parlist == 'cl'] = [0.]*self.options.n_colorpars
		if self.options.n_colorscatpars:
			guess[parlist == 'clscat'] = [0.]*self.options.n_colorscatpars

		guess[(parlist == 'm0') & (guess < 0)] = 1e-4
		for k in datadict.keys():
			guess[parlist == 'x0_%s'%k] = 10**(-0.4*(cosmo.distmod(datadict[k]['zHelio']).value-19.36-10.635))

		if self.options.resume_from_outputdir:
			names,pars = np.loadtxt('%s/salt3_parameters.dat'%self.options.outputdir,unpack=True,skiprows=1,dtype="U20,f8")
			for key in np.unique(parlist):
				guess[parlist == key] = pars[names == key]

		saltfitkwargs = self.get_saltkw(phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)

		print('training on %i SNe!'%len(datadict.keys()))
		if self.options.do_mcmc:
			saltfitter = saltfit.mcmc(guess,datadict,parlist,**saltfitkwargs)
			print('initial loglike: %.1f'%saltfitter.maxlikefit(guess,None,False))
		
			fitter = fitting(self.options.n_components,self.options.n_colorpars,
							 n_phaseknots,n_waveknots,
							 datadict)

			# do the fitting
			x_modelpars,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
				modelerr,clpars,clerr,clscat,SNParams,message = fitter.mcmc(
					saltfitter,guess,self.options.n_processes,
					self.options.n_steps_mcmc,self.options.n_burnin_mcmc)
			for k in datadict.keys():
				tpk_init = datadict[k]['photdata']['mjd'][0] - datadict[k]['photdata']['tobs'][0]
				#if not self.options.fix_t0:
				SNParams[k]['t0'] = -SNParams[k]['tpkoff'] + tpk_init
				#import pdb; pdb.set_trace()

		if self.options.do_gaussnewton:
			if not self.options.do_mcmc: x_modelpars = guess[:]
			saltfitter = saltfit.GaussNewton(x_modelpars,datadict,parlist,**saltfitkwargs)
			fitter = fitting(self.options.n_components,self.options.n_colorpars,
							 n_phaseknots,n_waveknots,
							 datadict)
			
			# do the fitting
			x_modelpars,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
				modelerr,clpars,clerr,clscat,SNParams,message = fitter.gaussnewton(
					saltfitter,guess,self.options.n_processes,
					self.options.n_steps_mcmc,self.options.n_burnin_mcmc)
			for k in datadict.keys():
				tpk_init = datadict[k]['photdata']['mjd'][0] - datadict[k]['photdata']['tobs'][0]
				SNParams[k]['t0'] = -SNParams[k]['tpkoff'] + tpk_init

		
		print('MCMC message: %s'%message)
		print('Final regularization chi^2 terms:', saltfitter.regularizationChi2(x_modelpars,1,0,0),
			  saltfitter.regularizationChi2(x_modelpars,0,1,0),saltfitter.regularizationChi2(x_modelpars,0,0,1))
		print('Final loglike'); saltfitter.maxlikefit(x_modelpars,None,False)

		print(x_modelpars.size)

		
		return phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
			modelerr,clpars,clerr,clscat,SNParams,x_modelpars,parlist,saltfitter.chain,saltfitter.loglikes

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

				foutpars.write('{: <30} {:.6e}\n'.format(name,par))
		
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
				print('%.1f %.2f %8.5e'%(p,w,M0[i,j]),file=foutm0)
				print('%.1f %.2f %8.5e'%(p,w,M1[i,j]),file=foutm1)
				print('%.1f %.2f %8.5e'%(p,w,M0err[i,j]),file=foutm0err)
				print('%.1f %.2f %8.5e'%(p,w,M1err[i,j]),file=foutm1err)
				print('%.1f %.2f %8.5e'%(p,w,cov_M0_M1[i,j]),file=foutcov)
				print('%.1f %.2f %8.5e'%(p,w,modelerr[i,j]),file=fouterrmod)

		foutclscat = open('%s/salt3_color_dispersion.dat'%outdir,'w')
		for w,j in zip(wave,range(len(wave))):
			print('%.2f %8.5e'%(w,clscat[j]),file=foutclscat)
		foutclscat.close()
				
		foutm0.close()
		foutm1.close()
		foutm0err.close()
		foutm1err.close()
		foutcov.close()
		fouterrmod.close()
		
		print('%i'%len(clpars),file=foutcl)
		for c in clpars:
			print('%8.5e'%c,file=foutcl)
		print("""Salt2ExtinctionLaw.version 1
Salt2ExtinctionLaw.min_lambda %i
Salt2ExtinctionLaw.max_lambda %i"""%(
	self.options.colorwaverange[0],
	self.options.colorwaverange[1]),file=foutcl)
		foutcl.close()

		#for c in foutclscat
		foutclscat.close()
		
		# best-fit and simulated SN params
		snfiles = np.genfromtxt(self.options.snlist,dtype='str')
		snfiles = np.atleast_1d(snfiles)
		
		foutsn = open('%s/salt3train_snparams.txt'%outdir,'w')
		print('# SN x0 x1 c t0 tpkoff SIM_x0 SIM_x1 SIM_c SIM_t0',file=foutsn)
		for k in SNParams.keys():
			foundfile = False
			for l in snfiles:
				if str(k) not in l: continue
				foundfile = True
				if '/' not in l:
					l = '%s/%s'%(os.path.dirname(self.options.snlist),l)
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
			print('%s %8.5e %.4f %.4f %.2f %.2f %8.5e %.4f %.4f %.2f'%(
				k,SNParams[k]['x0'],SNParams[k]['x1'],SNParams[k]['c'],SNParams[k]['t0'],
				SNParams[k]['tpkoff'],SIM_x0,SIM_x1,SIM_c,SIM_PEAKMJD),file=foutsn)
		foutsn.close()
			
		return

	def validate(self,outputdir):

		import pylab as plt
		plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=0.025, hspace=0)

		#plt.ion()
		
		from salt3.validation import ValidateLightCurves
		from salt3.validation import ValidateSpectra
		from salt3.validation import ValidateModel

		x0,x1,c,t0 = np.loadtxt('%s/salt3train_snparams.txt'%outputdir,unpack=True,usecols=[1,2,3,4])
		snid = np.genfromtxt('%s/salt3train_snparams.txt'%outputdir,unpack=True,dtype='str',usecols=[0])
		
		ValidateModel.main(
			'%s/spectralcomp.png'%outputdir,
			outputdir)

		ValidateSpectra.compareSpectra(self.options.speclist,
									   self.options.outputdir)
		
		snfiles = np.genfromtxt(self.options.snlist,dtype='str')
		snfiles = np.atleast_1d(snfiles)
		fitx1,fitc = False,False
		if self.options.n_components == 2:
			fitx1 = True
		if self.options.n_colorpars > 0:
			fitc = True

		from salt3.util.synphot import synphot
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
		for l in snfiles:
			if not i % 9:
				fig = plt.figure()
			try:
				ax1 = plt.subplot(gs1[i % 9]); ax2 = plt.subplot(gs1[(i+1) % 9]); ax3 = plt.subplot(gs1[(i+2) % 9])
			except:
				import pdb; pdb.set_trace()

			if '/' not in l:
				l = '%s/%s'%(os.path.dirname(self.options.snlist),l)
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
			
			ValidateLightCurves.customfilt(
				'%s/lccomp_%s.png'%(outputdir,sn.SNID),l,outputdir,
				t0=t0sn,x0=x0sn,x1=x1sn,c=csn,fitx1=fitx1,fitc=fitc,
				bandpassdict=self.kcordict,n_components=self.options.n_components,
				ax1=ax1,ax2=ax2,ax3=ax3)
			if i % 9 == 6:
				pdf_pages.savefig()
			else:
				for ax in [ax1,ax2,ax3]:
					ax.xaxis.set_ticklabels([])
					ax.set_xlabel(None)
			i += 3

		pdf_pages.savefig()
		pdf_pages.close()
		
	def main(self):

		if not len(self.surveylist):
			raise RuntimeError('surveys are not defined - see documentation')
		self.kcordict=readutils.rdkcor(self.surveylist,self.options,addwarning=self.addwarning)
		# TODO: ASCII filter files
		
		# read the data
		datadict = readutils.rdAllData(self.options.snlist,self.options.estimate_tpk,self.kcordict,
									   self.addwarning,speclist=self.options.speclist)
		
		if not os.path.exists(self.options.outputdir):
			os.makedirs(self.options.outputdir)

		datadict = self.mkcuts(datadict)
		
		# fit the model - initial pass
		if self.options.stage == "all" or self.options.stage == "train":
			phase,wave,M0,M0err,M1,M1err,cov_M0_M1,\
				modelerr,clpars,clerr,clscat,SNParams,pars,parlist,chain,loglikes = self.fitSALTModel(datadict)
		
			# write the output model - M0, M1, c
			self.wrtoutput(self.options.outputdir,phase,wave,M0,M0err,M1,M1err,cov_M0_M1,
						   modelerr,clpars,clerr,clscat,SNParams,
						   pars,parlist,chain,loglikes)

		if self.options.stage == "all" or self.options.stage == "validate":
			self.validate(self.options.outputdir)
		
		print('successful SALT2 training!  Output files written to %s'%self.options.outputdir)
		
	
if __name__ == "__main__":
	usagestring = """SALT3 Training

usage: python TrainSALT.py -c <configfile> <options>

config file options can be overwridden at the command line

Dependencies: sncosmo?
"""

	if sys.version_info < (3,0):
		sys.exit('Sorry, Python 2 is not supported')
	
	salt = TrainSALT()

	parser = argparse.ArgumentParser(usage=usagestring, conflict_handler="resolve")
	parser.add_argument('-c','--configfile', default=None, type=str,
					  help='configuration file')
	options, args = parser.parse_known_args()

	if options.configfile:
		config = configparser.ConfigParser()
		if not os.path.exists(options.configfile):
			raise RuntimeError('Configfile doesn\'t exist!')
		config.read(options.configfile)
	else:
		parser.print_help()
		raise RuntimeError('Configuration file must be specified at command line')

	parser = salt.add_options(usage=usagestring,config=config)
	options = parser.parse_args()
	
	salt.options = options
	salt.verbose = options.verbose
	salt.clobber = options.clobber
	
	salt.main()

	if len(salt.warnings):
		print('There were warnings!!')
		print(salt.warnings)

