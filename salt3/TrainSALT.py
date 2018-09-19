#!/usr/bin/env python
# D. Jones, R. Kessler - 8/31/18
from __future__ import print_function

import os
import argparse
import configparser
import numpy as np
import sys
from salt3.util import snana
from salt3.util.estimate_tpk_bazin import estimate_tpk_bazin
from scipy.optimize import minimize
from salt3.fitting import saltfit
from astropy.io import fits

class TrainSALT:
	def __init__(self):
		pass
	
	def addwarning(self,warning):
		print(warning)
		self.warnings.append(warning)
		
	def add_options(self, parser=None, usage=None, config=None):
		if parser == None:
			parser = argparse.ArgumentParser(usage=usage, conflict_handler="resolve")

		# The basics
		parser.add_argument('-v', '--verbose', action="count", dest="verbose",
							default=1,help='verbosity level')
		parser.add_argument('--debug', default=False, action="store_true",
							help='debug mode: more output and debug files')
		parser.add_argument('--clobber', default=False, action="store_true",
							help='clobber')
		parser.add_argument('-c','--configfile', default=None, type=str,
							help='configuration file')

		# input/output files
		parser.add_argument('--snphotlist', default=config.get('iodata','snphotlist'), type=str,
							help="""list of SNANA-formatted photometric lightcurves to train on.  
List format should be space-delimited 'SNID Filename' (default=%default)""")
		parser.add_argument('--snspeclist', default=config.get('iodata','snspeclist'), type=str,
							help="""list of SN spectra to train on, spectroscopy format should be ASCII 
							with columns wavelength, flux, fluxerr (optional). List format should be space-delimited 'SNID Filename'.  (default=%default)""")
		parser.add_argument('--filterdefs', default=config.get('iodata','filterdefs'), type=str,
							help="""File with filter definitions (python dictionary).  Either gives throughput file for
							each survey filter, or a single kcor file for each survey.  See example data for help (default=%default)""")
		parser.add_argument('--outputdir', default=config.get('iodata','outputdir'), type=str,
							help="""data directory for spectroscopy, format should be ASCII 
							with columns wavelength, flux, fluxerr (optional) (default=%default)""")
		parser.add_argument('--initmodelfile', default=config.get('iodata','initmodelfile'), type=str,
							help="""initial model to begin training, ASCII with columns
							phase, wavelength, flux (default=%default)""")

		# training parameters
		parser.add_argument('--waverange', default=list(map(int,config.get('trainparams','waverange').split(','))), type=int, nargs=2,
							help='wavelength range over which the model is defined (default=%default)')
		parser.add_argument('--colorwaverange', default=list(map(int,config.get('trainparams','colorwaverange').split(','))), type=int, nargs=2,
							help='wavelength range over which the color law is fit to data (default=%default)')
		parser.add_argument('--interpfunc', default=config.get('trainparams','interpfunc'), type=str,
							help='function to interpolate between control points in the fitting (default=%default)')
		parser.add_argument('--interporder', default=config.get('trainparams','interporder'), type=int,
							help='for splines/polynomial funcs, order of the function (default=%default)')
		parser.add_argument('--waveres', default=config.get('trainparams','waveres'), type=float,
							help='number of angstroms between each wavelength control point (default=%default)')
		parser.add_argument('--phaseres', default=config.get('trainparams','phaseres'), type=float,
							help='number of angstroms between each phase control point (default=%default)')
		parser.add_argument('--phaserange', default=list(map(int,config.get('trainparams','phaserange').split(','))), type=int, nargs=2,
							help='phase range over which model is trained (default=%default)')
		parser.add_argument('--minmethod', default=config.get('trainparams','minmethod'), type=str, nargs=2,
							help='minimization algorithm, passed to scipy.optimize.minimize (default=%default)')

		return parser

	def rdkcor(self,filterdefs):

		self.kcordict = {}
		for k in filterdefs.keys():
			self.kcordict[k] = {}
			if 'kcor' in filterdefs[k] and filterdefs[k]['kcor']:
				kcorfile = os.path.expandvars(filterdefs[k]['kcor'])
				hdu = fits.open(kcorfile)
				zpoff = hdu[1].data
				snsed = hdu[2].data
				filtertrans = hdu[5].data
				primarysed = hdu[6].data

				self.kcordict[k]['filtwave'] = filtertrans['wavelength (A)']
				self.kcordict[k]['primarywave'] = primarysed['wavelength (A)']
				self.kcordict[k]['snflux'] = snsed['SN Flux (erg/s/cm^2/A)']
				if 'AB' in primarysed.names:
					self.kcordict[k]['AB'] = primarysed['AB']
				if 'Vega' in primarysed.names:
					self.kcordict[k]['Vega'] = primarysed['Vega']
				for filt in zpoff['Filter Name']:
					self.kcordict[k][filt.split('-')[-1]] = {}
					self.kcordict[k][filt.split('-')[-1]]['filttrans'] = filtertrans[filt]
					self.kcordict[k][filt.split('-')[-1]]['zpoff'] = zpoff['ZPOff(Primary)'][zpoff['Filter Name'] == filt][0]
					self.kcordict[k][filt.split('-')[-1]]['magsys'] = zpoff['Primary Name'][zpoff['Filter Name'] == filt][0]
		
	def rdAllData(self,snphotlist,snspeclist,filterdefs):
		datadict = {}
		if os.path.exists(snphotlist):
			photids,photfiles = np.loadtxt(snphotlist,dtype='str')
			photids = np.atleast_1d(photids); photfiles = np.atleast_1d(photfiles)
		else:
			photids,photfiles = [],[]
			if self.verbose: raise RuntimeError('no photometry provided')
		if os.path.exists(snspeclist):
			specids,specfiles = np.loadtxt(snspeclist,dtype='str')
			specids = np.atleast_1d(specids); specfiles = np.atleast_1d(specfiles)
		else:
			specids,specfiles = [],[]
			if self.verbose: self.addwarning('no spectra provided')
		for snid in np.unique(np.append(specids,photids)):
			PhotSNID = photfiles[photids == snid]
			if len(PhotSNID) > 1: raise RuntimeError('More than one LC file for SNID %s'%snid)
			SpecSNID = specfiles[specids == snid]
			if len(SpecSNID) > 1: raise RuntimeError('More than one spectrum for SNID %s'%snid)

			sn = snana.SuperNova(PhotSNID[0])
			if not 'SURVEY' in sn.__dict__.keys():
				raise RuntimeError('File %s has no SURVEY key, which is needed to find the filter transmission curves'%PhotSNID[0])
			if not 'REDSHIFT_HELIO' in sn.__dict__.keys():
				raise RuntimeError('File %s has no heliocentric redshift information in the header'%PhotSNID[0])
			zHel = float(sn.REDSHIFT_HELIO.split('+-')[0])
			tpk,tpkmsg = estimate_tpk_bazin(sn.MJD,sn.FLUXCAL,sn.FLUXCALERR)
			if 'termination condition is satisfied' not in tpkmsg:
				self.addwarning('skipping SN %s; can\'t estimate t_max')
				continue

			datadict[snid] = {'photometry_file':PhotSNID,
							  'spectrum_file':SpecSNID,
							  'zHelio':zHel,
							  'survey':sn.SURVEY}
			#datadict[snid]['zHelio'] = zHel
			
			# TODO: flux errors
			datadict[snid]['specdata'] = {}
			if len(SpecSNID):
				wave,flux = np.loadtxt(SpecSNID[0],unpack=True,usecols=[0,1])
				datadict[snid]['specdata']['wavelength'] = wave
				datadict[snid]['specdata']['flux'] = flux
				
			datadict[snid]['photdata'] = {}
			datadict[snid]['photdata']['tobs'] = sn.MJD - tpk
			datadict[snid]['photdata']['fluxcal'] = sn.FLUXCAL
			datadict[snid]['photdata']['fluxcalerr'] = sn.FLUXCALERR
			datadict[snid]['photdata']['filt'] = sn.FLT
				
		return datadict

	def fitSALTModel(self,datadict,phaserange,phaseres,waverange,waveres,minmethod,kcordict):

		n_phaseknots = int((phaserange[1]-phaserange[0])/phaseres)
		n_waveknots = int((waverange[1]-waverange[0])/waveres)
		n_sn = len(datadict.keys())

		# x1,x0,c for each SN
		# phase/wavelength spline knots for M0, M1 (ignoring color for now)
		# TODO: spectral recalibration
		n_params = 2*n_phaseknots*n_waveknots + 3*n_sn
		
		guess = np.ones(n_params)*1e-7
		parlist = ['m0']*(n_phaseknots*n_waveknots) +\
				  ['m1']*(n_phaseknots*n_waveknots)
		for k in datadict.keys():
			parlist += ['x0_%s'%k,'x1_%s'%k,'c_%s'%k]
		parlist = np.array(parlist)

		md = minimize(saltfit.chi2fit,guess,
                      args=(datadict,parlist,phaserange,
							waverange,phaseres,waveres,kcordict),
					  #bounds=bounds,
					  method=minmethod,
					  options={'maxiter':10000,'maxfev':10000})

		if 'success' not in md.message:
			self.addwarning('Minimizer message: %s'%md.message)
		
		wave,phase,M0,M1,SNParams = \
			saltfit.getPars(md,parlist)

		return wave,phase,M0,M1,SNParams

	def wrtoutput(self,outdir,wave,phase,M0,M1,SNParams):

		foutm0 = open('%s/salt2_template_0.dat'%outdir)
		foutm1 = open('%s/salt2_template_1.dat'%outdir)

		for p in phase:
			for w,m0,m1 in zip(wave,M0,M1):
				print('%.1f %.2f %8.5e'%(p,w,m0),file=foutm0)
				print('%.1f %.2f %8.5e'%(p,w,m1),file=foutm1)

		foutm0.close()
		foutm1.close()

		foutsn = open('%s/snparams.txt'%outdir)
		print('# SN x0 x1',file=foutsn)
		for k in SNParams.keys():
			print('%s %8.5e %8.5e'%(k,SNParams[k]['x0'],SNParams[k]['x1']))

		return

	def main(self):

		# get the filter functions
		sys.path.append(os.path.dirname(options.snphotlist))
		try: from FILTERDEFS import filterdefs
		except:
			raise RuntimeError("""Failed to import filter definitions from FILTERDEFS.py""")
		self.filterdefs = filterdefs	
		self.rdkcor(filterdefs)
		# TODO: ASCII filter files
		
		# read the data
		datadict = self.rdAllData(self.options.snphotlist,self.options.snspeclist,self.filterdefs)
		
		# fit the model - initial pass
		wave,phase,M0,M1,SNParams = self.fitSALTModel(
			datadict,self.options.phaserange,self.options.phaseres,
			self.options.waverange,self.options.waveres,
			self.options.minmethod,self.kcordict)
		
		# write the output model - M0, M1, c
		self.wrtoutput(self.options.outputdir,wave,phase,M0,M1,SNParams)

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

