#!/usr/bin/env python
# D. Jones, R. Kessler - 8/31/18

import exceptions
import os
import argparse
import configparser
import numpy as np

class TrainSALT:
	def __init__(self):
		pass

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
		parser.add_argument('--photdatadir', default=config.get('iodata','photdatadir'), type=str,
							help='data directory for SNANA-formatted photometric lightcurves (default=%default)')
		parser.add_argument('--specdatadir', default=config.get('iodata','specdatadir'), type=str,
							help="""data directory for spectroscopy, format should be ASCII 
							with columns wavelength, flux, fluxerr (optional) (default=%default)""")
		parser.add_argument('--outputdir', default=config.get('iodata','outputdir'), type=str,
							help="""data directory for spectroscopy, format should be ASCII 
							with columns wavelength, flux, fluxerr (optional) (default=%default)""")
		parser.add_argument('--initmodelfile', default=config.get('iodata','initmodelfile'), type=str,
							help="""initial model to begin training, ASCII with columns
							phase, wavelength, flux (default=%default)""")

		# training parameters
		parser.add_argument('--waverange', default=config.get('trainparams','waverange'), type=str, nargs=2,
							help='wavelength range over which the model is defined (default=%default)')
		parser.add_argument('--colorwaverange', default=config.get('trainparams','colorwaverange'), type=str, nargs=2,
							help='wavelength range over which the color law is fit to data (default=%default)')
		parser.add_argument('--interpfunc', default=config.get('trainparams','interpfunc'), type=str,
							help='function to interpolate between control points in the fitting (default=%default)')
		parser.add_argument('--interporder', default=config.get('trainparams','interporder'), type=int,
							help='for splines/polynomial funcs, order of the function (default=%default)')
		parser.add_argument('--waveres', default=config.get('trainparams','waveres'), type=int,
							help='number of angstroms between each wavelength control point (default=%default)')
		parser.add_argument('--phaseres', default=config.get('trainparams','phaseres'), type=int,
							help='number of angstroms between each phase control point (default=%default)')
		parser.add_argument('--phaserange', default=config.get('trainparams','phaserange'), type=int, nargs=2,
							help='phase range over which model is trained (default=%default)')


		return parser

	def main(self):
		pass
	
if __name__ == "__main__":
	usagestring = """SALT3 Training

usage: python TrainSALT.py -c <configfile> <options>

config file options can be overwridden at the command line

Dependencies: sncosmo?
"""

	salt = TrainSALT()

	parser = argparse.ArgumentParser(usage=usagestring, conflict_handler="resolve")
	parser.add_argument('-c','--configfile', default=None, type=str,
					  help='configuration file')
	options, args = parser.parse_known_args()

	if options.configfile:
		config = configparser.ConfigParser()
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
