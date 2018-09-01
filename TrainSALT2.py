#!/usr/bin/env python
# D. Jones, R. Kessler - 8/31/18

import exceptions
import os
import argparse
import configparser
import numpy as np

class TrainSALT2:
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

		#if config:
		parser.add_argument('--photdatadir', default=config.get('inputdata','photdatadir'), type=str,
							help='data directory for SNANA-formatted photometric lightcurves (default=%default)')
		parser.add_argument('--specdatadir', default=config.get('inputdata','specdatadir'), type=str,
							help="""data directory for spectroscopy, format should be ASCII 
							with columns wavelength, flux, fluxerr (optional) (default=%default)""")

		return parser

	def main(self):
		pass
	
if __name__ == "__main__":
	usagestring = """SALT2 Training

usage: python TrainSALT2.py -c <configfile> <options>

config file options can be overwridden at the command line

Dependencies: sncosmo?
"""

	salt = TrainSALT2()

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
