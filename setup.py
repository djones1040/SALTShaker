from setuptools.command.test import test as TestCommand
from distutils.core import setup, Extension
# import numpy.distutils.misc_util
import sys
from setuptools import find_packages

if sys.version_info < (3,0):
	sys.exit('Sorry, Python 2 is not supported')

class SALTShakerTest(TestCommand):

	def run_tests(self):
		import shaker
		errno = shaker.test()
		sys.exit(errno)

AUTHOR = 'David Jones, Rick Kessler'
AUTHOR_EMAIL = 'dojones@hawaii.edu'
VERSION = '1.4'
LICENSE = 'BSD'
URL = 'saltshaker.readthedocs.org'

setup(
	name='saltshaker-sn',
	version=VERSION,
        packages=find_packages('.'),
	#packages=['bin','saltshaker','saltshaker.tests','saltshaker.data','saltshaker.simulation',
        #          'saltshaker.training','saltshaker.util','saltshaker.initfiles',
        #          'saltshaker.validation','saltshaker.pipeline','saltshaker.config'],
	cmdclass={'test': SALTShakerTest},
        #entry_points={'console_scripts':['trainsalt = saltshaker.scripts.trainsalt']},
        scripts=['saltshaker/scripts/trainsalt','saltshaker/scripts/runpipe'],
#	scripts=['bin/trainsalt','bin/runpipe','saltshaker/validation/SynPhotPlot.py','saltshaker/validation/ValidateLightcurves.py','saltshaker/validation/ValidateModel.py','saltshaker/validation/ValidateSpectra.py','saltshaker/validation/figs/plotSALTModel.py'],
	#package_data={'': ['saltshaker/initfiles/*.dat','saltshaker/initfiles/*.txt','saltshaker/data/kcor/*.fits','saltshaker/config/*conf','saltshaker/scripts/*']},
    package_data={'saltshaker.initfiles': ['*.dat','*.txt'],
                  'saltshaker.data.kcor':['*.fits'],
                  'saltshaker.config':['*.conf']},
	include_package_data=True,
	author=AUTHOR,
	author_email=AUTHOR_EMAIL,
	license=LICENSE,
	long_description=open('README.md').read(),
#         include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),

	install_requires=['astropy>=5.3',
                      'astroquery>=0.4.6',
                      'Cython>=0.29.35',
                      'emcee>=3.1.4',
                      'extinction>=0.4.6',
                      'f90nml>=0.1',
                      'iminuit>=2.16.0',
                      'jax>=0.4.11',
                      'jaxlib>=0.4.11',
                      'matplotlib>=3.7.1',
                      'pandas>=2.0.2',
                      'pyParz>=1.0.2',
                      'pytest>=7.3.1',
                      'sncosmo>=2.10.0',
                      'tqdm>=4.65.0',
                      'scipy>=1.10.1'])
