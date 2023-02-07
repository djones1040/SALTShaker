from setuptools.command.test import test as TestCommand
from distutils.core import setup, Extension
import numpy.distutils.misc_util
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
AUTHOR_EMAIL = 'david.jones@ucsc.edu'
VERSION = '0.1dev'
LICENSE = 'BSD'
URL = 'saltshaker.readthedocs.org'

setup(
	name='saltshaker',
	version=VERSION,
        packages=find_packages(include=['trainsalt','bin/*','saltshaker/scripts/*']),
	#packages=['bin','saltshaker','saltshaker.tests','saltshaker.data','saltshaker.simulation',
        #          'saltshaker.training','saltshaker.util','saltshaker.initfiles',
        #          'saltshaker.validation','saltshaker.pipeline','saltshaker.config'],
	cmdclass={'test': SALTShakerTest},
        #entry_points={'console_scripts':['trainsalt = saltshaker.scripts.trainsalt']},
        scripts=['saltshaker/scripts/trainsalt','saltshaker/scripts/runpipe'],
#	scripts=['bin/trainsalt','bin/runpipe','saltshaker/validation/SynPhotPlot.py','saltshaker/validation/ValidateLightcurves.py','saltshaker/validation/ValidateModel.py','saltshaker/validation/ValidateSpectra.py','saltshaker/validation/figs/plotSALTModel.py'],
	package_data={'': ['initfiles/*.dat','initfiles/*.txt','data/kcor/*.fits','config/*conf','scripts/*']},
	include_package_data=True,
	author=AUTHOR,
	author_email=AUTHOR_EMAIL,
	license=LICENSE,
	long_description=open('README.md').read(),
        include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
	install_requires=['cython',
                          'numpy>=1.5.0',
                          'scipy>=0.9.0',
                          'extinction>=0.2.2',
                          'astropy>=0.4.0',
                          'pytest-astropy',
                          'sncosmo',
                          'astroquery',
                          'matplotlib',
                          'emcee',
                          'ipython',
                          'pandas',
                          'f90nml',
                          'iminuit',
                          'tqdm',
                          'pyyaml',
                          'pyParz'],
	)
