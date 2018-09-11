from distutils.core import setup
import sys
from setuptools.command.test import test as TestCommand

if sys.version_info < (3,0):
	sys.exit('Sorry, Python 2 is not supported')

class SALT3Test(TestCommand):

	def run_tests(self):
		import salt3
		errno = salt3.test()
		sys.exit(errno)
	
AUTHOR = 'David Jones, Rick Kessler'
AUTHOR_EMAIL = 'david.jones@ucsc.edu'
VERSION = '0.1dev'
LICENSE = 'BSD'
URL = 'salt3.readthedocs.org'

setup(
	name='SALT3',
	version=VERSION,
	packages=['salt3','salt3.tests'],
	cmdclass={'test': SALT3Test},
	author=AUTHOR,
	author_email=AUTHOR_EMAIL,
	license=LICENSE,
	long_description=open('README.md').read(),
	install_requires=['numpy>=1.5.0',
					  'scipy>=0.9.0',
					  'extinction>=0.2.2',
					  'astropy>=0.4.0',
					  'pytest-astropy'],
)
