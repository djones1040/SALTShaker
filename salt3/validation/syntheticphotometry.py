import numpy as np
import pylab as plt
import sncosmo
import argparse
from salt3.util import snana
from astropy.table import Table
import astropy.units as u
from salt3.util.synphot import synphot
from scipy.interpolate import interp1d
from sncosmo.constants import HC_ERG_AA
from salt3.initfiles import init_rootdir
from salt3.training.init_hsiao import synphotB
from sncosmo.salt2utils import SALT2ColorLaw
_SCALE_FACTOR = 1e-12

def main(outfile,lcfile,salt3dir,
		 m0file='salt3_template_0.dat',
		 m1file='salt3_template_1.dat',
		 clfile='salt2_color_correction.dat',
		 cdfile='salt2_color_dispersion.dat',
		 errscalefile='salt2_lc_dispersion_scaling.dat',
		 lcrv00file='salt2_lc_relative_variance_0.dat',
		 lcrv11file='salt2_lc_relative_variance_1.dat',
		 lcrv01file='salt2_lc_relative_covariance_01.dat'):
	
	plt.clf()



	sdssFilters = ['sdss%s'%s for s in  'griz']

	zpsys='AB'
	
	salt2model = sncosmo.Model(source='salt2')
	salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,
								m1file=m1file,
								clfile=clfile,cdfile=cdfile,
								errscalefile=errscalefile,
								lcrv00file=lcrv00file,
								lcrv11file=lcrv11file,
								lcrv01file=lcrv01file)
	salt3model =  sncosmo.Model(salt3)
	
	salt2model.set(z=0)
	salt2model.set(x0=1)
	salt2model.set(t0=0)
	salt2model.set(c=0)
	
	salt3model.set(z=0)
	salt3model.set(x0=1)
	salt3model.set(t0=0)
	salt3model.set(c=0)

	plotmjd = np.linspace(-20, 55,100)
	
	fig = plt.figure(figsize=(15, 5))
	axes = [fig.add_subplot(130+i) for i in range(4)]
	
	for flt,ax in zip(sdssFilters,axes):
		for x1 in np.linspace(-2,2,100,True):
			salt2model.set(x1=-x1)
			salt2model.set(x1=-x1)
			salt2flux = salt2model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
			salt3flux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')#*\

			ax.plot(plotmjd,salt2flux,color='C1',
					label='SALT2')
			ax.plot(plotmjd,salt3flux,color='C2',
					label='SALT3')

			ax.set_title(flt)
			ax.set_xlim([-30,55])

	ax1.legend()
	plt.savefig(outfile)
	plt.show()
	plt.close(fig)
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Plot synthetic photometry for a range of x1 values')
	parser.add_argument('salt3dir',type=str,help='File with supernova fit parameters')
	parser.add_argument('outfile',type=str,nargs='?',default=None,help='File with supernova fit parameters')
	parser=parser.parse_args()
	args=vars(parser)
	if parser.outfile is None:
		args['outfile']='lccomp_%s.pdf'%sn.SNID
	main(**args)
