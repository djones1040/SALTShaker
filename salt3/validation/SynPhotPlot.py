#!/usr/bin/env python
import numpy as np
import pylab as plt
from matplotlib import colors
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

def main(outfile,salt3dir,filterset='SDSS',
		 m0file='salt3_template_0.dat',
		 m1file='salt3_template_1.dat',
		 clfile='salt3_color_correction.dat',
		 cdfile='salt3_color_dispersion.dat',
		 errscalefile='salt3_lc_dispersion_scaling.dat',
		 lcrv00file='salt3_lc_relative_variance_0.dat',
		 lcrv11file='salt3_lc_relative_variance_1.dat',
		 lcrv01file='salt3_lc_relative_covariance_01.dat'):
	
	plt.clf()



	filtdict = {'SDSS':['sdss%s'%s for s in  'ugri'],'Bessell':['bessell%s'%s +('x' if s=='u' else '')for s in  'ubvri']}
	filters=filtdict[filterset]
	
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
	salt3axes = [fig.add_subplot(2,len(filters),1+i) for i in range(len(filters))]
	salt2axes = [fig.add_subplot(2,len(filters),len(filters)+ 1+i,sharex=salt3ax) for i,salt3ax in enumerate(salt3axes)]
	xmin,xmax=-2,2
	norm=colors.Normalize(vmin=xmin,vmax=xmax)
	cmap=plt.get_cmap('RdBu')
	for flt,ax2,ax3 in zip(filters,salt2axes,salt3axes):
		
		for x1 in np.linspace(xmin,xmax,100,True):
			salt2model.set(x1=x1)
			salt3model.set(x1=x1)
			color=cmap(norm(x1))
			salt2flux = salt2model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
			salt3flux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')#*\
			
			ax2.set_yticks([])
			ax3.set_yticks([])
			ax2.plot(plotmjd,salt2flux,color=color,label='SALT2',linewidth=0.1)
			ax3.plot(plotmjd,salt3flux,color=color,label='SALT3',linewidth=0.1)
	
			ax3.set_title(flt,fontsize=20)
			#ax.set_xlim([-30,55])
	sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
	sm._A=[]
	fig.subplots_adjust(right=0.8,bottom=0.15,left=0.05)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	salt2axes[0].set_ylabel('SALT2 Flux',fontsize=20)
	salt3axes[0].set_ylabel('SALT3 Flux',fontsize=20)
	fig.colorbar(sm,cax=cbar_ax)
	cbar_ax.set_ylabel('Stretch  ($x_1$ parameter)',fontsize=20)
	
	fig.text(0.5,0.04,'Time since peak (days)',ha='center',fontsize=20)
	#axes[0].legend()
	plt.savefig(outfile)
	plt.close(fig)
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Plot synthetic photometry for a range of x1 values')
	parser.add_argument('salt3dir',type=str,help='File with supernova fit parameters')
	parser.add_argument('outfile',type=str,nargs='?',default=None,help='File with supernova fit parameters')
	parser.add_argument('--filterset',type=str,nargs='?',default='SDSS',help='File with supernova fit parameters')
	parser=parser.parse_args()
	args=vars(parser)
	if parser.outfile is None:
		args['outfile']='{}/synphot.pdf'.format(parser.salt3dir)
	main(**args)
