from astropy.cosmology import Planck15 as cosmo
import numpy as np
import matplotlib.pyplot as plt
import os


from .txtobj import txtobj
from .getmu import *
from .util import *

def getObj(fitresfile, fitresheader = True, makeCuts = True):
	fr = txtobj(fitresfile,fitresheader = fitresheader)
	if makeCuts:
		fr = mkcuts(fr)
	fr.filename=os.path.splitext(os.path.basename(fitresfile))[0]
	return(fr)

def calcMu(fr,alpha=0.14,beta=3.1,M=-19.36):
	fr.MU = fr.mB + alpha*fr.x1 - beta*fr.c - M
	fr.MUERR = np.sqrt(fr.mBERR**2 + alpha**2.*fr.x1ERR**2. + beta**2.*fr.cERR**2)
	return(fr)

def plot_hubble(fr):
	ax=plot('errorbar',fr.zCMB,y=fr.MU,yerr=fr.MUERR,y_lab=r'$\mu$',fmt='o')
	zinterp=np.arange(np.min(fr.zCMB),np.max(fr.zCMB),.01)
	ax.plot(zinterp,cosmo.distmod(zinterp).value,color='k',linewidth=3)
	ax=split_plot(ax,'errorbar',fr.zCMB,y=fr.MU-cosmo.distmod(fr.zCMB).value,yerr=fr.MUERR,x_lab=r'$z_{\rm{CMB}}$',y_lab='Residual',fmt='o')
	if not os.path.exists('figures'):
		os.makedirs('figures')
	if os.path.exists(os.path.join('figures',fr.filename+'_hubble_diagram.pdf')):
		ext=1
		while os.path.exists(os.path.join('figures',fr.filename+'_hubble_diagram_'+str(ext)+'.pdf')):
			ext+=1
		outname=os.path.join('figures',fr.filename+'_hubble_diagram_'+str(ext)+'.pdf')
	else:
		outname=os.path.join('figures',fr.filename+'_hubble_diagram.pdf')
	plt.tight_layout()
	plt.savefig(outname,format='pdf')

	plt.clf()

if __name__=='__main__':
	getObj('test')