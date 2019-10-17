from astropy.cosmology import Planck15 as cosmo
import numpy as np
import matplotlib.pyplot as plt
import os,scipy
import scipy.stats


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

def plot_hubble(fr,binned=True):
	if binned:
		stats,edges,bins = scipy.stats.binned_statistic(fr.zCMB,fr.MU,'mean')
		stat_err,edges2,bins2 = scipy.stats.binned_statistic(fr.zCMB,fr.MU,'std',bins=edges)
		bin_data=[]
		for i in range(1,len(edges)):
			inds=np.where(bins==i)
			bin_data.append(np.average(fr.MU[inds],weights=1./fr.MUERR[inds]))
		bin_data=np.array(bin_data)
		ax=plot('errorbar',[(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)],bin_data,yerr=stat_err,y_lab=r'$\mu$',fmt='o')
		ax,ax2=split_plot(ax,'errorbar',[(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)],
			y=bin_data-cosmo.distmod([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)]).value,yerr=stat_err,x_lab=r'$z_{\rm{CMB}}$',y_lab='Residual',fmt='o')
		lims=ax.get_xlim()
		ax2.plot(lims,[0,0],'k--',linewidth=3)
	else:
		ax=plot('errorbar',fr.zCMB,y=fr.MU,yerr=fr.MUERR,y_lab=r'$\mu$',fmt='o')
		ax,ax2=split_plot(ax,'errorbar',fr.zCMB,y=fr.MU-cosmo.distmod(fr.zCMB).value,yerr=fr.MUERR,x_lab=r'$z_{\rm{CMB}}$',y_lab='Residual',fmt='o')
		lims=ax.get_xlim()
		ax2.plot(lims,[0,0],'k--',linewidth=3)
	zinterp=np.arange(np.min(fr.zCMB),np.max(fr.zCMB),.01)
	ax.plot(zinterp,cosmo.distmod(zinterp).value,color='k',linewidth=3)
		
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