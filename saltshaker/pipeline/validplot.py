import matplotlib as mpl
mpl.use('agg')
import pylab as plt
import numpy as np
from saltshaker.util.txtobj import txtobj
from saltshaker.util import getmu
from functools import partial
from scipy.stats import binned_statistic
from astropy.cosmology import Planck15 as planck
from astropy.cosmology import FlatLambdaCDM
import astropy.table as at

__validfunctions__=dict()
def validfunction(validfunction):
	"""Decorator to register a given function as a plotting function"""
	__validfunctions__[validfunction.__name__]=validfunction
	return validfunction

class ValidPlots:
	def __init__(self):

		self.validfunctions = {}
		for key in __validfunctions__:
			if __validfunctions__[key].__qualname__.split('.')[0] == \
			   type(self).__name__:
				self.validfunctions[key] = partial(__validfunctions__[key],self)

	def input(self,inputfile=None):
		self.inputfile = inputfile

	def output(self,outputdir=None,prefix=''):
		if not outputdir.endswith('/'):
			self.outputdir = '%s/'%outputdir
		else: self.outputdir = outputdir
		self.prefix=prefix

	def run(self,*args):
		validfunctions = self.validfunctions
		for k in validfunctions.keys():
			validfunctions[k](*args)

class lcfitting_validplots(ValidPlots):
	
	@validfunction		
	def simvfit(self):
		plt.rcParams['figure.figsize'] = (12,4)
		plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=0, hspace=0)
		fr = txtobj(self.inputfile,fitresheader=True)
		if 'SIM_mB' not in fr.__dict__.keys():
			print('not a simulation!  skipping simvfit')
			return
		elif 'SIM_x1' not in fr.__dict__.keys() or 'SIM_c' not in fr.__dict__.keys():
			print('not a salt2-like simulation!  skipping simvfit')
			return
            
		ax1,ax2,ax3 = plt.subplot(131),plt.subplot(132),plt.subplot(133)

		cbins = np.linspace(-0.3,0.3,20)
		x1bins = np.linspace(-1,1,20)
		mubins = np.linspace(-1,1,20)
		mu = fr.mB + 0.14*fr.x1 - 3.1*fr.c + 19.36
		SIM_mu = fr.SIM_mB + 0.14*fr.SIM_x1 - 3.1*fr.SIM_c + 19.36
		
		ax1.hist(fr.c-fr.SIM_c,bins=cbins)
		ax1.set_xlabel('$c - c_{\mathrm{sim}}$',fontsize=15)
		ax1.set_ylabel('N$_{\mathrm{SNe}}$',fontsize=15)
		ax2.hist(fr.x1-fr.SIM_x1,bins=x1bins)
		ax2.set_xlabel('$x_1 - x_{1,\mathrm{sim}}$',fontsize=15)
		ax3.hist(mu-SIM_mu,bins=mubins)
		ax3.set_xlabel('$\mu - \mu_{\mathrm{sim}}$',fontsize=15)

		ax2.set_ylabel([])
		ax3.yaxis.tick_right()
		
		plt.savefig('%s%s_simvfit.png'%(self.outputdir,self.prefix))

		return

	@validfunction
	def hubbleplot(self):

		plt.rcParams['figure.figsize'] = (12,4)
		plt.subplots_adjust(
			left=None, bottom=0.2, right=None, top=None, wspace=0, hspace=0)
		fr = txtobj(self.inputfile,fitresheader=True)
		ax = plt.axes()
		if not hasattr(fr,'HOST_LOGMASS'):
			fr.__dict__['HOST_LOGMASS'] = None
		fr = getmu.getmu(fr)

		def errfnc(x):
			return(np.std(x)/np.sqrt(len(x)))
		
		zbins = np.logspace(np.log10(0.01),np.log10(1.0),25)
		mubins = binned_statistic(
			fr.zCMB,fr.mures,bins=zbins,statistic='mean').statistic
		mubinerr = binned_statistic(
			fr.zCMB,fr.mu,bins=zbins,statistic=errfnc).statistic
		ax.errorbar(fr.zCMB,fr.mures,yerr=fr.muerr,alpha=0.2,fmt='o')
		ax.errorbar(
			(zbins[1:]+zbins[:-1])/2.,mubins,yerr=mubinerr,fmt='o-')

		ax.axhline(0,color='k',lw=2)
		ax.set_xscale('log')
		ax.xaxis.set_major_formatter(plt.NullFormatter())
		ax.xaxis.set_minor_formatter(plt.NullFormatter())
		ax.set_ylabel('SNe',fontsize=11,labelpad=0)
		ax.set_xlim([0.01,1.0])
		ax.xaxis.set_ticks([0.01,0.02,0.05,0.1,0.2,0.3,0.5,1.0])
		ax.xaxis.set_ticklabels(['0.01','0.02','0.05','0.1','0.2','0.3','0.5','1.0'])

		ax.set_xlabel('$z_{CMB}$',fontsize=15)
		ax.set_ylabel('$\mu - \mu_{L CDM}$',fontsize=15)
		
		plt.savefig('%s%s_hubble.png'%(self.outputdir,self.prefix))

		return

class getmu_validplots(ValidPlots):
	
	@validfunction		
	def hubble(self):
		
		plt.clf()
		plt.rcParams['figure.figsize'] = (12,4)
		plt.subplots_adjust(
			left=None, bottom=0.2, right=None, top=None, wspace=0, hspace=0)
		fr = txtobj(self.inputfile,fitresheader=True)
		ax = plt.axes()
		cosmo = FlatLambdaCDM(H0=70, Om0=0.315, Tcmb0=planck.Tcmb0)
		fr.MURES = fr.MU - cosmo.distmod(fr.zCMB).value

		def errfnc(x):
			return(np.std(x)/np.sqrt(len(x)))
		
		zbins = np.logspace(np.log10(0.01),np.log10(1.0),25)
		mubins = binned_statistic(
			fr.zCMB,fr.MURES,bins=zbins,statistic='mean').statistic
		mubinerr = binned_statistic(
			fr.zCMB,fr.MU,bins=zbins,statistic=errfnc).statistic
		ax.errorbar(fr.zCMB,fr.MURES,yerr=fr.MUERR,alpha=0.2,fmt='o')
		ax.errorbar(
			(zbins[1:]+zbins[:-1])/2.,mubins,yerr=mubinerr,fmt='o-')

		ax.axhline(0,color='k',lw=2)
		ax.set_xscale('log')
		ax.xaxis.set_major_formatter(plt.NullFormatter())
		ax.xaxis.set_minor_formatter(plt.NullFormatter())
		ax.set_ylabel('SNe',fontsize=11,labelpad=0)
		ax.set_xlim([0.01,1.0])
		ax.xaxis.set_ticks([0.01,0.02,0.05,0.1,0.2,0.3,0.5,1.0])
		ax.xaxis.set_ticklabels(['0.01','0.02','0.05','0.1','0.2','0.3','0.5','1.0'])

		ax.set_xlabel('$z_{CMB}$',fontsize=15)
		ax.set_ylabel('$\mu - \mu_{L CDM}$',fontsize=15)
		
		plt.savefig('%s%s_BBC_hubble_prelim.png'%(self.outputdir,self.prefix))


	@validfunction
	def nuisancebias(self):
		sigint,alpha,beta = [],[],[]
		alphaerr,betaerr = [],[]
		with open(self.inputfile) as fin:
			for line in fin:
				if line.startswith('#') and 'sigint' in line and '=' in line:
					sigint.append(float(line.split()[3]))
				elif line.startswith('#') and 'alpha0' in line and '=' in line:
					alpha.append(float(line.split()[3]))
					alphaerr.append(float(line.split()[5]))
				elif line.startswith('#') and 'beta0' in line and '=' in line:
					beta.append(float(line.split()[3]))
					betaerr.append(float(line.split()[5]))

		if np.any(np.isnan(sigint+alpha+beta)):
			return
        
		fr = txtobj(self.inputfile,fitresheader=True)
		if 'SIM_alpha' not in fr.__dict__.keys() and 'SIM_beta' not in fr.__dict__.keys():
			sim_alpha = -99.
			sim_beta = -99.
		else:
			sim_alpha = fr.SIM_alpha[0]
			sim_beta = fr.SIM_beta[0]
		plt.clf()
		ax = plt.axes()
		ax.set_ylabel('Nuisance Parameters',fontsize=15)
		ax.xaxis.set_ticks([1,2,3])
		ax.xaxis.set_ticklabels(['alpha','beta',r'$\sigma_{\mathrm{int}}$'],rotation=30)

		ax.errorbar(1,np.mean(alpha)-sim_alpha,yerr=np.sqrt(np.std(alpha)**2/len(alpha)),fmt='o',color='C0',label='fit')
		ax.errorbar(2,np.mean(beta)-sim_beta,yerr=np.sqrt(np.std(beta)**2/len(beta)),fmt='o',color='C0')
		ax.errorbar(3,np.mean(sigint)-0.1,fmt='o',color='C0')
		ax.axhline(0,color='k',lw=2)
		
		ax.text(0.17,0.9,r"""$\alpha_{sim} = %.3f$
$\alpha_{fit} = %.3f \pm %.3f$"""%(
	sim_alpha,np.mean(alpha),np.sqrt(np.std(alpha)**2/len(alpha))),transform=ax.transAxes,ha='center',va='center')
		ax.text(0.5,0.9,r"""$\beta_{sim} = %.3f$
$\beta_{fit} = %.3f \pm %.3f$"""%(
	sim_beta,np.mean(beta),np.sqrt(np.std(beta)**2/len(beta))),transform=ax.transAxes,ha='center',va='center')
		ax.text(0.83,0.9,"""$\sigma_{int} = %.3f$"""%(
			np.mean(sigint)),transform=ax.transAxes,ha='center',va='center')

		ax.set_xlim([0.5,3.5])
		plt.savefig('%s%s_nuisancebias.png'%(self.outputdir,self.prefix))

class cosmofit_validplots(ValidPlots):
	
	@validfunction		
	def cosmopar(self):
		# plot w, Om numbers and biases
		# could make a tex table also
		Om_Planck = 0.315
		w_Planck = -1
		
		data = at.Table.read(self.inputfile,format='ascii')
		print(data)
		
		plt.clf()
		ax = plt.axes()
		ax.set_ylabel('Cosmo Parameters',fontsize=15)
		ax.xaxis.set_ticks([1,2])
		ax.xaxis.set_ticklabels(['$w$','$\Omega_M$'],rotation=30)

		ax.errorbar(1,np.mean(data['w'])-w_Planck,yerr=np.sqrt(np.std(data['w'])**2/len(data['w'])),fmt='o',color='C0',label='fit')
		ax.errorbar(2,np.mean(data['OM'])-Om_Planck,yerr=np.sqrt(np.std(data['OM'])**2/len(data['OM'])),fmt='o',color='C0')
		ax.axhline(0,color='k',lw=2)
		
		ax.text(0.17,0.9,r"""$w = %.3f \pm %.3f$"""%(
			np.mean(data['w']),np.sqrt(np.std(data['w'])**2/len(data['w']))),transform=ax.transAxes,ha='center',va='center')
		ax.text(0.5,0.9,r"""$\Omega_M = %.3f \pm %.3f$"""%(
			np.mean(data['OM']),np.sqrt(np.std(data['OM'])**2/len(data['OM']))),transform=ax.transAxes,ha='center',va='center')

		ax.set_xlim([0.5,2.5])
		plt.savefig('%s%s_cosmopar.png'%(self.outputdir,self.prefix))
