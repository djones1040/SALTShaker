#!/usr/bin/env python

import numpy as np
import pylab as plt
from salt3.util import snana
import os

def checkSALT(parameters,parlist,lcfiles,snlist,outdir,idx=0):

	plt.close('all')
	plt.rcParams['figure.figsize'] = (12,4)
	ax1,ax2,ax3 = plt.subplot(131),plt.subplot(132),plt.subplot(133)

	x1_salt3,x0_salt3,c_salt3,z,x1_salt2,x0_salt2,c_salt2 = \
		np.array([]),np.array([]),np.array([]),np.array([]),\
		np.array([]),np.array([]),np.array([])
	for l in lcfiles:
		sn = snana.SuperNova('%s/%s'%(os.path.dirname(snlist),l))
		if 'x0_%s'%sn.SNID not in parlist: continue
		if 'SIM_SALT2x0' not in sn.__dict__.keys():
			return(0)
		x0_salt2 = np.append(x0_salt2,float(sn.SIM_SALT2x0))
		x1_salt2 = np.append(x1_salt2,float(sn.SIM_SALT2x1))
		c_salt2 = np.append(c_salt2,float(sn.SIM_SALT2c))

		x0_salt3 = np.append(x0_salt3,parameters[parlist == 'x0_%s'%sn.SNID])
		x1_salt3 = np.append(x1_salt3,parameters[parlist == 'x1_%s'%sn.SNID])
		c_salt3 = np.append(c_salt3,parameters[parlist == 'c_%s'%sn.SNID])

		z = np.append(z,sn.REDSHIFT_FINAL.split('+-')[0])
		#if len(x0_salt2) != len(x0_salt3): import pdb; pdb.set_trace()
		
	mb_salt2 = -2.5*np.log10(x0_salt2) + 10.635
	mb_salt3 = -2.5*np.log10(x0_salt3) + 10.635
	mbbins = np.linspace(-2,2,30)
	x1bins = np.linspace(-3,3,30)
	cbins = np.linspace(-0.3,0.3,30)

	ax1.hist(mb_salt3-mb_salt2,bins=mbbins)
	ax2.hist(x1_salt3-x1_salt2,bins=x1bins)
	ax3.hist(c_salt3-c_salt2,bins=cbins)

	ax1.set_xlabel('$\Delta m_B$')
	ax2.set_xlabel('$\Delta x_1$')
	ax3.set_xlabel('$\Delta c$')
	
	plt.savefig('%s/saltparcomp_%i.png'%(outdir,idx))
	print(np.mean(x1_salt3),np.mean(x1_salt2))
	print(np.mean(c_salt3),np.mean(c_salt2))	
	return
