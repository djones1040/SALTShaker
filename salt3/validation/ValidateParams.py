#!/usr/bin/env python
# D. Jones - 5/2/20

import astropy.table as at
import numpy as np
import pylab as plt
plt.rcParams['figure.figsize'] = (8,8)
plt.subplots_adjust(left=0.1, bottom=None, right=0.95, top=0.95, wspace=0.15, hspace=0.15)

def main(infile,outfile):

	data = at.Table.read(infile,format='ascii')
	ax1,ax2,ax3,ax4 = \
		plt.subplot(221),plt.subplot(222),\
		plt.subplot(223),plt.subplot(224)

	salt3mB = -2.5*np.log10(data['x0'][data['SALT2_x0'] > 0])+10.635
	salt2mB = -2.5*np.log10(data['SALT2_x0'][data['SALT2_x0'] > 0])+10.635

	mbbins = np.linspace(-2,2,20)
	x1bins = np.linspace(-1,1,20)
	cbins = np.linspace(-0.2,0.2,20)
	t0bins = np.linspace(-5,5,20)

	ax1.hist(salt3mB-salt2mB,bins=mbbins)
	ax2.hist(data['x1']-data['SALT2_x1'],bins=x1bins)
	ax3.hist(data['c']-data['SALT2_c'],bins=cbins)
	ax4.hist(data['t0']-data['SALT2_t0'],bins=t0bins)

	ax1.set_xlabel('SALT3 $m_B$ - SALT2 $m_B$')
	ax2.set_xlabel('SALT3 $x_1$ - SALT2 $x_1$')
	ax3.set_xlabel('SALT3 $c$ - SALT2 $c$')
	ax4.set_xlabel('SALT3 $t_0$ - SALT2 $t_0$')

	for ax in [ax1,ax2,ax3,ax4]:
		ax.set_ylabel('N$_{SNe}$')

	ax1.text(0.5,0.85,f"SALT3 $m_B$ = {np.median(salt3mB):.2f}\nSALT2 $m_B$ = {np.median(salt2mB):.2f}",
			 ha='center',va='center',transform=ax1.transAxes,bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.5})
	ax2.text(0.5,0.85,f"SALT3 $x_1$ = {np.median(data['x1']):.2f}\nSALT2 $x_1$ = {np.median(data['SALT2_x1']):.2f}",
			 ha='center',va='center',transform=ax2.transAxes,bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.5})
	ax3.text(0.5,0.85,f"SALT3 c = {np.median(data['c']):.3f}\nSALT2 c = {np.median(data['SALT2_c']):.3f}",
   			 ha='center',va='center',transform=ax3.transAxes,bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.5})
	ax4.text(0.5,0.85,f"SALT3 t$_0$ = {np.median(data['t0']):.3f}\nSALT2 t$_0$ = {np.median(data['SALT2_t0']):.3f}",
			 ha='center',va='center',transform=ax4.transAxes,bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.5})
		
	plt.savefig(outfile)
	print('finished plotting SALT parameter comparisons')
	
if __name__ == "__main__":
	main()
