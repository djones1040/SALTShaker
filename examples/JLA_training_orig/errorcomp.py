#!/usr/bin/env python
# D. Jones - 5/7/20

import numpy as np
import pylab as plt
plt.ion()
from scipy.interpolate import RegularGridInterpolator

def main():
	phase,wave,err = np.loadtxt('output_fullerr/salt3_lc_relative_variance_0.dat',unpack=True)
	phase2,wave2,err2 = np.loadtxt('../../salt3/initfiles/salt2_lc_relative_variance_0.dat',unpack=True)
	phasem0,wavem0,fluxm0 = np.loadtxt('../../salt3/initfiles/salt2_template_0.dat',unpack=True)

	errinterp = RegularGridInterpolator((np.unique(phase2),np.unique(wave2)),err2.reshape([len(np.unique(phase2)),len(np.unique(wave2))]),'nearest',False,0)
	
	ax = plt.axes()
	ax.plot(wavem0[phasem0 == 0],fluxm0[phasem0 == 0]*errinterp(([0],wavem0[phasem0 == 0])),label='SALT2')
	ax.plot(wave2[phase2 == 0.5],err2[phase2 == 0.5],label='SALT2 relative')
	ax.plot(wave[phase == 0],err[phase == 0],label='SALT3')

	ax.set_xlabel('Wavelength',fontsize=15)
	ax.set_ylabel('$M_0$ Error',fontsize=15)
	ax.legend()
	
	import pdb; pdb.set_trace()
	
if __name__ == "__main__":
	main()
