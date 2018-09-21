#!/usr/bin/env python
import numpy as np
import pylab as plt

def plot_io_sed(inputfile='../initfiles/Hsiao07.dat',
				outputfile='../output/salt2_template_0.dat'):

	iphase,iwave,iflux = np.loadtxt(inputfile,unpack=True)
	owave,ophase,oflux = np.loadtxt(outputfile,unpack=True)

	for p in np.unique(iphase):
		if p < -14: continue
		plt.close()

		plt.plot(iwave[iphase == p],iflux[iphase == p],label='Hsiao model')
		plt.plot(owave[(ophase-p)**2. == np.min((ophase-p)**2.)],
				 oflux[(ophase-p)**2. == np.min((ophase-p)**2.)],label='output model')
		plt.xlabel('Wavelength ($\AA$)')
		plt.ylabel('Flux')
		plt.title('phase = %.1f days'%p)
		plt.xlim([2000,9200])
		import pdb; pdb.set_trace()
		
	return
