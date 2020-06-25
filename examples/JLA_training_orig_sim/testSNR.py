#!/usr/bin/env python

from salt3.util import snana
import pylab as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import glob
from scipy.stats import binned_statistic

libid_map = {}

def main():

	simlibdir = '/Users/David/Dropbox/research/SALT3/examples/JLA_training_orig/JLA_training_subdirs/simlib'
	
	simlibs = ['Hamuy96.SIMLIB','OTHER_LOWZ.SIMLIB',
			   'Hicken2009.SIMLIB','Riess1999.SIMLIB',
			   'SDSS.SIMLIB','Jha2006.SIMLIB',
			   'SNLS3.SIMLIB']
	for s in simlibs:
		with open(f'{simlibdir}/{s}') as fin:
			for line in fin:
				if line.startswith('LIBID:'):
					snid = line.split('SNID=')[-1].split(' ')[0]
					libid = int(line.split()[1])
					libid_map[f"{s.split('.')[0]}_{libid}"] = snid
	#import pdb; pdb.set_trace()
	pdf_pages = PdfPages('specsim.pdf')
	fig = plt.figure()
	axcount = 0
	
	snfiles_data = glob.glob('/Users/David/Dropbox/research/SALT3/examples/JLA_training_orig/JLA_training_origlc/*dat')
	snfiles_sim = glob.glob('/Users/David/Dropbox/research/SALT3/examples/JLA_training_orig_sim/JLA_training_sim/*DAT')
	for i,s in enumerate(snfiles_sim):
		if 'SDSS' in s: continue
		
		sns = snana.SuperNova(s)
		if sns.NSPECTRA == 0: continue

		try: snid = libid_map[f"{sns.SURVEY.split('(')[0].replace('_LC','')}_{sns.SIM_LIBID}"]
		except: import pdb; pdb.set_trace()
		snd = snana.SuperNova(glob.glob(f'/Users/David/Dropbox/research/SALT3/examples/JLA_training_orig/JLA_training_origlc/*{snid}*dat')[0])
		zd = float(snd.REDSHIFT_HELIO.split()[0])
		zs = float(sns.REDSHIFT_HELIO.split()[0])
		
		for k in snd.SPECTRA.keys():
			if k not in sns.SPECTRA.keys(): continue
			if not axcount % 3 and axcount != 0:
				fig = plt.figure()
			
			ax = plt.subplot(3,1,axcount % 3 + 1)
			dataAvg = np.median(snd.SPECTRA[k]['FLAM'][(snd.SPECTRA[k]['LAMMIN'] > 6000) &
													   (snd.SPECTRA[k]['LAMMIN'] < 8000)])
			simAvg = np.median(sns.SPECTRA[k]['FLAM'][(sns.SPECTRA[k]['LAMMIN'] > 6000) &
													  (sns.SPECTRA[k]['LAMMIN'] < 8000)])
			specdataflux = binned_statistic(snd.SPECTRA[k]['LAMMIN'],snd.SPECTRA[k]['FLAM'],
											bins=(sns.SPECTRA[k]['LAMMIN']+sns.SPECTRA[k]['LAMMAX'])/2.,
											statistic='median').statistic
			ax.plot(sns.SPECTRA[k]['LAMMIN'][:-1]/(1+zd),specdataflux*simAvg/dataAvg,color='C0')
			ax.plot(sns.SPECTRA[k]['LAMMIN']/(1+zs),sns.SPECTRA[k]['FLAM'],color='C1')
			ax.set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
			ax.set_ylabel('flux')
			#if axcount == 5:
			#	import pdb; pdb.set_trace()
			
			axcount += 1
			if not axcount % 3:
				pdf_pages.savefig()
			#import pdb; pdb.set_trace()
		#if axcount >= 10:
		#	break

	pdf_pages.savefig()
	pdf_pages.close()
	
if __name__ == "__main__":
	main()
