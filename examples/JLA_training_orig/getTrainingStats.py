#!/usr/bin/env python
# D. Jones - 5/20/20
"""For the training sample, get distribution of SNR, n_spec, 
and phases for every subsurvey.  Make simlib from every 
subsurvey as well for the photometry?"""

import glob
import numpy as np
import pylab as plt
import snana
import os
from saltshaker.util import snana

def specStats():

	surveylist = ["Hamuy1996_LC","Riess1999_LC","Jha2006_LC",
				  "Hicken2009_LC","OTHER_LOWZ_LC","SDSS","SNLS3_LC"]
	surveydict = {"Hamuy1996_LC":{"SNR":[],"NSPEC":[],"phase":[],"NWITHSPEC":0},
				  "Riess1999_LC":{"SNR":[],"NSPEC":[],"phase":[],"NWITHSPEC":0},
				  "Jha2006_LC":{"SNR":[],"NSPEC":[],"phase":[],"NWITHSPEC":0},
				  "Hicken2009_LC":{"SNR":[],"NSPEC":[],"phase":[],"NWITHSPEC":0},
				  "OTHER_LOWZ_LC":{"SNR":[],"NSPEC":[],"phase":[],"NWITHSPEC":0},
				  "SDSS":{"SNR":[],"NSPEC":[],"phase":[],"NWITHSPEC":0},
				  "SNLS3_LC":{"SNR":[],"NSPEC":[],"phase":[],"NWITHSPEC":0}}

	snls3count = 0
	SNLS_snidlist = np.loadtxt('JLA_training_subdirs/SNLS3_LC/SNLS3_LC.LIST',unpack=True,dtype=str)
	for snfile in glob.glob('JLA_training_origlc/*.dat'):
		sn = snana.SuperNova(snfile)
		#with open(snfile) as fin:
		#	for line in fin:
		#		if line.startswith('SURVEY'):
		#			survey = line.split()[1]
		#			break
		#os.system(f'cp {snfile} JLA_training_subdirs/{survey}/')
		#os.system(f'cp {snfile} JLA_training_subdirs/{sn.SURVEY}/')
		#continue


		if sn.SURVEY == 'SNLS3_LC' and snfile.split('/')[-1] not in SNLS_snidlist:
			continue
		elif sn.SURVEY == 'SNLS3_LC':
			snls3count += 1
			
		if len(list(sn.SPECTRA.keys())):
			surveydict[sn.SURVEY]["NWITHSPEC"] += 1
			surveydict[sn.SURVEY]["NSPEC"] += [len(list(sn.SPECTRA.keys()))]
		
		for k in sn.SPECTRA.keys():
			iSNR = np.where((sn.SPECTRA[k]['LAMMIN'] > 4000) &
							(sn.SPECTRA[k]['LAMMIN'] < 6000))[0]
			if np.median(sn.SPECTRA[k]['FLAM'][iSNR]/sn.SPECTRA[k]['FLAMERR'][iSNR]) is not np.nan and\
			   len(sn.SPECTRA[k]['FLAM'][iSNR]):
				#surveydict[sn.SURVEY]["SNR"] += [np.median(sn.SPECTRA[k]['FLAM'][iSNR]/sn.SPECTRA[k]['FLAMERR'][iSNR])]
				surveydict[sn.SURVEY]["SNR"] += [np.sum(sn.SPECTRA[k]['FLAM'][iSNR])/np.sqrt(np.sum(sn.SPECTRA[k]['FLAMERR'][iSNR]**2.))]
			if (sn.SPECTRA[k]['SPECTRUM_MJD']-sn.SEARCH_PEAKMJD)/(1+float(sn.REDSHIFT_HELIO.split('+-')[0])) > -20 and \
			   (sn.SPECTRA[k]['SPECTRUM_MJD']-sn.SEARCH_PEAKMJD)/(1+float(sn.REDSHIFT_HELIO.split('+-')[0])) < 50:
				surveydict[sn.SURVEY]["phase"] += [sn.SPECTRA[k]['SPECTRUM_MJD']-sn.SEARCH_PEAKMJD]
	#import pdb; pdb.set_trace()
	print(snls3count)
	for s in surveydict.keys():
		print("")
		print(f"survey {s}")
		print("median SNR, phase, NOBS")
		print(np.median(surveydict[s]['SNR']),np.median(surveydict[s]['phase']),np.median(surveydict[s]["NSPEC"]))
		print("std. dev. SNR, phase, NOBS")
		print(np.std(surveydict[s]['SNR']),np.std(surveydict[s]['phase']),np.std(surveydict[s]["NSPEC"]))
		print("min, max phase")
		if len(surveydict[s]['phase']): print(np.min(surveydict[s]['phase']),np.max(surveydict[s]['phase']))
		print(f'number with spec {surveydict[s]["NWITHSPEC"]}')
		
if __name__ == "__main__":
	specStats()
