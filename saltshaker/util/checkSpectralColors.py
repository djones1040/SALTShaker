#!/usr/bin/env python
# D. Jones - 10/17/19
"""compare spectra to colors from corresponding photometry"""
import glob
import snana
import os
import numpy as np

class spec_colors:
	def __init__(self):
		pass

	def spec_colors(self,specfile,photfile,kcor):
		
		pass

	def all_spec_colors(self,specdir,photdir,conf):
	
		pass

def check_for_found_bk():

	lc_snid,spec_snid = np.loadtxt('foundnames.txt',unpack=True,dtype=str)
	files = glob.glob('/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/FoundPubSpec/*Foundation.dat')
	for f in files:
		snid = f.split('/')[-1].split('_')[0]
		if snid not in spec_snid:
			print(snid)

def check_for_found():

	lcdir = '/Users/David/Dropbox/research/SALT3_runpipe/SALT3_NIR/data/snana/found*txt'
	files = glob.glob(lcdir)
	count = 0
	namesfile = 'foundnames.txt'
	fout = open(namesfile,'w')
	print('# lc_snid spec_snid',file=fout)
	for f in files:
		sn = snana.SuperNova(f)
		if sn.SNID.startswith('PSNJ'):
			specfile = glob.glob('/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/FoundPubSpec/%s*Foundation.dat'%sn.SNID.lower().replace('-','')[:8])
		elif sn.SNID.startswith('MASTER'):
			specfile = glob.glob('/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/FoundPubSpec/%s*Foundation.dat'%sn.SNID.lower().replace('-','')[:9])
		else:
			specfile = glob.glob('/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/FoundPubSpec/%s*Foundation.dat'%sn.SNID.lower().replace('-',''))
		if not len(specfile):
			specfile = glob.glob('/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/FoundPubSpec/%s*Foundation.dat'%sn.SNID.replace('-',''))
			if not len(specfile):
				specfile = glob.glob('/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/FoundPubSpec/%s*Foundation.dat'%sn.SNID.replace('SN','').replace('AT','').lower().replace('-',''))
				if not len(specfile):
					specfile = glob.glob('/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/FoundPubSpec/%s*Foundation.dat'%sn.SNID.replace('SN','').replace('AT','').lower().replace('-',''))
					if not len(specfile):
						specfile = glob.glob('/Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/FoundPubSpec/%s*Foundation.dat'%sn.SNID.lower().replace('20','').lower().replace('-',''))
						if not len(specfile):
							#if '17dzg' in sn.SNID:
							#import pdb; pdb.set_trace()
							print(sn.SNID)
							count += 1
		if len(specfile): print('%s %s'%(sn.SNID,specfile[0].split('/')[-1].split('_')[0]),file=fout)
	print('%i files missing'%count)
	fout.close()
if __name__ == "__main__":
	check_for_found_bk()
	#check_for_found()
