#!/usr/bin/env python

import numpy as np
from txtobj import txtobj

def main():

	fr = txtobj('SNLS3_lowz.FITRES.TEXT',fitresheader=True)
	snid,pkmjd,pkmjderr = np.loadtxt('Pantheon_Found_PeakMJD.LIST',unpack=True,dtype=str)
	pkmjd = pkmjd.astype(float)
	pkmjderr = pkmjderr.astype(float)

	fout = open('JLA_orig_PeakMJD.LIST','w')
	fr.exists = np.zeros(len(fr.CID))
	for s,p,pe in zip(snid,pkmjd,pkmjderr):
		if s.lower() in fr.CID or s in fr.CID:
			print('%s %.2f %.2f'%(s,p,pe),file=fout)
			fr.exists[fr.CID == s] = True
		else:
			print('%s %.2f %.2f'%(s,p,pe),file=fout)
			fr.exists[fr.CID == s] = False

	for s,p,pe,e in zip(fr.CID,fr.PKMJD,fr.PKMJDERR,fr.exists):
		if not e:
			print('%s %.2f %.2f'%(s,p,pe),file=fout)

	fout.close()

def fromsnpca():
	from snana import SuperNova
	import subprocess
	from subprocess import Popen, PIPE
	
	snorig = np.loadtxt('/Users/David/Dropbox/research/salt2/trainsalt/trainingsample_snls_sdss_v6.list',unpack=True,usecols=[0],dtype=str)
	snlcs = np.loadtxt('JLA_training_origlc/JLA_training_origlc.LIST',unpack=True,dtype=str)
	
	snidlist,newsnidlist = [],[]
	tmaxlist,tmaxerrlist = [],[]
	for s in snlcs:
		sn = SuperNova('JLA_training_origlc/%s'%s)
		if isinstance(sn.SNID,float) or isinstance(sn.SNID,int): snid = 'SDSS%i'%sn.SNID
		else: snid = sn.SNID[:]
		if snid not in snorig and 'sn'+snid not in snorig and snid.lower() not in snorig and 'sn'+snid.lower() not in snorig:
			print(snid)
			if snid == '2000E': import pdb; pdb.set_trace()
		else:
			idx = np.where((snid == snorig) | ('sn'+snid == snorig) | (snid.lower() == snorig) | ('sn'+snid.lower() == snorig))[0]
			snidlist += [snorig[idx][0]]
			newsnidlist += [sn.SNID]
			if 'SDSS' not in snorig[idx][0]:
				grep_out = subprocess.check_output(
					"grep @DayMax /Users/David/Dropbox/research/salt2/trainsalt/v6/lc*%s*list"%snorig[idx][0],shell=True).decode('utf-8')
			else:
				grep_out = subprocess.check_output(
					"grep @DayMax /Users/David/Dropbox/research/salt2/trainsalt/v6/SDSS3*%06i*"%int(snorig[idx][0][4:]),shell=True).decode('utf-8')
				
			tmaxlist += [float(grep_out.split()[1])]
			tmaxerrlist += [float(grep_out.split()[2].replace('\n',''))]

	fout = open('JLA_orig_PEAKMJD.LIST','w')
	for s,t,te in zip(newsnidlist,tmaxlist,tmaxerrlist):
		print('%s %.3f %.3f'%(s,t,te),file=fout)
	fout.close()
		
if __name__ == "__main__":
	#main()
	fromsnpca()
