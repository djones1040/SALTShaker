#!/usr/bin/env python
# D. Jones - 5/31/20

from saltshaker.util import snana
import glob
import numpy as np
import astropy.table as at

snparfile = "/Users/David/Dropbox/research/SALT3/examples/JLA_training_orig/snpca_training_v6_resids.txt"
snpar = at.Table.read(snparfile,format='ascii')

def main():

	for lcdir in ['SDSS','Hamuy1996_LC','Hicken2009_LC','Jha2006_LC',
				  'OTHER_LOWZ_LC','Riess1999_LC',
				  'SNLS3_LC']:
		listfile = glob.glob(f"{lcdir}/*LIST")[0]
		snfiles = np.loadtxt(listfile,unpack=True,dtype=str)
		with open(f"{lcdir}.FITRES",'w') as fout:
			print("""VARNAMES:  CID x1 c mB""",file=fout)
			
			for s in snfiles:
				sn = snana.SuperNova(f"{lcdir}/{s}")
				iPar = snpar['SNID'] == str(sn.SNID)
				try: print(f"SN: {sn.SNID} {snpar['x1'][iPar][0]} {snpar['c'][iPar][0]} {-2.5*np.log10(snpar['x0'][iPar][0]) + 10.635}",file=fout)
				except:
					print(sn.SNID)
					import pdb; pdb.set_trace()
		
if __name__ == "__main__":
	main()
