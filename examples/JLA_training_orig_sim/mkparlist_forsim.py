import glob
import numpy as np
import snana

def main():

	snfiles = np.loadtxt('JLA_training_sim/JLA_TRAINING_SIM.LIST',unpack=True,dtype=str)
	
	with open('saltparlist_sim.txt','w') as fout, open('saltparlist_pkmjd.txt','w') as fout2:
	
		print('# SNID zHelio x0 x1 c',file=fout)
		for snfile in snfiles:
			sn = snana.SuperNova(f"JLA_training_sim/{snfile}")
			print(f"{sn.SNID} {sn.REDSHIFT_HELIO.split('+-')[0]} {sn.SIM_SALT2x0*10**(-0.4*0.27)} {sn.SIM_SALT2x1} {sn.SIM_SALT2c}",file=fout)
			print(f"{sn.SNID} {sn.SIM_PEAKMJD.split()[0]}",file=fout2)
			
if __name__ == "__main__":
	main()
