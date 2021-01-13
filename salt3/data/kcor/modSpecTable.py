#!/usr/bin/env python
# D. Jones - 5/22/20
import numpy as np

def main():
	with open('spectrograph_table_new.dat','w') as fout:
		print("""INSTRUMENT: MYSPECDEVICE
MAGREF_LIST: 17.0 32.0 #used to define SNR1 and SNR2
TEXPOSE_LIST: 30 1920 7680 30720 #seconds

#     LAM LAM LAM
		#     MIN MAX RES SNR1 SNR2""",file=fout)

		with open('spectrograph_table.dat') as fin:
			wavebins = np.arange(2975.27,3815.27,10)
			for w1,w2 in zip(wavebins[:-1],wavebins[1:]):
				snr1 = (w1*0.049 -50)*1
				snr2 = snr1/1000
				print(f"SPECBIN: {w1:.2f} {w2:.2f} 0.10 {snr1/10:.3f} {snr2/10:.3f} {snr1/10*8:.3f} {snr2/10*8:.3f} {snr1/10*16:.3f} {snr2/10*16:.3f} {snr1/10*32:.3f} {snr2/10*32:.3f}",file=fout)
			
			for line in fin:
				if line.startswith('SPECBIN'):
					lineparts = line.split()
					lineparts[-4] = f'{float(lineparts[-4])/10*1:.3f}'
					lineparts[-3] = f'{float(lineparts[-4])/1000:.3f}'
					lineparts[-2] = f'{float(lineparts[-4])*8:.3f}'
					lineparts[-1] = f'{float(lineparts[-3])*8:.3f}'
					lineparts += [f'{float(lineparts[-4])*16:.3f}',f'{float(lineparts[-3])*16:.3f}']
					lineparts += [f'{float(lineparts[-6])*32:.3f}',f'{float(lineparts[-5])*32:.3f}']
					print(' '.join(lineparts),file=fout)
					#if float(lineparts[1]) > 3804:
					#	import pdb; pdb.set_trace()

def lowz():
	with open('spectrograph_table_lowz.dat','w') as fout:
		print("""INSTRUMENT: MYSPECDEVICE
MAGREF_LIST: 7.0 22.0 #used to define SNR1 and SNR2
TEXPOSE_LIST: 30 1920 7680 30720 #seconds

#     LAM LAM LAM
		#     MIN MAX RES SNR1 SNR2""",file=fout)

		with open('spectrograph_table.dat') as fin:
			wavebins = np.arange(2975.27,3815.27,10)
			for w1,w2 in zip(wavebins[:-1],wavebins[1:]):
				snr1 = (w1*0.049 -50)*1
				snr2 = snr1/1000
				print(f"SPECBIN: {w1:.2f} {w2:.2f} 0.10 {snr1/10:.3f} {snr2/10:.3f} {snr1/10*8:.3f} {snr2/10*8:.3f} {snr1/10*16:.3f} {snr2/10*16:.3f} {snr1/10*32:.3f} {snr2/10*32:.3f}",file=fout)
			
			for line in fin:
				if line.startswith('SPECBIN'):
					lineparts = line.split()
					lineparts[-4] = f'{float(lineparts[-4])/10*1:.3f}'
					lineparts[-3] = f'{float(lineparts[-4])/1000:.3f}'
					lineparts[-2] = f'{float(lineparts[-4])*8:.3f}'
					lineparts[-1] = f'{float(lineparts[-3])*8:.3f}'
					lineparts += [f'{float(lineparts[-4])*16:.3f}',f'{float(lineparts[-3])*16:.3f}']
					lineparts += [f'{float(lineparts[-6])*32:.3f}',f'{float(lineparts[-5])*32:.3f}']
					print(' '.join(lineparts),file=fout)
					#if float(lineparts[1]) > 3804:
					#	import pdb; pdb.set_trace()

def hicken():
	with open('spectrograph_table_hicken.dat','w') as fout:
		print("""INSTRUMENT: MYSPECDEVICE
MAGREF_LIST: 7.0 22.0 #used to define SNR1 and SNR2
TEXPOSE_LIST: 30 1920 7680 30720 #seconds

#     LAM LAM LAM
		#     MIN MAX RES SNR1 SNR2""",file=fout)

		with open('spectrograph_table.dat') as fin:
			wavebins = np.arange(2975.27,3815.27,10)
			for w1,w2 in zip(wavebins[:-1],wavebins[1:]):
				snr1 = (w1*0.049 -50)*1
				snr2 = snr1/1000
				print(f"SPECBIN: {w1:.2f} {w2:.2f} 0.10 {snr1/10:.3f} {snr2/10:.3f} {snr1/10*8:.3f} {snr2/10*8:.3f} {snr1/10*16:.3f} {snr2/10*16:.3f} {snr1/10*32:.3f} {snr2/10*32:.3f}",file=fout)
			
			for line in fin:
				if line.startswith('SPECBIN'):
					lineparts = line.split()
					lineparts[-4] = f'{float(lineparts[-4])/10*1:.3f}'
					lineparts[-3] = f'{float(lineparts[-4])/1000:.3f}'
					lineparts[-2] = f'{float(lineparts[-4])*8:.3f}'
					lineparts[-1] = f'{float(lineparts[-3])*8:.3f}'
					lineparts += [f'{float(lineparts[-4])*16:.3f}',f'{float(lineparts[-3])*16:.3f}']
					lineparts += [f'{float(lineparts[-6])*32:.3f}',f'{float(lineparts[-5])*32:.3f}']
					print(' '.join(lineparts),file=fout)
					#if float(lineparts[1]) > 3804:
					#	import pdb; pdb.set_trace()

					
if __name__ == "__main__":
	#main()
	lowz()
	#hicken()
