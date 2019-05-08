import os
import glob
from scipy.interpolate import bisplev, bisplrep
from scipy.interpolate import interp1d
import numpy as np
from astropy.io import ascii

init_rootdir = os.path.dirname(os.path.abspath(__file__))
init_filelist = ['Hsiao07.dat','flatnu.dat','vegased_2004_stis.txt']

def mkKaeporaSpec():

	x10files = glob.glob('David_Comps/David_x1=0_*txt')
	x11files = glob.glob('David_Comps/David_x1=1_*txt')

	phasex10,wavex10,fluxx10 = np.array([]),np.array([]),np.array([])
	phasex11,wavex11,fluxx11 = np.array([]),np.array([]),np.array([])
	outwave = np.arange(2000,10001,1)
	outphase = np.arange(-20,60,1)

	filephase = np.array([])
	for x10 in x10files:
#		import pdb; pdb.set_trace()
		filephase = np.append(filephase,float(x10.split('phase=')[-1].replace('m','-').replace('p','').split('_')[0]))
	
	for x10 in np.array(x10files)[np.argsort(filephase)]:
		
		data = ascii.read(x10)
		#if np.min(data['Wavelength']) > outwave[0] or np.max(data['Wavelength']) < outwave[-1]:
		#	continue
		data['Wavelength'] = data['Wavelength']/(1 + data['Redshift'][0])
		data['Flux'] *= (1 + data['Redshift'][0])
		data['Phase'] /= (1 + data['Redshift'][0])
		int1d = interp1d(data['Wavelength'],data['Flux'],
						 fill_value=(0.0,0.0),bounds_error=False)
		newflux = int1d(outwave)
		phasex10 = np.append(phasex10,[data['Phase'][0]]*len(outwave))
		wavex10 = np.append(wavex10,outwave)
		fluxx10 = np.append(fluxx10,newflux)
		#import pdb; pdb.set_trace()
		
	for x11 in x11files:
		
		data = ascii.read(x11)
		data['Wavelength'] = data['Wavelength']/(1 + data['Redshift'][0])
		data['Flux'] *= (1 + data['Redshift'][0])
		data['Phase'] /= (1 + data['Redshift'][0])
		
		int1d = interp1d(data['Wavelength'],data['Flux'],
						 fill_value=(0.0,0.0),bounds_error=False)#'extrapolate')
		newflux = int1d(outwave)
		phasex11 = np.append(phasex11,[data['Phase'][0]]*len(outwave))
		wavex11 = np.append(wavex11,outwave)
		fluxx11 = np.append(fluxx11,newflux)

	splinewave=np.linspace(2000,10000,8000/50,False)
	bsplx10 = bisplrep(phasex10,wavex10,fluxx10,tx=np.linspace(-20,60,20,False),ty=splinewave,kx=3,ky=3,task=-1)		
	outfluxx10 = bisplev(np.unique(outphase),np.unique(outwave),bsplx10)
	
	bsplx11 = bisplrep(phasex11,wavex11,fluxx11,tx=np.linspace(-20,60,20,False),ty=splinewave,kx=3,ky=3,task=-1)
	outfluxx11 = bisplev(np.unique(outphase),np.unique(outwave),bsplx11)

	outphaseunq,outwaveunq = np.unique(outphase),np.unique(outwave)
	import pdb; pdb.set_trace()

	
	fout = open('Kaepora_dm15_1.1.txt','w')
	for op in range(len(outphaseunq)):
		for ow in range(len(outwaveunq)):
			print('%.1f %.1f %8.5e'%(outphaseunq[op],outwaveunq[ow],outfluxx10[op,ow]),file=fout)
	fout.close()

	fout = open('Kaepora_dm15_0.94.txt','w')
	for op in range(len(outphaseunq)):
		for ow in range(len(outwaveunq)):
			print('%.1f %.1f %8.5e'%(outphaseunq[op],outwaveunq[ow],outfluxx11[op,ow]),file=fout)
	fout.close()
	
	return
