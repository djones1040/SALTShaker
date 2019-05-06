#!/usr/bin/env python
# D. Jones - 4/16/19

import numpy as np
import pylab as plt
from scipy.interpolate import interp1d, interp2d
from sncosmo.salt2utils import SALT2ColorLaw

def mkModelPlot(salt3dir='modelfiles/salt3'):
	plt.rcParams['figure.figsize'] = (9,3)
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
						wspace=0, hspace=0)
	plt.clf()
	ax1 = plt.subplot(311)
	ax2 = plt.subplot(312)
	ax3 = plt.subplot(313)

	salt3m0phase,salt3m0wave,salt3m0flux = \
		np.loadtxt('%s/salt3_template_0.dat'%salt3dir,unpack=True)
	salt3m1phase,salt3m1wave,salt3m1flux = \
		np.loadtxt('%s/salt3_template_1.dat'%salt3dir,unpack=True)
	salt3m0errphase,salt3m0errwave,salt3m0fluxerr = \
		np.loadtxt('%s/salt3_lc_relative_variance_0.dat'%salt3dir,unpack=True)
	salt3m1errphase,salt3m1errwave,salt3m1fluxerr = \
		np.loadtxt('%s/salt3_lc_relative_variance_1.dat'%salt3dir,unpack=True)

	salt2m0phase,salt2m0wave,salt2m0flux = \
		np.loadtxt('modelfiles/salt2/salt2_template_0.dat',unpack=True)
	salt2m1phase,salt2m1wave,salt2m1flux = \
		np.loadtxt('modelfiles/salt2/salt2_template_1.dat',unpack=True)
	salt2m0errphase,salt2m0errwave,salt2m0fluxerr = \
		np.loadtxt('modelfiles/salt2/salt2_lc_relative_variance_0.dat',unpack=True)
	salt2m1errphase,salt2m1errwave,salt2m1fluxerr = \
		np.loadtxt('modelfiles/salt2/salt2_lc_relative_variance_1.dat',unpack=True)

	salt2m0flux = salt2m0flux.reshape([len(np.unique(salt2m0phase)),len(np.unique(salt2m0wave))])
	salt2m0fluxerr = salt2m0fluxerr.reshape([len(np.unique(salt2m0errphase)),len(np.unique(salt2m0errwave))])
	salt2m1flux = salt2m1flux.reshape([len(np.unique(salt2m1phase)),len(np.unique(salt2m1wave))])
	salt2m1fluxerr = salt2m1fluxerr.reshape([len(np.unique(salt2m1errphase)),len(np.unique(salt2m1errwave))])

	salt3m0flux = salt3m0flux.reshape([len(np.unique(salt3m0phase)),len(np.unique(salt3m0wave))])
	salt3m0fluxerr = salt3m0fluxerr.reshape([len(np.unique(salt3m0errphase)),len(np.unique(salt3m0errwave))])
	salt3m1flux = salt3m1flux.reshape([len(np.unique(salt3m1phase)),len(np.unique(salt3m1wave))])
	salt3m1fluxerr = salt3m1fluxerr.reshape([len(np.unique(salt3m1errphase)),len(np.unique(salt3m1errwave))])

	salt2m0phase = np.unique(salt2m0phase)
	salt2m0wave = np.unique(salt2m0wave)
	salt2m1phase = np.unique(salt2m1phase)
	salt2m1wave = np.unique(salt2m1wave)

	salt3m0phase = np.unique(salt3m0phase)
	salt3m0wave = np.unique(salt3m0wave)
	salt3m1phase = np.unique(salt3m1phase)
	salt3m1wave = np.unique(salt3m1wave)

	salt2m0errphase = np.unique(salt2m0errphase)
	salt2m0errwave = np.unique(salt2m0errwave)
	salt2m1errphase = np.unique(salt2m1errphase)
	salt2m1errwave = np.unique(salt2m1errwave)

	salt3m0errphase = np.unique(salt3m0errphase)
	salt3m0errwave = np.unique(salt3m0errwave)
	salt3m1errphase = np.unique(salt3m1errphase)
	salt3m1errwave = np.unique(salt3m1errwave)


	spacing = 0.5
	for plotphase,i,plotphasestr in zip([-5,0,10],range(3),['-5','+0','+10']):
		int_salt2m0 = interp2d(salt2m0wave,salt2m0phase,salt2m0flux)
		int_salt2m0err = interp2d(salt2m0errwave,salt2m0errphase,salt2m0fluxerr)
		salt2m0flux_0 = int_salt2m0(salt2m0wave,plotphase)
		salt2m0fluxerr_0 = int_salt2m0err(salt2m0wave,plotphase)

		int_salt3m0 = interp2d(salt3m0wave,salt3m0phase,salt3m0flux)
		int_salt3m0err = interp2d(salt3m0errwave,salt3m0errphase,salt3m0fluxerr)
		salt3m0flux_0 = int_salt3m0(salt3m0wave,plotphase)
		salt3m0fluxerr_0 = int_salt3m0err(salt3m0wave,plotphase)

		ax1.plot(salt2m0wave,salt2m0flux_0+spacing*i,color='b',label='SALT2')
		ax1.fill_between(salt2m0wave,
						 salt2m0flux_0-salt2m0fluxerr_0+spacing*i,
						 salt2m0flux_0+salt2m0fluxerr_0+spacing*i,
						 color='b',alpha=0.5)
		ax1.plot(salt3m0wave,salt3m0flux_0+spacing*i,color='r',label='SALT3')
		ax1.fill_between(salt3m0wave,
						 salt3m0flux_0-salt3m0fluxerr_0+spacing*i,
						 salt3m0flux_0+salt3m0fluxerr_0+spacing*i,
						 color='r',alpha=0.5)
		ax1.set_xlim([2500,9200])
		ax1.set_ylim([0,1.35])

		ax1.text(9100,spacing*(i+0.2),'%s'%plotphasestr,ha='right')
		
	spacing = 0.15		
	for plotphase,i,plotphasestr in zip([-5,0,10],range(3),['-5','+0','+10']):
		int_salt2m1 = interp2d(salt2m1wave,salt2m1phase,salt2m1flux)
		int_salt2m1err = interp2d(salt2m1errwave,salt2m1errphase,salt2m1fluxerr)
		salt2m1flux_0 = int_salt2m1(salt2m1wave,plotphase)
		salt2m1fluxerr_0 = int_salt2m1err(salt2m1wave,plotphase)

		int_salt3m1 = interp2d(salt3m1wave,salt3m1phase,salt3m1flux)
		int_salt3m1err = interp2d(salt3m1errwave,salt3m1errphase,salt3m1fluxerr)
		salt3m1flux_0 = int_salt3m1(salt3m1wave,plotphase)
		salt3m1fluxerr_0 = int_salt3m1err(salt3m1wave,plotphase)

		ax2.plot(salt2m1wave,salt2m1flux_0+spacing*i,color='b',label='SALT2')
		ax2.fill_between(salt2m1wave,
						 salt2m1flux_0-salt2m1fluxerr_0+spacing*i,
						 salt2m1flux_0+salt2m1fluxerr_0+spacing*i,
						 color='b',alpha=0.5)
		ax2.plot(salt3m1wave,salt3m1flux_0+spacing*i,color='r',label='SALT3')
		ax2.fill_between(salt3m1wave,
						 salt3m1flux_0-salt3m1fluxerr_0+spacing*i,
						 salt3m1flux_0+salt3m1fluxerr_0+spacing*i,
						 color='r',alpha=0.5)
		ax2.set_xlim([2500,9200])
		ax2.set_ylim([-0.05,0.4])

		ax2.text(9100,spacing*(i+0.2),'%s'%plotphasestr,ha='right')
		
		#import pdb; pdb.set_trace()
		
	with open('modelfiles/salt2/salt2_color_correction.dat') as fin:
		lines = fin.readlines()
	for i in range(len(lines)):
		lines[i] = lines[i].replace('\n','')
	colorlaw_salt2_coeffs = np.array(lines[1:5]).astype('float')
	salt2_colormin = float(lines[6].split()[1])
	salt2_colormax = float(lines[7].split()[1])
	colorlaw_salt2 = SALT2ColorLaw([salt2_colormin,salt2_colormax],colorlaw_salt2_coeffs)
	wave = np.arange(salt2_colormin,salt2_colormax,1)
	ax3.plot(wave,colorlaw_salt2(wave),color='b')

	
	with open('%s/salt3_color_correction.dat'%salt3dir) as fin:
		lines = fin.readlines()
	if len(lines):
		for i in range(len(lines)):
			lines[i] = lines[i].replace('\n','')
		colorlaw_salt3_coeffs = np.array(lines[1:5]).astype('float')
		salt3_colormin = float(lines[6].split()[1])
		salt3_colormax = float(lines[7].split()[1])

		colorlaw_salt3 = SALT2ColorLaw([salt3_colormin,salt3_colormax],colorlaw_salt3_coeffs)
		ax3.plot(wave,colorlaw_salt3(wave),color='r')
		
if __name__ == "__main__":
	mkModelPlot()
