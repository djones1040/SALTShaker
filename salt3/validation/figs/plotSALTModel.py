#!/usr/bin/env python
# D. Jones - 4/16/19

import numpy as np
import pylab as plt
import sys
from scipy.interpolate import interp1d, interp2d
from sncosmo.salt2utils import SALT2ColorLaw
from salt3.initfiles import init_rootdir
from argparse import ArgumentParser
def mkModelPlot(salt3dir='modelfiles/salt3',
				xlimits=[2000,9200],outfile=None,plotErr=True):
	
	plt.figure(figsize=(5,8))
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
		np.loadtxt('%s/salt2_template_0.dat'%init_rootdir,unpack=True)
	salt2m1phase,salt2m1wave,salt2m1flux = \
		np.loadtxt('%s/salt2_template_1.dat'%init_rootdir,unpack=True)
	salt2m0errphase,salt2m0errwave,salt2m0fluxerr = \
		np.loadtxt('%s/salt2_lc_relative_variance_0.dat'%init_rootdir,unpack=True)
	salt2m1errphase,salt2m1errwave,salt2m1fluxerr = \
		np.loadtxt('%s/salt2_lc_relative_variance_1.dat'%init_rootdir,unpack=True)

	salt2m0flux = salt2m0flux.reshape([len(np.unique(salt2m0phase)),len(np.unique(salt2m0wave))])
	salt2m0fluxerr = salt2m0fluxerr.reshape([len(np.unique(salt2m0errphase)),len(np.unique(salt2m0errwave))])
	salt2m1flux = salt2m1flux.reshape([len(np.unique(salt2m1phase)),len(np.unique(salt2m1wave))])
	salt2m1fluxerr = salt2m1fluxerr.reshape([len(np.unique(salt2m1errphase)),len(np.unique(salt2m1errwave))])
	
	salt3m0flux = salt3m0flux.reshape([len(np.unique(salt3m0phase)),len(np.unique(salt3m0wave))])
	salt3m0fluxerr = salt3m0fluxerr.reshape([len(np.unique(salt3m0errphase)),len(np.unique(salt3m0errwave))])
	salt3m1flux = salt3m1flux.reshape([len(np.unique(salt3m1phase)),len(np.unique(salt3m1wave))])
	salt3m1fluxerr = salt3m1fluxerr.reshape([len(np.unique(salt3m1errphase)),len(np.unique(salt3m1errwave))])
	#print('hack!')
	#salt3m1flux /= 300

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
	
# 	if salt2m1wave.size > salt3m1wave.size:
# 		maxSalt3 = interp2d(salt3m1wave,salt3m1phase,salt3m1flux)(salt2m1wave,0)
# 		maxSalt2 = interp1d(salt2m1phase,salt2m1flux,axis=0)(0)
# 		sgn=np.sign(
# 	else:
# 		maxsalt2 = interp2d(salt2m1wave,salt2m1phase,salt2m1flux)(salt3m1wave,0)
# 		maxsalt3 = interp1d(salt3m1phase,salt3m1flux,axis=0)(0)
# 		sgn=np.sign(((np.sign(maxsalt3)==1) & (np.sign(maxsalt2)==1)).sum() - 0.5* salt3m1wave.size)
	maxSalt2 = interp1d(salt2m1phase,salt2m1flux,axis=0)(15)
	maxSalt3 = interp1d(salt3m1phase,salt3m1flux,axis=0)(15)
	sgn=np.sign(maxSalt3.sum())*np.sign(maxSalt2.sum())
	salt3m1flux*=sgn
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
						 salt2m0flux_0-np.sqrt(salt2m0fluxerr_0)+spacing*i,
						 salt2m0flux_0+np.sqrt(salt2m0fluxerr_0)+spacing*i,
						 color='b',alpha=0.5)
		ax1.plot(salt3m0wave,salt3m0flux_0+spacing*i,color='r',label='SALT3')
		ax1.fill_between(salt3m0wave,
						 salt3m0flux_0-np.sqrt(salt3m0fluxerr_0)+spacing*i,
						 salt3m0flux_0+np.sqrt(salt3m0fluxerr_0)+spacing*i,
						 color='r',alpha=0.5)
		ax1.set_xlim(xlimits)
		ax1.set_ylim([0,1.35])

		ax1.text(xlimits[1]-100,spacing*(i+0.2),'%s'%plotphasestr,ha='right')
		
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
		if plotErr: 
			ax2.fill_between(salt2m1wave,
							 salt2m1flux_0-np.sqrt(salt2m1fluxerr_0)+spacing*i,
							 salt2m1flux_0+np.sqrt(salt2m1fluxerr_0)+spacing*i,
							 color='b',alpha=0.5)
		m1scale = np.mean(np.abs(salt2m1flux_0[(salt2m1wave > 4000) & (salt2m1wave < 7000)]))/np.mean(np.abs(salt3m1flux_0[(salt3m1wave > 4000) & (salt3m1wave < 7000)]))
		ax2.plot(salt3m1wave,salt3m1flux_0+spacing*i,color='r',label='SALT3')
		if plotErr:
			ax2.fill_between(salt3m1wave,
							 salt3m1flux_0-np.sqrt(salt3m1fluxerr_0)+spacing*i,
							 salt3m1flux_0+np.sqrt(salt3m1fluxerr_0)+spacing*i,
							 color='r',alpha=0.5)
		ax2.set_xlim(xlimits)
		ax2.set_ylim([-0.05,0.39])

		ax2.text(xlimits[1]-100,spacing*(i+0.2),'%s'%plotphasestr,ha='right')
		
		#import pdb; pdb.set_trace()
		
	with open('%s/salt2_color_correction.dat'%init_rootdir) as fin:
		lines = fin.readlines()
	for i in range(len(lines)):
		lines[i] = lines[i].replace('\n','')
	colorlaw_salt2_coeffs = np.array(lines[1:5]).astype('float')
	salt2_colormin = float(lines[6].split()[1])
	salt2_colormax = float(lines[7].split()[1])
	colorlaw_salt2 = SALT2ColorLaw([salt2_colormin,salt2_colormax],colorlaw_salt2_coeffs)
	wave = np.arange(xlimits[0],xlimits[1],1,dtype='float')#salt2_colormin,salt2_colormax,1)
	ax3.plot(wave,colorlaw_salt2(wave),color='b',label='SALT2')

	
	with open('%s/salt3_color_correction.dat'%salt3dir) as fin:
		lines = fin.readlines()
	if len(lines):
		for i in range(len(lines)):
			lines[i] = lines[i].replace('\n','')
		colorlaw_salt3_coeffs = np.array(lines[1:5]).astype('float')
		salt3_colormin = float(lines[6].split()[1])
		salt3_colormax = float(lines[7].split()[1])

		colorlaw_salt3 = SALT2ColorLaw([salt3_colormin,salt3_colormax],colorlaw_salt3_coeffs)
		ax3.plot(wave,colorlaw_salt3(wave),color='r',label='SALT3')
	ax3.legend(prop={'size':13})
	ax1.set_ylabel('M0',fontsize=15)
	ax2.set_ylabel('M1',fontsize=15)
	ax3.set_ylabel('Color Law',fontsize=15)
	for ax in [ax1,ax2,ax3]:
		ax.set_xlim(xlimits)
	ax1.xaxis.set_ticklabels([])
	ax2.xaxis.set_ticklabels([])
	ax3.set_xlabel('Wavelength ($\AA$)',fontsize=15)
	plt.tight_layout()
	if not outfile is None:
		plt.savefig(outfile)
def mkModelErrPlot(salt3dir='modelfiles/salt3',xlimits=[2000,9200]):
	plt.rcParams['figure.figsize'] = (9,3)
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
						wspace=0, hspace=0)
	plt.clf()
	ax1 = plt.subplot(211)
	ax2 = plt.subplot(212)

	salt3scalephase,salt3scalewave,salt3scaleflux = \
		np.loadtxt('%s/salt3_lc_dispersion_scaling.dat'%salt3dir,unpack=True)
	salt3cdispphase,salt3cdispwave,salt3cdispflux = \
		np.loadtxt('%s/salt3_color_dispersion.dat'%salt3dir,unpack=True)

	salt2scalephase,salt2scalewave,salt2scaleflux = \
		np.loadtxt('modelfiles/salt2/salt2_lc_dispersion_scaling.dat',unpack=True)
	salt2cdispphase,salt2cdispwave,saltscdispflux = \
		np.loadtxt('modelfiles/salt2/salt2_color_dispersion.dat',unpack=True)

	salt2scaleflux = salt2scaleflux.reshape([len(np.unique(salt2scalephase)),len(np.unique(salt2scalewave))])
	salt2cdispflux = salt2cdispflux.reshape([len(np.unique(salt2cdispphase)),len(np.unique(salt2cdispwave))])

	salt3scaleflux = salt3scaleflux.reshape([len(np.unique(salt3scalephase)),len(np.unique(salt3scalewave))])
	salt3cdispflux = salt3cdispflux.reshape([len(np.unique(salt3cdispphase)),len(np.unique(salt3cdispwave))])

	salt2scalephase = np.unique(salt2scalephase)
	salt2scalewave = np.unique(salt2scalewave)
	salt2cdispphase = np.unique(salt2cdispphase)
	salt2cdispwave = np.unique(salt2cdispwave)

	salt3scalephase = np.unique(salt3scalephase)
	salt3scalewave = np.unique(salt3scalewave)
	salt3cdispphase = np.unique(salt3cdispphase)
	salt3cdispwave = np.unique(salt3cdispwave)

	spacing = 0.5
	for plotphase,i,plotphasestr in zip([-5,0,10],range(3),['-5','+0','+10']):
		int_salt2scale = interp2d(salt2scalewave,salt2scalephase,salt2scaleflux)
		salt2scaleflux_0 = int_salt2scale(salt2scalewave,plotphase)

		int_salt3scale = interp2d(salt3scalewave,salt3scalephase,salt3scaleflux)
		salt3scaleflux_0 = int_salt3scale(salt3scalewave,plotphase)

		ax1.plot(salt2scalewave,salt2scaleflux_0+spacing*i,color='b',label='SALT2')
		ax1.plot(salt3scalewave,salt3scaleflux_0+spacing*i,color='r',label='SALT3')
		ax1.set_xlim(xlimits)
		ax1.set_ylim([0,1.35])

		ax1.text(xlimits[1]-100,spacing*(i+0.2),'%s'%plotphasestr,ha='right')

	for plotphase,i,plotphasestr in zip([-5,0,10],range(3),['-5','+0','+10']):
		int_salt2cdisp = interp2d(salt2cdispwave,salt2cdispphase,salt2cdispflux)
		salt2cdispflux_0 = int_salt2cdisp(salt2cdispwave,plotphase)

		int_salt3cdisp = interp2d(salt3cdispwave,salt3cdispphase,salt3cdispflux)
		salt3cdispflux_0 = int_salt3cdisp(salt3cdispwave,plotphase)

		ax2.plot(salt2cdispwave,salt2cdispflux_0+spacing*i,color='b',label='SALT2')
		ax2.plot(salt3cdispwave,salt3cdispflux_0+spacing*i,color='r',label='SALT3')
		ax2.set_xlim(xlimits)
		ax2.set_ylim([0,1.35])

		ax2.text(xlimits[1]-100,spacing*(i+0.2),'%s'%plotphasestr,ha='right')
		
	
if __name__ == "__main__":
	parser=ArgumentParser(description='Plot SALT model components at peak and color law as compared to SALT2')
	parser.add_argument('modeldir',type=str,help='SALT3 model directory',default='model/salt3',nargs='?')
	parser.add_argument('outfile',type=str,help='File to save plots to',default=None,nargs='?')
	parser.add_argument('--noErr',dest='plotErr',help='Flag to choose whether or not to show model uncertainties on plot',action='store_const',const=False,default=True)
	parser=parser.parse_args()
	mkModelPlot(parser.modeldir ,outfile= '{}/SALTmodelcomp.pdf'.format(parser.modeldir) if parser.outfile is None else parser.outfile,plotErr=parser.plotErr)
