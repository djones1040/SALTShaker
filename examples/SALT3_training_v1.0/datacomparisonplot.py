#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import argparse
import configparser
import numpy as np
import sys
import multiprocessing
import pickle

import os
from os import path

from scipy.linalg import lstsq
from scipy.optimize import minimize, least_squares, differential_evolution
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from sncosmo.constants import HC_ERG_AA
import astropy.table as at

from saltshaker.util import snana,readutils
from saltshaker.util.estimate_tpk_bazin import estimate_tpk_bazin
from saltshaker.util.txtobj import txtobj
from saltshaker.util.specSynPhot import getScaleForSN
from saltshaker.util.specrecal import SpecRecal

from saltshaker.training.init_hsiao import init_hsiao, init_kaepora, init_errs,init_errs_percent,init_custom,init_salt2
from saltshaker.training.base import TrainSALTBase
from saltshaker.training.saltfit import fitting
from saltshaker.training import saltfit as saltfit
from saltshaker.validation import ValidateParams,datadensity

from saltshaker.data import data_rootdir
from saltshaker.initfiles import init_rootdir
from saltshaker.config import config_rootdir,loggerconfig

import astropy.units as u
import sncosmo
import yaml

from astropy.table import Table
from saltshaker.initfiles import init_rootdir as salt2dir
from saltshaker.training.TrainSALT import TrainSALT
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick,colors


# In[ ]:





# In[ ]:





# In[5]:


def getdatadict(dirname,configfile):
    os.chdir(dirname)

    salt = TrainSALT()

    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(configfile)

    user_parser = salt.add_user_options(usage='',config=config)
    user_options = user_parser.parse_known_args()[0]

    # loggerconfig.dictconfigfromYAML(user_options.loggingconfig,user_options.outputdir)

    trainingconfig = configparser.ConfigParser(inline_comment_prefixes='#')
    trainingconfig.read(user_options.trainingconfig)
    training_parser = salt.add_training_options(parser=user_parser,
        usage='',config=trainingconfig)
    training_options = training_parser.parse_args(args=[])

    salt.options = training_options
    salt.verbose = training_options.verbose
    salt.clobber = training_options.clobber
    if salt.options.binspec:
        binspecres = salt.options.binspecres
    else:
        binspecres = None

    datadict = readutils.rdAllData(salt.options.snlists,salt.options.estimate_tpk,
                               dospec=salt.options.dospec,
                               peakmjdlist=salt.options.tmaxlist,
                               binspecres=binspecres,snparlist=salt.options.snparlist,maxsn=salt.options.maxsn)


    salt.kcordict=readutils.rdkcor(salt.surveylist,salt.options)
    datadict = salt.mkcuts(datadict)[0]

    return salt,datadict

salt2obj,salt2data=getdatadict('../JLA_training_orig/','Train_JLA_origlc.conf')
salt3obj,salt3data=getdatadict('../SALT3_training_v1.0/','Train_SALT3.conf')


# In[13]:


import logging
log=logging.getLogger(__name__)


# In[ ]:





# In[ ]:





# In[7]:


salt=salt3obj
phasebins=np.linspace(*salt.options.phaserange,int((salt.options.phaserange[1]-salt.options.phaserange[0])/salt.options.phasesplineres)+1,True)
wavebins=np.linspace(*salt.options.waverange,int((salt.options.waverange[1]-salt.options.waverange[0])/salt.options.wavesplineres)+1,True)
# datadensity.datadensityplot(path.join(salt.options.outputdir,'datadensity.pdf') ,phasebins,wavebins,datadict,salt.kcordict)


# In[16]:





# In[11]:


#salt3phot,salt3spec=photdensity,specdensity
salt2phot,salt2spec=datadensity.getphotdensity(phasebins,wavebins,salt2data,salt2obj.kcordict),datadensity.getspecdensity(phasebins,wavebins,salt2data)
salt3phot,salt3spec=datadensity.getphotdensity(phasebins,wavebins,salt3data,salt3obj.kcordict),datadensity.getspecdensity(phasebins,wavebins,salt3data)




# In[12]:


temp=plt.cm.jet._segmentdata.copy()
for color in temp:
    temp[color]=list(temp[color])
    temp[color][0]=(temp[color][0][0],1,temp[color][0][2])
    temp[color]=[(0,1,1)]+temp[color]


discontmap=colors.LinearSegmentedColormap('jetdiscont',temp)
ticksize=15
def tickspacing(n):
	if n==0: return np.linspace(0,1,4,True)
	leadingzeros=10**np.log10(n).astype(int)
	leadingdigit=n//leadingzeros
	spacing= leadingzeros/2 if leadingdigit==1 else leadingzeros
	return np.arange(0,n, spacing)

kcordict=salt.kcordict


# In[ ]:





# In[13]:


fig,axes=plt.subplots(2,4,sharex=True,sharey=True,squeeze=True,figsize=np.array([6,4])*7/6,)
labelsize=11
ticksize=8

axes
axes,coloraxes=[x[:-1] for x in axes], [x[-1] for x in axes]

for name,axlist,colorax,datacomb in [('spec', axes[1],coloraxes[1],[salt2spec,salt3spec-salt2spec,salt3spec]), ('phot',axes[0],coloraxes[0],[salt2phot,salt3phot-salt2phot,salt3phot])]:

    top=max([x.max() for x in datacomb])
    for ax,data in zip(axlist,datacomb):
        i=0
        plt.sca(ax)
        scale=0,top
        colorax.axis('off')
        #normalize=colors.SymLogNorm(linthresh=scale[1]**.5, linscale=1, vmin=scale[0], vmax=scale[1], base=10)
        normalize=colors.Normalize(0,top)
        image=plt.imshow(data,extent=[0,5,5,0],norm=normalize,cmap=discontmap,interpolation='nearest')
        ax.set_aspect('auto')


    label='Number of\n light curves' if (name == 'phot' )else 'Number of\n spectra'
    cbar=fig.colorbar(image,ax=colorax,fraction=1, pad=0.04,label=label,ticks=tickspacing(top))
    if name == 'phot':
        cbar.ax.tick_params(labelsize=ticksize)
        cbar.set_label(label=label,size=labelsize,labelpad=8)


    else:
        cbar.ax.tick_params(labelsize=ticksize)

        cbar.set_label(label=label,size=labelsize,labelpad=18)

#     cbar=plt.colorbar()
    #cbar.ax.tick_params(labelsize=ticksize)
for ax in axes[-1]:
    ticklocs=np.linspace(0,5,4,True)
    ax.set_xticks(ticklocs)
    ax.set_xticklabels(labels=[f'{tick*(wavebins[-1]-wavebins[0])/(ticklocs[-1])+wavebins[0]:.0f}' for tick in ticklocs],fontsize=ticksize,rotation=-45)



axes[1][1].set_xlabel('Wavelength ($\mathrm{\AA}$)',fontsize=labelsize)
axes[0][0].set_ylabel('Phase (days)',fontsize=labelsize,verticalalignment='top')
axes[0][0].yaxis.set_label_coords(-0.5,0)
for ax in [x[0] for x in axes]:
    ticklocs=np.linspace(0,5,5,True)
    ax.set_yticks(ticklocs)
    ax.set_yticklabels(labels=[f'{tick*(phasebins[-1]-phasebins[0])/(ticklocs[-1])+phasebins[0]:.0f}' for tick in ticklocs],fontsize=ticksize)
plt.savefig('temp.pdf')


# In[47]:


fig,axes=plt.subplots(2,6,sharex=True,sharey=True,squeeze=True,figsize=np.array([6,4])*7/6,gridspec_kw={'width_ratios':[1,.1,1,.1,1,.5]})
labelsize=11
ticksize=8

axes
for ax in axes[:,1::2].flatten(): ax.axis('off')
for ax in axes[:,1]: ax.text(.4,.5,'+',fontsize=24,transform=ax.transAxes,horizontalalignment='center',verticalalignment='center')
for ax in axes[:,3]: ax.text(.4,.5,'=',fontsize=24,transform=ax.transAxes,horizontalalignment='center',verticalalignment='center')

axes,coloraxes=[x[::2] for x in axes], [x[-1] for x in axes]
 
for name,axlist,colorax,datacomb in [('spec', axes[1],coloraxes[1],[salt2spec,salt3spec-salt2spec,salt3spec]), ('phot',axes[0],coloraxes[0],[salt2phot,salt3phot-salt2phot,salt3phot])]:

    top=max([x.max() for x in datacomb])
    for ax,data in zip(axlist,datacomb):
        i=0
        plt.sca(ax)
        scale=0,top
        #normalize=colors.SymLogNorm(linthresh=scale[1]**.5, linscale=1, vmin=scale[0], vmax=scale[1], base=10)
        normalize=colors.Normalize(0,top)
        image=plt.imshow(data,extent=[0,5,5,0],norm=normalize,cmap=discontmap,interpolation='nearest')
        ax.set_aspect('auto')


    label='Number of\n light curves' if (name == 'phot' )else 'Number of\n spectra'
    cbar=fig.colorbar(image,ax=colorax,fraction=1, pad=0.04,label=label,ticks=tickspacing(top))
    if name == 'phot':
        cbar.ax.tick_params(labelsize=ticksize)
        cbar.set_label(label=label,size=labelsize,labelpad=8)
        for ax,title in zip(axlist, ['JLA sample','K21 add.', 'K21 Full' ]):
            ax.set_title(title)

    else:
        cbar.ax.tick_params(labelsize=ticksize)

        cbar.set_label(label=label,size=labelsize,labelpad=18)

#     cbar=plt.colorbar()
    #cbar.ax.tick_params(labelsize=ticksize)
for ax in axes[-1]:
    ticklocs=np.linspace(0,5,4,True)
    ax.set_xticks(ticklocs)
    ax.set_xticklabels(labels=[f'{tick*(wavebins[-1]-wavebins[0])/(ticklocs[-1])+wavebins[0]:.0f}' for tick in ticklocs],fontsize=ticksize,rotation=-45)



axes[1][1].set_xlabel('Wavelength ($\mathrm{\AA}$)',fontsize=labelsize)
axes[0][0].set_ylabel('Phase (days)',fontsize=labelsize,verticalalignment='top')
axes[0][0].yaxis.set_label_coords(-0.5,0)
for ax in [x[0] for x in axes]:
    ticklocs=np.linspace(0,5,5,True)
    ax.set_yticks(ticklocs)
    ax.set_yticklabels(labels=[f'{tick*(phasebins[-1]-phasebins[0])/(ticklocs[-1])+phasebins[0]:.0f}' for tick in ticklocs],fontsize=ticksize)
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('datacoverage.pdf')


# In[32]:





# In[ ]:




