#!/usr/bin/env python
# D. Jones - 6/29/21

import numpy as np
import pylab as plt
plt.ion()
from txtobj import txtobj
import astropy.table as at

import palettable
import glob
from palettable.colorbrewer.qualitative import Dark2_8 as palettable_color

surveydict = {'SALT3TRAIN_K21_CSPDR2':'CSP',
              'SALT3TRAIN_K21_Foundation_DR1':'Foundation',
              'SALT3TRAIN_K21_OTHER_LOWZ':'Other',
              'SALT3TRAIN_K21_SNLS3':'SNLS',
              'SALT3TRAIN_K21_CfA4p1':'CfA',
              'SALT3TRAIN_K21_Hamuy1996':'Other',#  'Calan/Tololo',
              'SALT3TRAIN_K21_PS1MD':'PS1MD',
              'SALT3TRAIN_K21_CfA4p2':'CfA',
              'SALT3TRAIN_K21_Hicken2009':'CfA',
              'SALT3TRAIN_K21_Riess1999':'CfA',
              'SALT3TRAIN_K21_DESSN3YR':'Other', #'DES',
              'SALT3TRAIN_K21_Jha2006':'CfA',
              'SALT3TRAIN_K21_SDSS':'SDSS'}

def cdf():
    fr = txtobj('hostpars_salt3_lowztraining.txt')

    # now figure out surveys for everything
    fr.survey = np.array(['NoneNoneNoneNoneNone']*len(fr.snid))
    for i,s in enumerate(fr.snid):
        snfile = glob.glob(f'SALT3TRAIN_K21/*/*{s}*')        
        try:
            fr.survey[i] = surveydict[snfile[0].split('/')[-2]]
        except:
            continue
    ax = plt.axes()
    ax.tick_params(top="on",bottom="on",left="on",right="on",direction="inout",length=8, width=1.5)
    ax.set_ylabel(r'Cumulative Fraction',fontsize=15)
    ax.set_xlabel(r'log($M_{\ast}/M_{\odot}$)',fontsize=15,labelpad=0)
    #import pdb; pdb.set_trace()
    massbins = np.linspace(7,13,150)
    for i,s in enumerate(np.unique(fr.survey)):
        if s == 'NoneNoneNoneNoneNone' or s == 'Other': continue
        mass_hist = np.histogram(fr.logmass[fr.survey == s],bins=massbins)
        ax.plot(mass_hist[1][:-1],
			    np.cumsum(mass_hist[0])/float(np.sum(mass_hist[0])),drawstyle='steps-mid',
			    label=s,color='C'+str(i),lw=2)
    
    ax.set_xlim([8,11.5])
    
    ax.legend(prop={'size':12})
    ax.set_ylim([0,1.05])
    ax.axvline(10,color='0.5',ls='--')
    plt.savefig('salt3_masses.png',dpi=200)
    import pdb; pdb.set_trace()

    
if __name__ == "__main__":
    cdf()
