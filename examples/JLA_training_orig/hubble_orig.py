#!/usr/bin/env python
# D. Jones - 7/30/20

# cd /Users/David/Dropbox/research/SALT3/examples/JLA_training_orig_sim
# conda activate salt3
# python runpipe.py

import numpy as np
import glob
import pylab as plt
plt.ion()
from scipy.stats import binned_statistic
from scipy.optimize import least_squares,minimize
import getmu
import os
from matplotlib.gridspec import GridSpec
from txtobj import txtobj
from matplotlib.ticker import NullFormatter
plt.rcParams['figure.figsize'] = (10,4)

def salt2mu(x1=None,x1err=None,
            c=None,cerr=None,
            mb=None,mberr=None,
            cov_x1_c=None,cov_x1_x0=None,cov_c_x0=None,
            alpha=None,beta=None,hostmass=None,
            M=None,x0=None,sigint=None,z=None,peczerr=0.00083,deltam=None):

    sf = -2.5/(x0*np.log(10.0))
    cov_mb_c = cov_c_x0*sf
    cov_mb_x1 = cov_x1_x0*sf
    mu_out = mb + x1*alpha - beta*c + 19.36
    invvars = 1.0 / (mberr**2.+ alpha**2. * x1err**2. + beta**2. * cerr**2. + \
                         2.0 * alpha * (cov_x1_x0*sf) - 2.0 * beta * (cov_c_x0*sf) - \
                         2.0 * alpha*beta * (cov_x1_c) )

    if deltam:
        if len(np.where(hostmass > 10)[0]):
            mu_out[hostmass > 10] += deltam/2.
        if len(np.where(hostmass < 10)[0]):
            mu_out[hostmass < 10] -= deltam/2.

    zerr = peczerr*5.0/np.log(10)*(1.0+z)/(z*(1.0+z/2.0))
    muerr_out = np.sqrt(1/invvars + zerr**2. + 0.055**2.*z**2.)
    if sigint: muerr_out = np.sqrt(muerr_out**2. + sigint**2.)
    return(mu_out,muerr_out)

def concatfitres():
    frfiles = glob.glob('*FITRES.TEXT')

    count2,count3 = 0,0
    with open('JLA_TRAINING_SALT2.FITRES','w') as fout2,open('JLA_TRAINING_SALT3.FITRES','w') as fout3:
        for f in frfiles:
            if 'SALT2' not in f:
                with open(f) as fin:
                    for line in fin:
                        if (not line.startswith('#') and not line.startswith('VARNAMES')) or count3 == 0:
                            #if line.startswith('SN') and float(line.split()[8]) > 0:
                            #    print(line.replace('\n',''),file=fout3)
                            #elif not line.startswith('SN'):
                            print(line.replace('\n',''),file=fout3)

                count3 += 1
            else:
                with open(f) as fin:
                    for line in fin:
                        if (not line.startswith('#') and not line.startswith('VARNAMES')) or count2 == 0:
                            #if line.startswith('SN') and float(line.split()[8]) > 0:
                            #    print(line.replace('\n',''),file=fout2)
                            #elif not line.startswith('SN'):
                            print(line.replace('\n',''),file=fout2)
                                
                count2 += 1

def run_salt2mu():
    # run SALT2mu
    os.system('SALT2mu.exe SALT2mu.default file=JLA_TRAINING_SALT2.FITRES prefix=SALT2mu_SALT2')
    os.system('SALT2mu.exe SALT2mu.default file=JLA_TRAINING_SALT3.FITRES prefix=SALT2mu_SALT3')    

def errfnc(x):
    return(np.std(x)/np.sqrt(len(x)))
    
def main():
    
    # figure out alpha, beta for each.  comp against simulation
    with open('SALT2mu_SALT2.FITRES') as fin:
        for line in fin:
            if line.startswith('#  alpha0'): salt2alpha,salt2alphaerr = float(line.split()[3]),float(line.split()[5].replace('\n',''))
            if line.startswith('#  beta0'): salt2beta,salt2betaerr = float(line.split()[3]),float(line.split()[5].replace('\n',''))
            if line.startswith('#  sigint'): salt2sigint = float(line.split()[3])
    with open('SALT2mu_SALT3.FITRES') as fin:
        for line in fin:
            if line.startswith('#  alpha0'): salt3alpha,salt3alphaerr = float(line.split()[3]),float(line.split()[5].replace('\n',''))
            if line.startswith('#  beta0'): salt3beta,salt3betaerr = float(line.split()[3]),float(line.split()[5].replace('\n',''))
            if line.startswith('#  sigint'): salt3sigint = float(line.split()[3])

    # avg chi2/SN
    fr2 = txtobj('JLA_TRAINING_SALT2.FITRES',fitresheader=True)
    fr3 = txtobj('JLA_TRAINING_SALT3.FITRES',fitresheader=True)
    med_chi2 = np.median(fr2.FITCHI2/fr2.NDOF)
    med_chi3 = np.median(fr3.FITCHI2/fr3.NDOF)
    
    # hubble diagram w/ bins
    fr2 = getmu.getmu(fr2,salt2alpha=salt2alpha,salt2beta=salt2beta,sigint=0.0)
    fr3 = getmu.getmu(fr3,salt2alpha=salt3alpha,salt2beta=salt3beta,sigint=0.0)
    zbins = np.logspace(np.log10(0.01),np.log10(1.0),20)
    salt2mubins = binned_statistic(fr2.zCMB,fr2.mures,bins=zbins,statistic='mean').statistic
    salt2muerrbins = binned_statistic(fr2.zCMB,fr2.mures,bins=zbins,statistic=errfnc).statistic
    salt3mubins = binned_statistic(fr3.zCMB,fr3.mures,bins=zbins,statistic='mean').statistic
    salt3muerrbins = binned_statistic(fr3.zCMB,fr3.mures,bins=zbins,statistic=errfnc).statistic

    plt.subplots_adjust(wspace=0)
    fig = plt.figure()
    gs = GridSpec(1, 5, figure=fig)
    gs.update(wspace=0.0, hspace=0.0,bottom=0.2)
    ax1 = fig.add_subplot(gs[0, 0:4])
    ax2 = fig.add_subplot(gs[0, 4])
    ax1.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
    ax2.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
    
    ax1.axhline(0,lw=2,color='k')
    ax1.errorbar(fr2.zCMB,fr2.mures,yerr=fr2.muerr,fmt='o',color='b',alpha=0.1)
    ax1.errorbar(fr3.zCMB,fr3.mures,yerr=fr3.muerr,fmt='D',color='r',alpha=0.1)
    ax1.errorbar((zbins[1:]+zbins[:-1])/2.,salt2mubins,yerr=salt2muerrbins,fmt='o-',color='b')
    ax1.errorbar((zbins[1:]+zbins[:-1])/2.,salt3mubins,yerr=salt3muerrbins,fmt='D-',color='r')
    ax1.set_xscale('log')
    ax1.xaxis.set_major_formatter(NullFormatter())
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.xaxis.set_ticks([0.01,0.02,0.05,0.1,0.3,0.7])
    ax1.xaxis.set_ticklabels(['0.01','0.02','0.05','0.1','0.3','0.7'])
    
    muresbins = np.linspace(-1,1,40)
    ax2.hist(fr2.mures,bins=muresbins,orientation='horizontal',color='b',alpha=0.5)
    ax2.hist(fr3.mures,bins=muresbins,orientation='horizontal',color='r',alpha=0.5)

    ax1.text(0.03,0.97,fr"""SALT2
$\alpha = {salt2alpha:.3f}\pm{salt2alphaerr:.3f}$
$\beta = {salt2beta:.2f}\pm{salt2betaerr:.2f}$
$\sigma_{{int}}$ = {salt2sigint:.3f}
$med. \chi^2_{{\nu}}$ = {med_chi2:.2f}""",
             ha='left',va='top',color='b',transform=ax1.transAxes,bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.5})

    ax1.text(0.3,0.97,fr"""SALT3
$\alpha = {salt3alpha:.3f}\pm{salt3alphaerr:.3f}$
$\beta = {salt3beta:.2f}\pm{salt3betaerr:.2f}$
$\sigma_{{int}}$ = {salt3sigint:.3f}
$med. \chi^2_{{\nu}}$ = {med_chi3:.2f}""",
             ha='left',va='top',color='r',transform=ax1.transAxes,bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.5})

    ax2.text(0.03,0.97,f"""SALT2 RMS = {np.std(fr2.mures[fr2.zCMB > 0.015]):.3f}""",transform=ax2.transAxes,ha='left',va='top',color='b')
    ax2.text(0.03,0.92,f"""SALT3 RMS = {np.std(fr3.mures[fr3.zCMB > 0.015]):.3f}""",transform=ax2.transAxes,ha='left',va='top',color='r')
    ax2.yaxis.tick_right()
    ax1.set_ylim([-0.5,0.5])
    ax2.set_ylim([-0.5,0.5])
    ax1.set_xlabel('$z_{CMB}$',fontsize=15)
    ax1.set_ylabel('$\mu-\mu_{\Lambda CDM}$',fontsize=15)
    ax2.xaxis.set_ticks([])
    
    import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    concatfitres()
    run_salt2mu()
    main()
