#!/usr/bin/env python
# D. Jones - 7/30/20

# cd /Users/David/Dropbox/research/SALT3/examples/SALT3_training_v1.0
# conda activate salt3
# python runpipe.py

import numpy as np
import glob
import pylab as plt
plt.ion()
from scipy.stats import binned_statistic
from scipy.optimize import least_squares,minimize
from util import getmu
import os
from matplotlib.gridspec import GridSpec
from util.txtobj import txtobj
from matplotlib.ticker import NullFormatter
from astropy.stats import sigma_clipped_stats
import textwrap
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
import math
import sys
plt.rcParams['figure.figsize'] = (10,4)

__band_order__=np.append(['u','b','g','v','r','i','z','y','j','h','k'],
     [x.upper() for x in ['u','b','g','v','r','i','z','y','j','h','k']])

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
    frfiles = glob.glob('fit_output/*FITRES.TEXT')

    count2,count3 = 0,0
    with open('fit_output/JLA_TRAINING_SALT2.FITRES','w') as fout2,open('fit_output/JLA_TRAINING_SALT3.FITRES','w') as fout3:
        print(frfiles)
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

def concatfitres_valid():
    frfiles = glob.glob('fit_output_valid/*FITRES.TEXT')

    count2,count3 = 0,0
    with open('fit_output_valid/JLA_TRAINING_SALT2.FITRES','w') as fout2,open('fit_output_valid/JLA_TRAINING_SALT3.FITRES','w') as fout3:
        print(frfiles)
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
    os.system('SALT2mu.exe SALT2mu.default file=fit_output/JLA_TRAINING_SALT2.FITRES prefix=SALT2mu_SALT2')
    os.system('SALT2mu.exe SALT2mu.default file=fit_output/JLA_TRAINING_SALT3.FITRES prefix=SALT2mu_SALT3')    

def run_salt2mu_valid():
    # run SALT2mu
    os.system('SALT2mu.exe SALT2mu.default file=fit_output_valid/JLA_TRAINING_SALT2.FITRES prefix=SALT2mu_VALID_SALT2')
    os.system('SALT2mu.exe SALT2mu.default file=fit_output_valid/JLA_TRAINING_SALT3.FITRES prefix=SALT2mu_VALID_SALT3')    

    
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
    #fr2 = txtobj('fit_output/JLA_TRAINING_SALT2.FITRES',fitresheader=True)
    #fr3 = txtobj('fit_output/JLA_TRAINING_SALT3.FITRES',fitresheader=True)
    fr2 = txtobj('SALT2mu_SALT2.FITRES',fitresheader=True)
    fr3 = txtobj('SALT2mu_SALT3.FITRES',fitresheader=True)
    
    # hubble diagram w/ bins
    fr2 = getmu.getmu(fr2,salt2alpha=salt2alpha,salt2beta=salt2beta,sigint=0.0)
    fr3 = getmu.getmu(fr3,salt2alpha=salt3alpha,salt2beta=salt3beta,sigint=0.0)
    fr2 = getmu.mkcuts(fr2,salt2alpha=salt2alpha,salt2beta=salt2beta,fitprobmin=0.0)#1e-5)
    fr3 = getmu.mkcuts(fr3,salt2alpha=salt3alpha,salt2beta=salt3beta,fitprobmin=0.0)#1e-5)

    iGood2 = np.array([],dtype=int)
    for j,i in enumerate(fr2.CID):
        if i in fr3.CID:
            iGood2 = np.append(iGood2,j)
    for k in fr2.__dict__.keys():
        fr2.__dict__[k] = fr2.__dict__[k][iGood2]
    iGood3 = np.array([],dtype=int)
    for j,i in enumerate(fr3.CID):
        if i in fr2.CID:
            iGood3 = np.append(iGood3,j)
    for k in fr3.__dict__.keys():
        fr3.__dict__[k] = fr3.__dict__[k][iGood3]
    
    mn2,md2,std2 = sigma_clipped_stats(fr2.mures)
    mn3,md3,std3 = sigma_clipped_stats(fr3.mures)
    iOut2 = len(np.where(((fr2.mures < md2-3*std2) | (fr2.mures > md2+3*std2)) & (fr2.zCMB > 0.015))[0])/float(len(fr2.CID))
    iOut3 = len(np.where(((fr3.mures < md3-3*std3) | (fr3.mures > md3+3*std3)) & (fr3.zCMB > 0.015))[0])/float(len(fr3.CID))
    #iClip2 = np.where((fr2.mures > md2-3*std2) & (fr2.mures < md2+3*std2))[0]
    #iClip3 = np.where((fr3.mures > md3-3*std3) & (fr3.mures < md3+3*std3))[0]
    #for k in fr2.__dict__.keys():
    #    fr2.__dict__[k] = fr2.__dict__[k][iClip2]
    #for k in fr3.__dict__.keys():
    #    fr3.__dict__[k] = fr3.__dict__[k][iClip3]

    #iClip2 = fr2.IDSURVEY == 150
    #iClip3 = fr3.IDSURVEY == 150
    #for k in fr2.__dict__.keys():
    #    fr2.__dict__[k] = fr2.__dict__[k][iClip2]
    #for k in fr3.__dict__.keys():
    #    fr3.__dict__[k] = fr3.__dict__[k][iClip3]
    
    
    med_chi2 = np.median(fr2.FITCHI2/fr2.NDOF)
    med_chi3 = np.median(fr3.FITCHI2/fr3.NDOF)
        
    zbins = np.logspace(np.log10(0.01),np.log10(1.0),20)
    salt2mubins = binned_statistic(fr2.zCMB,fr2.mures-np.median(fr2.mures[fr2.zCMB > 0.01]),
                                   bins=zbins,statistic='mean').statistic
    salt2muerrbins = binned_statistic(fr2.zCMB,fr2.mures-np.median(fr2.mures[fr2.zCMB > 0.01]),
                                      bins=zbins,statistic=errfnc).statistic
    salt3mubins = binned_statistic(fr3.zCMB,fr3.mures-np.median(fr3.mures[fr3.zCMB > 0.01]),
                                   bins=zbins,statistic='mean').statistic
    salt3muerrbins = binned_statistic(fr3.zCMB,fr3.mures-np.median(fr3.mures[fr3.zCMB > 0.01]),
                                      bins=zbins,statistic=errfnc).statistic

    plt.subplots_adjust(wspace=0)
    fig = plt.figure()
    gs = GridSpec(1, 5, figure=fig)
    gs.update(wspace=0.0, hspace=0.0,bottom=0.2)
    ax1 = fig.add_subplot(gs[0, 0:4])
    ax2 = fig.add_subplot(gs[0, 4])
    ax1.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
    ax2.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
    
    ax1.axhline(0,lw=2,color='k')
    ax1.errorbar(fr2.zCMB,fr2.mures-np.median(fr2.mures[fr2.zCMB > 0.01]),yerr=fr2.muerr,fmt='o',color='b',alpha=0.1)
    ax1.errorbar(fr3.zCMB,fr3.mures-np.median(fr3.mures[fr3.zCMB > 0.01]),yerr=fr3.muerr,fmt='D',color='r',alpha=0.1)
    ax1.errorbar((zbins[1:]+zbins[:-1])/2.,salt2mubins,yerr=salt2muerrbins,fmt='o-',color='b')
    ax1.errorbar((zbins[1:]+zbins[:-1])/2.,salt3mubins,yerr=salt3muerrbins,fmt='D-',color='r')
    ax1.set_xscale('log')
    ax1.xaxis.set_major_formatter(NullFormatter())
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.xaxis.set_ticks([0.01,0.02,0.05,0.1,0.3,0.7])
    ax1.xaxis.set_ticklabels(['0.01','0.02','0.05','0.1','0.3','0.7'])
    
    muresbins = np.linspace(-1,1,40)
    ax2.hist(fr2.mures[fr2.zCMB > 0.01]-np.median(fr2.mures[fr2.zCMB > 0.01]),bins=muresbins,
             orientation='horizontal',color='b',alpha=0.5)
    ax2.hist(fr3.mures[fr3.zCMB > 0.01]-np.median(fr3.mures[fr3.zCMB > 0.01]),bins=muresbins,
             orientation='horizontal',color='r',alpha=0.5)

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

    ax2.text(0.97,0.97,f"""SALT2 RMS = {np.std(fr2.mures[fr2.zCMB > 0.015]):.3f}
{iOut2*len(fr2.CID):.0f} outliers ({iOut2*100:.1f}%)""",transform=ax2.transAxes,ha='right',va='top',color='b')
    ax2.text(0.97,0.03,f"""SALT3 RMS = {np.std(fr3.mures[fr3.zCMB > 0.015]):.3f}
{iOut3*len(fr3.CID):.0f} outliers ({iOut3*100:.1f}%)""",transform=ax2.transAxes,ha='right',va='bottom',color='r')
    ax2.yaxis.tick_right()
    ax1.set_ylim([-0.5,0.5])
    ax2.set_ylim([-0.5,0.5])
    ax1.set_xlabel('$z_{CMB}$',fontsize=15)
    ax1.set_ylabel('$\mu-\mu_{\Lambda CDM}$',fontsize=15)
    ax2.xaxis.set_ticks([])
    plt.savefig('hubblediagsalt3.pdf')
    import pdb; pdb.set_trace()

def plot_outliers():

    idsurvey_map = {1:('fit_output/SDSS.FITRES.TEXT','fit_output/SDSS_SALT2.FITRES.TEXT','SDSS',
                       'pipeline/SDSS.NML','pipeline/SDSS_SALT2.NML'),
                    10:('fit_output/DES-SN3YR_DES.FITRES.TEXT','fit_output/DES-SN3YR_DES_SALT2.FITRES.TEXT','DES-SN3YR_DES',
                        'pipeline/DES-SN3YR_DES.NML','pipeline/DES-SN3YR_DES_SALT2.NML'),
                    15:('fit_output/Pantheon_PS1MD_TEXT.FITRES.TEXT','fit_output/Pantheon_PS1MD_TEXT_SALT2.FITRES.TEXT',
                        'Pantheon_PS1MD_TEXT','pipeline/Pantheon_PS1MD_TEXT.NML','pipeline/Pantheon_PS1MD_TEXT_SALT2.NML'),
                    150:('fit_output/Foundation_DR1.FITRES.TEXT','fit_output/Foundation_DR1_SALT2.FITRES.TEXT',
                         'Foundation_DR1','pipeline/Foundation_DR1.NML','pipeline/Foundation_DR1.NML'),
                    191:('fit_output/Hamuy1996_LC.FITRES.TEXT','fit_output/Hamuy1996_LC_SALT2.FITRES.TEXT',
                         'Hamuy1996_LC','pipeline/Hamuy1996_LC.NML','pipeline/Hamuy1996_LC_SALT2.NML'),
                    192:('fit_output/Hicken2009_LC.FITRES.TEXT','fit_output/Hicken2009_LC_SALT2.FITRES.TEXT',
                         'Hicken2009_LC','pipeline/Hicken2009_LC.NML','pipeline/Hicken2009_LC_SALT2.NML'),
                    193:('fit_output/Jha2006_LC.FITRES.TEXT','fit_output/Jha2006_LC_SALT2.FITRES.TEXT',
                         'Jha2006_LC','pipeline/Jha2006_LC.NML','pipeline/Jha2006_LC_SALT2.NML'),
                    194:('fit_output/OTHER_LOWZ_LC.FITRES.TEXT','fit_output/OTHER_LOWZ_LC_SALT2.FITRES.TEXT',
                         'OTHER_LOWZ_LC','pipeline/OTHER_LOWZ_LC.NML','pipeline/OTHER_LOWZ_LC_SALT2.NML'),
                    195:('fit_output/Riess1999_LC.FITRES.TEXT','fit_output/Riess1999_LC_SALT2.FITRES.TEXT',
                         'Riess1999_LC','pipeline/Riess1999_LC.NML','pipeline/Riess1999_LC_SALT2.NML'),
                    196:('fit_output/SNLS3_LC.FITRES.TEXT','fit_output/SNLS3_LC_SALT2.FITRES.TEXT',
                         'SNLS3_LC','pipeline/SNLS3_LC.NML','pipeline/SNLS3_LC_SALT2.NML')}
    
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
    fr2 = txtobj('fit_output/JLA_TRAINING_SALT2.FITRES',fitresheader=True)
    fr3 = txtobj('fit_output/JLA_TRAINING_SALT3.FITRES',fitresheader=True)
    
    # hubble diagram w/ bins
    fr2 = getmu.getmu(fr2,salt2alpha=salt2alpha,salt2beta=salt2beta,sigint=0.0)
    fr3 = getmu.getmu(fr3,salt2alpha=salt3alpha,salt2beta=salt3beta,sigint=0.0)
    #fr2 = getmu.mkcuts(fr2,salt2alpha=salt2alpha,salt2beta=salt2beta)
    #fr3 = getmu.mkcuts(fr3,salt2alpha=salt3alpha,salt2beta=salt3beta)

    iGood2 = np.array([],dtype=int)
    for j,i in enumerate(fr2.CID):
        if i in fr3.CID:
            iGood2 = np.append(iGood2,j)
    for k in fr2.__dict__.keys():
        fr2.__dict__[k] = fr2.__dict__[k][iGood2]
    iGood3 = np.array([],dtype=int)
    for j,i in enumerate(fr3.CID):
        if i in fr2.CID:
            iGood3 = np.append(iGood3,j)
    for k in fr3.__dict__.keys():
        fr3.__dict__[k] = fr3.__dict__[k][iGood3]
            
    mn2,md2,std2 = sigma_clipped_stats(fr2.mures)
    mn3,md3,std3 = sigma_clipped_stats(fr2.mures)
    #iOut2 = np.where(((fr2.mures < md2-3*std2) | (fr2.mures > md2+3*std2)) & (fr2.zCMB > 0.015))[0]
    #iOut3 = np.where(((fr3.mures < md3-3*std3) | (fr3.mures > md3+3*std3)) & (fr3.zCMB > 0.015))[0]
    iOut2=np.where(fr2.FITCHI2/fr2.NDOF> 10)
    iOut3=np.where(fr3.FITCHI2/fr3.NDOF> 100)
#    import pdb;pdb.set_trace(n)
    for j,i in enumerate(fr3.CID[iOut3]):
        if i in fr2.CID[iOut2] or i not in fr2.CID: continue
        with PdfPages(f'{i}.pdf') as pdf:
            plotter_choice,base_name,CID=run_snana_plot(
                idsurvey_map[fr3.IDSURVEY[iOut3][j]][2],i,idsurvey_map[fr3.IDSURVEY[iOut3][j]][3],False,None)
            figs,fits=plot_lc([i],base_name,False,plotter_choice,None,None,None,None,'zCMB',title=f'{i}, SALT3')
            for f in figs:
                pdf.savefig(f)
            #os.system(f'rm {base_name}*')
            plotter_choice,base_name,CID=run_snana_plot(
                idsurvey_map[fr3.IDSURVEY[iOut3][j]][2],i,idsurvey_map[fr3.IDSURVEY[iOut3][j]][4],False,None)
            figs,fits=plot_lc([i],base_name,False,plotter_choice,None,None,None,None,'zCMB',title=f'{i}, SALT2')
            for f in figs:
                pdf.savefig(f)
            #os.system(f'rm {base_name}*')
                
        import pdb; pdb.set_trace()

def plot_91t():

    idsurvey_map = {1:('fit_output/SDSS.FITRES.TEXT','fit_output/SDSS_SALT2.FITRES.TEXT','SDSS',
                       'pipeline/SDSS.NML','pipeline/SDSS_SALT2.NML'),
                    10:('fit_output/DES-SN3YR_DES.FITRES.TEXT','fit_output/DES-SN3YR_DES_SALT2.FITRES.TEXT','DES-SN3YR_DES',
                        'pipeline/DES-SN3YR_DES.NML','pipeline/DES-SN3YR_DES_SALT2.NML'),
                    15:('fit_output/Pantheon_PS1MD_TEXT.FITRES.TEXT','fit_output/Pantheon_PS1MD_TEXT_SALT2.FITRES.TEXT',
                        'Pantheon_PS1MD_TEXT','pipeline/Pantheon_PS1MD_TEXT.NML','pipeline/Pantheon_PS1MD_TEXT_SALT2.NML'),
                    150:('fit_output/Foundation_DR1.FITRES.TEXT','fit_output/Foundation_DR1_SALT2.FITRES.TEXT',
                         'Foundation_DR1','pipeline/Foundation_DR1.NML','pipeline/Foundation_DR1.NML'),
                    191:('fit_output/Hamuy1996_LC.FITRES.TEXT','fit_output/Hamuy1996_LC_SALT2.FITRES.TEXT',
                         'Hamuy1996_LC','pipeline/Hamuy1996_LC.NML','pipeline/Hamuy1996_LC_SALT2.NML'),
                    192:('fit_output/Hicken2009_LC.FITRES.TEXT','fit_output/Hicken2009_LC_SALT2.FITRES.TEXT',
                         'Hicken2009_LC','pipeline/Hicken2009_LC.NML','pipeline/Hicken2009_LC_SALT2.NML'),
                    193:('fit_output/Jha2006_LC.FITRES.TEXT','fit_output/Jha2006_LC_SALT2.FITRES.TEXT',
                         'Jha2006_LC','pipeline/Jha2006_LC.NML','pipeline/Jha2006_LC_SALT2.NML'),
                    194:('fit_output/OTHER_LOWZ_LC.FITRES.TEXT','fit_output/OTHER_LOWZ_LC_SALT2.FITRES.TEXT',
                         'OTHER_LOWZ_LC','pipeline/OTHER_LOWZ_LC.NML','pipeline/OTHER_LOWZ_LC_SALT2.NML'),
                    195:('fit_output/Riess1999_LC.FITRES.TEXT','fit_output/Riess1999_LC_SALT2.FITRES.TEXT',
                         'Riess1999_LC','pipeline/Riess1999_LC.NML','pipeline/Riess1999_LC_SALT2.NML'),
                    196:('fit_output/SNLS3_LC.FITRES.TEXT','fit_output/SNLS3_LC_SALT2.FITRES.TEXT',
                         'SNLS3_LC','pipeline/SNLS3_LC.NML','pipeline/SNLS3_LC_SALT2.NML')}
    
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
    fr2 = txtobj('fit_output/JLA_TRAINING_SALT2.FITRES',fitresheader=True)
    fr3 = txtobj('fit_output/JLA_TRAINING_SALT3.FITRES',fitresheader=True)
    
    # hubble diagram w/ bins
    #fr2 = getmu.getmu(fr2,salt2alpha=salt2alpha,salt2beta=salt2beta,sigint=0.0)
    #fr3 = getmu.getmu(fr3,salt2alpha=salt3alpha,salt2beta=salt3beta,sigint=0.0)
    #fr2 = getmu.mkcuts(fr2,salt2alpha=salt2alpha,salt2beta=salt2beta)
    #fr3 = getmu.mkcuts(fr3,salt2alpha=salt3alpha,salt2beta=salt3beta)

    iGood2 = np.array([],dtype=int)
    for j,i in enumerate(fr2.CID):
        if i in fr3.CID:
            iGood2 = np.append(iGood2,j)
    for k in fr2.__dict__.keys():
        fr2.__dict__[k] = fr2.__dict__[k][iGood2]
    iGood3 = np.array([],dtype=int)
    for j,i in enumerate(fr3.CID):
        if i in fr2.CID:
            iGood3 = np.append(iGood3,j)
    for k in fr3.__dict__.keys():
        fr3.__dict__[k] = fr3.__dict__[k][iGood3]

    iPlot = fr3.CID == 'sn1991t'
    iPlot2 = fr2.CID == 'sn1991t'
    for j,i in enumerate(fr3.CID[iPlot]):
        if i not in fr2.CID[iPlot2] or i not in fr2.CID: continue

        with PdfPages(f'{i}.pdf') as pdf:
            plotter_choice,base_name,CID=run_snana_plot(
                idsurvey_map[fr3.IDSURVEY[iPlot][j]][2],i,idsurvey_map[fr3.IDSURVEY[iPlot][j]][3],False,None)
            figs,fits=plot_lc([i],base_name,False,plotter_choice,None,None,None,None,'zCMB',title=f'{i}, SALT3')
            for f in figs:
                pdf.savefig(f)
            #os.system(f'rm {base_name}*')
            plotter_choice,base_name,CID=run_snana_plot(
                idsurvey_map[fr3.IDSURVEY[iPlot][j]][2],i,idsurvey_map[fr3.IDSURVEY[iPlot][j]][4],False,None)
            figs,fits=plot_lc([i],base_name,False,plotter_choice,None,None,None,None,'zCMB',title=f'{i}, SALT2')
            for f in figs:
                pdf.savefig(f)
            #os.system(f'rm {base_name}*')
                
            import pdb; pdb.set_trace()

        

def plot_lc(cid,base_name,noGrid,plotter_choice,tmin,tmax,filter_list,plot_all,zname,title=None):
    if tmin is None:
        tmin=-np.inf
    if tmax is None:
        tmax=np.inf
        
    sn,fits,peak,minmaxtime=read_lc(cid,base_name,plotter_choice,tmin,tmax,filter_list)
    z=read_snana(base_name,cid,zname)
    if len(sn['time'])==0:
        return [[],[]]
    rows=int(math.ceil(len(np.unique(sn['filter']))))
    figs=[]
    all_bands=np.append([x for x in __band_order__ if x in np.unique(sn['filter'])],
                        [x for x in np.unique(sn['filter']) if x not in __band_order__])
    
    j=0
    minx=np.min(sn['time'])
    maxx=np.max(sn['time'])
    if minx<0:
        minx=min(minx*1.1,minx-5)
    else:
        minx=min(minx*.9,minx-5)
    if maxx<0:
        maxx=max(maxx*.9,maxx+5)
    else:
        maxx=max(maxx*1.1,maxx+5)
    
    xlims=(minx,maxx)
    
    sharedx=True
    for nfig in range(int(math.ceil(rows/4.))): 
        fig,ax=plt.subplots(nrows=min(len(all_bands),4),ncols=1,figsize=(8,8),sharex=sharedx)
        if title is None: ax[0].set_title('SNID=%s'%cid[0],fontsize=16)
        else: ax[0].set_title(title,fontsize=16)
        fit_print=False
        for i in range(min(len(all_bands[j:]),4)):
            temp_sn={k:sn[k][np.where(sn['filter']==all_bands[j])[0]] for k in sn.keys()}
            chi2=np.mean(temp_sn['chi2'])
            if chi2>0:
                lab=r'%s: $\chi^2_{red}$=%.1f'%(all_bands[j],np.mean(temp_sn['chi2']))
                leg_size=10
            else:
                lab=all_bands[j]
                leg_size=12
            
            ax[i].errorbar(temp_sn['time'],temp_sn['flux'],yerr=temp_sn['fluxerr'],
                          fmt='.',markersize=8,color='k',
                          label=lab)
            
            if len(fits)>0:
                if not plot_all:
                    
                    fit_time=np.arange(np.min(temp_sn['time'])-5,np.max(temp_sn['time'])+5,1)

                    
                else:
                    fit_time=np.arange(fits['trange'][all_bands[j]][0],fits['trange'][all_bands[j]][1],1)
                
                fit_time=fit_time[np.where(np.logical_and(fit_time>=minx,fit_time<=maxx))[0]]
                ax[i].plot(fit_time,fits[all_bands[j]](fit_time),color='r',label='Best Fit',linewidth=3)

                if not fit_print:
                    to_print=[]
                    for fit_key in fits['params'].keys():
                        if fit_key =='x0':
                            to_print.append(['$%s: %.2e'%(fit_key,fits['params'][fit_key][0]),'%.2e$\n'%fits['params'][fit_key][1]])
                        elif fit_key in ['x1','c']:
                            to_print.append(['$%s: %.2f'%(fit_key,fits['params'][fit_key][0]),'%.2f$\n'%fits['params'][fit_key][1]])
                        elif fit_key=='NDOF':
                            to_print.append('CHI2/NDOF: %.2f/%.2f\n'%(fits['params']['FITCHI2'],fits['params'][fit_key]))
                        else:
                            pass
                    if z is not None:
                        if np.any([x!='0' for x in str(z[1])[str(z[1]).find('.')+1:str(z[1]).find('.')+4]]):
                            print(str(z[1]))
                            to_print.append('z: %.2f'%float(z[0])+r'$\pm$'+'%.3f'%float(z[1]))
                        else:
                            to_print.append('z: %.2f'%float(z[0]))

                    ax[i].annotate(''.join([x[0]+r'\pm'+x[1] if isinstance(x,list) else x for x in to_print]),xy=(.02,.55),xycoords='axes fraction',fontsize=6)
                fit_print=True

            ax[i].legend(fontsize=leg_size)
            ax[i].set_ylabel('Flux',fontsize=16)
            
            if len(fits)>0:
                try:
                    maxFlux=max(np.max(temp_sn['flux']),np.max(fits[all_bands[j]](fit_time)))
                except:
                    maxFlux=np.max(temp_sn['flux']) 
            else:
                maxFlux=np.max(temp_sn['flux'])
            
            ax[i].set_ylim((-.1*np.max(temp_sn['flux']),1.1*maxFlux))
            if not noGrid:
                ax[i].grid()
            j+=1
            #i+=1
        for k in range(i+1,min(len(all_bands),4)):
            fig.delaxes(ax[k])
        ax[i].tick_params(axis='x',labelbottom=True,bottom=True)
        ax[i].set_xlabel('MJD-%.2f'%peak,fontsize=16)
        ax[i].set_xlim(xlims)
        figs.append(fig)
        plt.close()

    return(figs,fits)
        
def run_snana_plot(genversion,cid_list,nml,isdist,private):
    plotter='normal'
    if nml is not None:
        if os.path.splitext(nml)[1].upper()!='.NML':
            nml=os.path.splitext(nml)[0]+'.NML'
        for filename in os.listdir(os.path.split(nml)[0]):
        	if os.path.split(nml)[1] in filename:
        		nml=os.path.join(os.path.split(nml)[0],filename)
        	
        with open(nml,'r') as f:
            p=f.readlines()
    
        #for line in p:
        #    if 'FITMODEL_NAME' in line:
        #        if 'SALT2' in line:
        plotter='salt2'
                
        
    rand=str(np.random.randint(10000,100000))
    if private is not None:
        private_path=" PRIVATE_DATA_PATH %s"%private
    else:
        private_path=""
    genversion+=private_path
    
    if nml is not None:
        if cid_list is not None:
            cmd="snlc_fit.exe "+nml+" VERSION_PHOTOMETRY "+genversion+\
                " SNCCID_LIST "+cid_list+\
                " CUTWIN_CID 0 0 SNTABLE_LIST 'FITRES(text:key) SNANA(text:key) LCPLOT(text:key) SPECPLOT(text:key)' TEXTFILE_PREFIX 'OUT_TEMP_"+rand+\
                "' > OUT_TEMP_"+rand+".LOG"
        elif isdist:
            cmd="snlc_fit.exe "+nml+" VERSION_PHOTOMETRY "+genversion+" SNTABLE_LIST "+\
                "'FITRES(text:key) SNANA(text:key) LCPLOT(text:key) SPECPLOT(text:key)' TEXTFILE_PREFIX OUT_TEMP_"+rand+" > OUT_TEMP_"+rand+".LOG"
        else:
            cmd="snlc_fit.exe "+nml+" VERSION_PHOTOMETRY "+genversion+" MXEVT_PROCESS 200 SNTABLE_LIST "+\
                "'FITRES(text:key) SNANA(text:key) LCPLOT(text:key) SPECPLOT(text:key)' TEXTFILE_PREFIX OUT_TEMP_"+rand+" > OUT_TEMP_"+rand+".LOG"
    else:
        cmd="snana.exe NOFILE VERSION_PHOTOMETRY "+genversion+\
            " SNCCID_LIST "+cid_list+\
            " CUTWIN_CID 0 0 SNTABLE_LIST 'SNANA(text:key) LCPLOT(text:key) SPECPLOT(text:key)' TEXTFILE_PREFIX 'OUT_TEMP_"+rand+\
            "' > OUT_TEMP_"+rand+".LOG"

    print(cmd)
    os.system(cmd)
    with open('OUT_TEMP_'+rand+'.LOG','rb+') as f:
        content=f.read()
        f.seek(0,0)
        f.write(b'SNANA COMMAND:\n\n'+bytearray(textwrap.fill(cmd,80),encoding='utf-8')+b'\n'+content)
    if len(glob.glob('OUT_TEMP_'+rand+'*.TEXT'))==0:
        print("There was an error in retrieving your SN")
        sys.exit()

    if cid_list is None:
        with open("OUT_TEMP_"+rand+".FITRES.TEXT",'rb') as f:
            all_dat=f.readlines()
        all_cids=[]
        for line in all_dat:
            temp=line.split()
            if len(temp)>0 and b'VARNAMES:' in temp:
                varnames=[str(x.decode('utf-8')) for x in temp]
            elif len(temp)>0 and b"SN:" in temp:
                all_cids.append(str(temp[varnames.index('CID')].decode('utf-8')))
        all_cids=','.join(all_cids)
    else:
        all_cids=cid_list
    return(plotter,'OUT_TEMP_'+rand,all_cids)

def read_lc(cid,base_name,plotter_choice,tmin,tmax,filter_list):
    names=['time','flux','fluxerr','filter','chi2']
    peak=None
    sn={k:[] for k in names} 
    fit={k:[] for k in ['time','flux','filter']}
    with open(base_name+".LCPLOT.TEXT",'rb') as f:
        dat=f.readlines()
    fitted=False
    mintime=np.inf
    maxtime=-np.inf
    for line in dat:
        temp=line.split()
        if len(temp)>0 and b'VARNAMES:' in temp:
            varnames=[str(x.decode('utf-8')) for x in temp]
        elif len(temp)>0 and b'OBS:' in temp and str(temp[varnames.index('CID')].decode('utf-8')) in cid:
            if float(temp[varnames.index('Tobs')])<tmin or float(temp[varnames.index('Tobs')])>tmax:
                continue
            
            if filter_list is not None and str(temp[varnames.index('BAND')].decode('utf-8')) not in filter_list:
                continue
            if int(temp[varnames.index('DATAFLAG')])==1:
                if peak is None:
                    peak=float(temp[varnames.index('MJD')])-float(temp[varnames.index('Tobs')])
                t=float(temp[varnames.index('Tobs')])
                if t>maxtime:
                    maxtime=t
                if t<mintime:
                    mintime=t
                sn['time'].append(t)
                sn['flux'].append(float(temp[varnames.index('FLUXCAL')]))
                sn['fluxerr'].append(float(temp[varnames.index('FLUXCAL_ERR')]))
                sn['filter'].append(str(temp[varnames.index('BAND')].decode('utf-8')))
                sn['chi2'].append(float(temp[varnames.index('CHI2')]))
            elif int(temp[varnames.index('DATAFLAG')])==0:
                fitted=True
                fit['time'].append(float(temp[varnames.index('Tobs')]))
                fit['flux'].append(float(temp[varnames.index('FLUXCAL')]))
                fit['filter'].append(str(temp[varnames.index('BAND')].decode('utf-8')))
    if fitted and plotter_choice=='salt2':
        with open(base_name+".FITRES.TEXT",'rb') as f:
            dat=f.readlines()
        for line in dat:
            temp=line.split()
            if len(temp)>0 and b'VARNAMES:' in temp:
                varnames=[str(x.decode('utf-8')) for x in temp]
            elif len(temp)>0 and b'SN:' in temp and str(temp[varnames.index('CID')].decode('utf-8')) in cid: 
                fit['params']={p:(float(temp[varnames.index(p)]),float(temp[varnames.index(p+'ERR')])) if p in ['x0','x1','c'] else float(temp[varnames.index(p)]) for p in ['x0','x1','c','NDOF','FITCHI2']}
                break
    
    sn={k:np.array(sn[k]) for k in sn.keys()}
    fit={k:np.array(fit[k]) if k !='params' else fit['params'] for k in fit.keys()}
    if len(fit['filter'])>0:
        fits={k:interp1d(fit['time'][fit['filter']==k],
                     fit['flux'][fit['filter']==k]) for k in np.unique(fit['filter'])}
        if 'params' in fit.keys():
            fits['params']=fit['params']
        else:
            fits['params']={}
        fits['trange']={k:[np.min(fit['time'][fit['filter']==k]),np.max(fit['time'][fit['filter']==k])] for k in np.unique(fit['filter'])}
    else:
        fits=[]
    return(sn,fits,peak,(mintime,maxtime))

def read_snana(snana_filename,cid,param):
    if isinstance(cid,(list,tuple,np.ndarray)) and len(cid)==1:
        cid=cid[0]

    with open(snana_filename+'.SNANA.TEXT','r') as f:
        dat=f.readlines()
    for line in dat:
        temp=line.split()
        if len(temp)>0 and 'VARNAMES:' in temp:
            varnames=temp
        if len(temp)>0 and 'SN:' in temp and str(cid) in temp:
            try:
                ind=varnames.index(param)
            except:
                print('Either format of SNANA.TEXT file is wrong and no varnames, or %s not in file'%param)
            return(temp[ind],temp[ind+1])

    print('%s not found in SNANA.TEXT file.'%(str(cid)))
    return

def salt2to3_disp():
    from scipy.interpolate import splprep,splev,BSpline,griddata,bisplev,bisplrep,interp1d,interp2d

    for salt2file,salt3file in zip([os.path.expandvars('$SNDATA_ROOT/models/SALT2/SALT2.JLA-B14/salt2_lc_relative_variance_0.dat'),
    								os.path.expandvars('$SNDATA_ROOT/models/SALT2/SALT2.JLA-B14/salt2_lc_relative_variance_0.dat'),
                                    os.path.expandvars('$SNDATA_ROOT/models/SALT2/SALT2.JLA-B14/salt2_lc_relative_variance_1.dat'),
                                    os.path.expandvars('$SNDATA_ROOT/models/SALT2/SALT2.JLA-B14/salt2_lc_relative_covariance_01.dat')],
                                   ['SALT3.K20/salt3_lc_dispersion_scaling.dat'
                                   	'SALT3.K20/salt3_lc_variance_0.dat',
                                    'SALT3.K20/salt3_lc_variance_1.dat',
                                    'SALT3.K20/salt3_lc_covariance_01.dat']):
        phase2,wave2,var2 = np.loadtxt(salt2file,
                                       unpack=True)
        var2 = var2.reshape([len(np.unique(phase2)),len(np.unique(wave2))])
        phase2 = np.unique(phase2)
        wave2 = np.unique(wave2)

        phase3,wave3,var3 = np.loadtxt(salt3file,
                                       unpack=True)
        var3 = var3.reshape([len(np.unique(phase3)),len(np.unique(wave3))])
        phase3 = np.unique(phase3)
        wave3 = np.unique(wave3)

        with open(salt3file,'w') as fout:
            for i,p3 in enumerate(phase2):
                var3_p = np.interp(wave3,wave2,var2[i,:])
                for j,w3 in enumerate(wave3):
                    print(f'{p3:.1f} {w3:.2f} {var3_p[j]:8.15e}',file=fout)
    

if __name__ == "__main__":
    #concatfitres()
    #run_salt2mu()
    concatfitres_valid()
    run_salt2mu_valid()
    #main()
    #plot_outliers()
    #plot_91t()
    #salt2to3_disp()
