#!/usr/bin/env python
# D. Jones - 12/17/21

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import os
from matplotlib.gridspec import GridSpec

snialines = [
[3850,'CaII H&K'],
[4000,'SiII'],
[4300,'MgII'],
[4800,'FeII'],
[5400,'SiII-W'],
[5800,'SiII'],
[6150,'SiII 6150'],
[8100,'CaII IR'],
]

def main(plotphase=0,modelname='SALT3.',hostmodelname='output'):

    plt.clf()
    fig = plt.figure()
    gs = GridSpec(4, 1, figure=fig, hspace=0, wspace=0)
    ax = fig.add_subplot(gs[0:3,0])
    axresid = fig.add_subplot(gs[3,0])
    
    highmassphase,highmasswave,highmassflux = \
        np.loadtxt(f'{modelname}HighMass/salt3_template_0.dat',unpack=True)
    lowmassphase,lowmasswave,lowmassflux = \
        np.loadtxt(f'{modelname}LowMass/salt3_template_0.dat',unpack=True)
    highmassphase,highmasswave,highmassfluxvar = \
        np.loadtxt(f'{modelname}HighMass/salt3_lc_variance_0.dat',unpack=True)
    lowmassphase,lowmasswave,lowmassfluxvar = \
        np.loadtxt(f'{modelname}LowMass/salt3_lc_variance_0.dat',unpack=True)
    massphase,masswave,massflux = \
        np.loadtxt(f'{hostmodelname}/salt3_template_host.dat',unpack=True)
    masserrphase,masserrwave,masserrflux = \
        np.loadtxt(f'{hostmodelname}/salt3_lc_variance_host.dat',unpack=True)

    # mass flux - need to subtract off the B band flux
    #from saltshaker.training.init_hsiao import synphotBflux
    #Bfilt = '../../saltshaker/initfiles/Bessell90_B.dat'
    #refWave,refFlux=np.loadtxt('../../saltshaker/initfiles/flatnu.dat',unpack=True)
    #flux_adj = synphotBflux(masswave[massphase==0],massflux[massphase==0],0,0,Bfilt)
    massflux += -0.0189705 #flux_adj
    #import pdb; pdb.set_trace()
    
    highmassfluxerr = np.sqrt(highmassfluxvar)
    lowmassfluxerr = np.sqrt(lowmassfluxvar)
    
    k21phase,k21wave,k21flux = \
        np.loadtxt(os.path.expandvars('/Users/David/Dropbox/research/SALTShaker/examples/SALT3TRAIN_K21_PUBLIC/SALT3.Fragilistic/salt3_template_0.dat'),unpack=True)
        #np.loadtxt(os.path.expandvars('$SNDATA_ROOT/models/SALT3/SALT3.K21/salt3_template_0.dat'),unpack=True)
    k21m1phase,k21m1wave,k21m1flux = \
        np.loadtxt(os.path.expandvars('/Users/David/Dropbox/research/SALTShaker/examples/SALT3TRAIN_K21_PUBLIC/SALT3.Fragilistic/salt3_template_1.dat'),unpack=True)
        #np.loadtxt(os.path.expandvars('$SNDATA_ROOT/models/SALT3/SALT3.K21/salt3_template_1.dat'),unpack=True)

    
    highmasswave = highmasswave[highmassphase == plotphase]
    highmassflux = highmassflux[highmassphase == plotphase]
    highmassfluxerr = highmassfluxerr[highmassphase == plotphase]
    lowmasswave = lowmasswave[lowmassphase == plotphase]
    lowmassflux = lowmassflux[lowmassphase == plotphase]
    lowmassfluxerr = lowmassfluxerr[lowmassphase == plotphase]
    k21wave = k21wave[k21phase == plotphase]
    k21flux = k21flux[k21phase == plotphase]
    k21m1wave = k21m1wave[k21m1phase == plotphase]
    k21m1flux = k21m1flux[k21m1phase == plotphase]
    massflux = massflux[massphase == plotphase]
    masserrflux = np.sqrt(masserrflux[masserrphase == plotphase])
    #k21flux = k21flux + k21m1flux
    
    k21flux_norm = k21flux/(k21flux.max()-k21flux.min())
    lowmassflux_norm = lowmassflux/(lowmassflux.max()-lowmassflux.min())
    highmassflux_norm = highmassflux/(highmassflux.max()-highmassflux.min())
    
    ax.plot(k21wave,k21flux_norm+1,color='k',label='SALT3.K21')
    ax.plot(lowmasswave,lowmassflux_norm+1,label=r'log($M_{\ast}/M_{\odot}$) < 10',color='b')
    ax.fill_between(lowmasswave,lowmassflux_norm+1-highmassfluxerr,lowmassflux_norm+1+highmassfluxerr,color='b',alpha=0.5)
    ax.plot(k21wave,k21flux_norm+2,color='k')
    ax.plot(highmasswave,highmassflux_norm+2,label=r'log($M_{\ast}/M_{\odot}$) > 10',color='r')
    ax.fill_between(highmasswave,highmassflux_norm+2-highmassfluxerr,highmassflux_norm+2+highmassfluxerr,color='r',alpha=0.5)
    ax.legend(loc='lower center')

    axresid.plot(highmasswave,massflux,color='0.5')
    axresid.fill_between(highmasswave,massflux-masserrflux,massflux+masserrflux,color='0.5',alpha=0.5)
    axresid.axhline(0,color='k',lw=2)
    axresid.set_ylabel(r'$M_{\rm host}$')
    axresid.yaxis.set_ticklabels([])
    
    ax.xaxis.set_ticklabels([])
    ax.set_xlim([3000,7000])
    axresid.set_xlim([3000,7000])
    ax.set_ylim([0.4,3.5])
    ax.yaxis.set_ticklabels([])
    #import pdb; pdb.set_trace()
    ax.set_ylabel('flux + offset')
    axresid.set_ylim([-0.1,0.1])
    axresid.set_xlabel(r'wavelength (${\rm \AA}$)')
    #import pdb; pdb.set_trace()
    for s in snialines:
        ax.axvline(s[0],ls='--',color='0.6')
        axresid.axvline(s[0],ls='--',color='0.6')
        ax.text(s[0]-20,3.48,s[1],ha='right',va='top',rotation='90')
    plt.savefig('specplot.png',dpi=200)

def sequence(plotphase=00):
    plt.clf()
    fig = plt.figure()
    gs = GridSpec(3, 1, figure=fig, hspace=0, wspace=0)
    ax = fig.add_subplot(gs[0:3,0])
    #axresid = fig.add_subplot(gs[3,0])
    
    highmassphase,highmasswave,highmassflux = \
        np.loadtxt('SALT3.HighMass/salt3_template_0.dat',unpack=True)
    lowmassphase,lowmasswave,lowmassflux = \
        np.loadtxt('SALT3.LowMass/salt3_template_0.dat',unpack=True)
    highmassphase,highmasswave,highmassfluxvar = \
        np.loadtxt('SALT3.HighMass/salt3_lc_variance_0.dat',unpack=True)
    lowmassphase,lowmasswave,lowmassfluxvar = \
        np.loadtxt('SALT3.LowMass/salt3_lc_variance_0.dat',unpack=True)
    massphase,masswave,massflux = \
        np.loadtxt('output/salt3_template_host.dat',unpack=True)
    masserrphase,masserrwave,masserrflux = \
        np.loadtxt('output/salt3_lc_variance_host.dat',unpack=True)

    # mass flux - need to subtract off the B band flux
    #from saltshaker.training.init_hsiao import synphotBflux
    #Bfilt = '../../saltshaker/initfiles/Bessell90_B.dat'
    #refWave,refFlux=np.loadtxt('../../saltshaker/initfiles/flatnu.dat',unpack=True)
    #flux_adj = synphotBflux(masswave[massphase==0],massflux[massphase==0],0,0,Bfilt)
    massflux += -0.0189705 #flux_adj
    #import pdb; pdb.set_trace()
    
    highmassfluxerr = np.sqrt(highmassfluxvar)
    lowmassfluxerr = np.sqrt(lowmassfluxvar)
    
    k21phase,k21wave,k21flux = \
        np.loadtxt(os.path.expandvars('$SNDATA_ROOT/models/SALT3/SALT3.K21/salt3_template_0.dat'),unpack=True)
    k21m1phase,k21m1wave,k21m1flux = \
        np.loadtxt(os.path.expandvars('$SNDATA_ROOT/models/SALT3/SALT3.K21/salt3_template_1.dat'),unpack=True)

    for plotphase,offset in zip([-5,5,15,25,35],range(10)):
        highmasswave_single = highmasswave[highmassphase == plotphase]
        highmassflux_single = highmassflux[highmassphase == plotphase]
        highmassfluxerr_single = highmassfluxerr[highmassphase == plotphase]
        lowmasswave_single = lowmasswave[lowmassphase == plotphase]
        lowmassflux_single = lowmassflux[lowmassphase == plotphase]
        lowmassfluxerr_single = lowmassfluxerr[lowmassphase == plotphase]
        #k21flux = k21flux + k21m1flux

        k21flux_norm = k21flux/(k21flux.max()-k21flux.min())
        lowmassflux_norm = lowmassflux_single/(lowmassflux_single.max()-lowmassflux_single.min())
        highmassflux_norm = highmassflux_single/(highmassflux_single.max()-highmassflux_single.min())
        lowmassfluxerr_norm = lowmassfluxerr_single/(lowmassflux_single.max()-lowmassflux_single.min())
        highmassfluxerr_norm = highmassfluxerr_single/(highmassflux_single.max()-highmassflux_single.min())

        ax.plot(lowmasswave_single,lowmassflux_norm+offset,label=r'log($M_{\ast}/M_{\odot}$) < 10',color='b')
        ax.fill_between(lowmasswave_single,lowmassflux_norm+offset-highmassfluxerr_norm,lowmassflux_norm+offset+highmassfluxerr_norm,color='b',alpha=0.5)

        ax.plot(highmasswave_single,highmassflux_norm+offset,label=r'log($M_{\ast}/M_{\odot}$) > 10',color='r')
        ax.fill_between(highmasswave_single,highmassflux_norm+offset-highmassfluxerr_norm,highmassflux_norm+offset+highmassfluxerr_norm,color='r',alpha=0.5)
        if offset == 0: ax.legend(loc='lower center')
    
    #ax.xaxis.set_ticklabels([])
    ax.set_xlim([3000,7000])
    #axresid.set_xlim([3000,7000])
    ax.set_ylim([0,5])
    ax.yaxis.set_ticklabels([])
    #import pdb; pdb.set_trace()
    ax.set_ylabel('flux + offset')
    #axresid.set_ylim([-0.1,0.1])
    ax.set_xlabel(r'wavelength (${\rm \AA}$)')
    #import pdb; pdb.set_trace()
    for s in snialines:
        ax.axvline(s[0],ls='--',color='0.6')
        #axresid.axvline(s[0],ls='--',color='0.6')
        ax.text(s[0]-20,3.48,s[1],ha='right',va='top',rotation='90')
    plt.savefig('specplot_sequence.png',dpi=200)

def sequence_resid(plotphase=0):
    plt.rcParams['figure.figsize'] = (6,12)
    plt.clf()
    fig = plt.figure()
    gs = GridSpec(3, 1, figure=fig, hspace=0, wspace=0)
    ax = fig.add_subplot(gs[0:3,0])
    #axresid = fig.add_subplot(gs[3,0])
    
    highmassphase,highmasswave,highmassflux = \
        np.loadtxt('SALT3.HighMass/salt3_template_0.dat',unpack=True)
    lowmassphase,lowmasswave,lowmassflux = \
        np.loadtxt('SALT3.LowMass/salt3_template_0.dat',unpack=True)
    highmassphase,highmasswave,highmassfluxvar = \
        np.loadtxt('SALT3.HighMass/salt3_lc_variance_0.dat',unpack=True)
    lowmassphase,lowmasswave,lowmassfluxvar = \
        np.loadtxt('SALT3.LowMass/salt3_lc_variance_0.dat',unpack=True)
    massphase,masswave,massflux = \
        np.loadtxt('output/salt3_template_host.dat',unpack=True)
    masserrphase,masserrwave,masserrflux = \
        np.loadtxt('output/salt3_lc_variance_host.dat',unpack=True)

    # mass flux - need to subtract off the B band flux
    from saltshaker.training.init_hsiao import synphotBflux
    Bfilt = '../../saltshaker/initfiles/Bessell90_B.dat'
    #refWave,refFlux=np.loadtxt('../../saltshaker/initfiles/flatnu.dat',unpack=True)
    #flux_adj = synphotBflux(masswave[massphase==0],massflux[massphase==0],0,0,Bfilt)
    massflux += -0.0189705 #flux_adj
    #import pdb; pdb.set_trace()
    
    highmassfluxerr = np.sqrt(highmassfluxvar)
    lowmassfluxerr = np.sqrt(lowmassfluxvar)
    
    k21phase,k21wave,k21flux = \
        np.loadtxt(os.path.expandvars('$SNDATA_ROOT/models/SALT3/SALT3.K21/salt3_template_0.dat'),unpack=True)
    k21m1phase,k21m1wave,k21m1flux = \
        np.loadtxt(os.path.expandvars('$SNDATA_ROOT/models/SALT3/SALT3.K21/salt3_template_1.dat'),unpack=True)

    offset_list = np.arange(5)*0.15
    #offset_list = [0,0.15,0.25,0.3,0.35]
    for plotphase,offset in zip([-5,5,15,25,35],np.arange(5)*1.75):
        highmasswave_single = highmasswave[(highmassphase == plotphase) & (masswave > 3300)]
        highmassflux_single = highmassflux[(highmassphase == plotphase) & (masswave > 3300)]
        highmassfluxerr_single = highmassfluxerr[highmassphase == plotphase]
        lowmasswave_single = lowmasswave[lowmassphase == plotphase]
        lowmassflux_single = lowmassflux[lowmassphase == plotphase]
        lowmassfluxerr_single = lowmassfluxerr[lowmassphase == plotphase]
        massflux_single = massflux[(massphase == plotphase) & (masswave > 3300)]
        masserrflux_single = np.sqrt(masserrflux[(masserrphase == plotphase) & (masswave > 3300)])
        massflux_single -= np.median(massflux_single)
        
        #massflux_norm = massflux_single/((massflux_single+masserrflux_single).max()-(massflux_single-masserrflux_single).min())
        #masserrflux_norm = masserrflux_single/((massflux_single+masserrflux_single).max()-(massflux_single-masserrflux_single).min())
        #flux_adj = synphotBflux(highmasswave_single,highmassflux_single,0,0,Bfilt)
        
        massflux_norm = massflux_single/np.median(highmassflux_single) #(massflux_single.max()-massflux_single.min())
        masserrflux_norm = masserrflux_single/np.median(highmassflux_single) #(massflux_single.max()-massflux_single.min())
        masscompmed = np.median(massflux_norm)
        massflux_norm -= masscompmed
        masserrflux_norm -= masscompmed
        
        ax.plot(highmasswave_single,massflux_norm+offset,color='0.5')
        ax.fill_between(highmasswave_single,massflux_norm-masserrflux_norm+offset,massflux_norm+masserrflux_norm+offset,color='0.5',alpha=0.5)
        ax.plot(highmasswave_single,massflux_norm+offset,color='0.5')
        ax.fill_between(highmasswave_single,massflux_norm-masserrflux_norm+offset,massflux_norm+masserrflux_norm+offset,color='0.5',alpha=0.5)

        ax.axhline(offset,color='k')
        ax.set_ylabel(r'$M_{\rm host}$')
        ax.yaxis.set_ticklabels([])
        if plotphase < 0:
            ax.text(6900,offset+0.15,f"{plotphase} days",ha='right',va='bottom',bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.5})
        else:
            ax.text(6900,offset+0.15,f"+{plotphase} days",ha='right',va='bottom',bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.5})
        #import pdb; pdb.set_trace()
    #ax.xaxis.set_ticklabels([])
    ax.set_xlim([3300,7000])
    #axresid.set_xlim([3000,7000])
    ax.set_ylim([-0.5,4.5*1.75])
    ax.yaxis.set_ticklabels([])
    #import pdb; pdb.set_trace()
    ax.set_ylabel(r'SALT3 $M_{\rm host}$',fontsize=15)
    #axresid.set_ylim([-0.1,0.1])
    ax.set_xlabel(r'wavelength (${\rm \AA}$)',fontsize=15)
    #import pdb; pdb.set_trace()
    for s in snialines:
        ax.axvline(s[0],ls='--',color='0.6')
        #axresid.axvline(s[0],ls='--',color='0.6')
        ax.text(s[0]-20,4.48*1.75,s[1],ha='right',va='top',rotation='90')
    plt.savefig('specplot_sequence.png',dpi=200)

    
if __name__ == "__main__":
    main()
    #sequence()
    #sequence_resid()
