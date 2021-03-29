#!/usr/bin/env python
import numpy as np
import pylab as plt
plt.rcParams['figure.figsize'] = (6,12)
plt.ion()
from scipy.interpolate import interp2d
from matplotlib.gridspec import GridSpec
from txtobj import txtobj
import getmu
from astropy.stats import sigma_clipped_stats
from scipy.stats import binned_statistic
from matplotlib.ticker import NullFormatter

def errfnc(x):
    return(np.std(x)/np.sqrt(len(x)))

def main(xlimits=[2500,9200]):
    plt.subplots_adjust(wspace=0,top=0.9,bottom=0.1,hspace=0.4)
    fig = plt.gcf()
    gs = GridSpec(3, 5, figure=fig)
    #gs.update(wspace=0.0, hspace=0.0,bottom=0.2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax1.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
    ax2.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)


    #ax1 = plt.subplot(311)
    #ax2 = plt.subplot(312)
    #ax3 = plt.subplot(313)

    for salt3dir,color,label in zip(['output','output_halfspec'],['b','r'],['SALT3','SALT3 (no spectral timeseries)']):
        
        salt3m0phase,salt3m0wave,salt3m0flux = \
            np.loadtxt('%s/salt3_template_0.dat'%salt3dir,unpack=True)
        salt3m1phase,salt3m1wave,salt3m1flux = \
            np.loadtxt('%s/salt3_template_1.dat'%salt3dir,unpack=True)
        salt3m0errphase,salt3m0errwave,salt3m0fluxerr = \
            np.loadtxt('%s/salt3_lc_variance_0.dat'%salt3dir,unpack=True)
        salt3m1errphase,salt3m1errwave,salt3m1fluxerr = \
            np.loadtxt('%s/salt3_lc_variance_1.dat'%salt3dir,unpack=True)
        salt3m0phase = np.unique(salt3m0phase)
        salt3m0wave = np.unique(salt3m0wave)
        salt3m1phase = np.unique(salt3m1phase)
        salt3m1wave = np.unique(salt3m1wave)
        salt3m0flux = salt3m0flux.reshape([len(np.unique(salt3m0phase)),len(np.unique(salt3m0wave))])
        salt3m0fluxerr = salt3m0fluxerr.reshape([len(np.unique(salt3m0phase)),len(np.unique(salt3m0wave))])
        salt3m1flux = salt3m1flux.reshape([len(np.unique(salt3m1phase)),len(np.unique(salt3m1wave))])
        salt3m1fluxerr = salt3m1fluxerr.reshape([len(np.unique(salt3m1phase)),len(np.unique(salt3m1wave))])

        salt3m0errphase = np.unique(salt3m0errphase)
        salt3m0errwave = np.unique(salt3m0errwave)
        salt3m1errphase = np.unique(salt3m1errphase)
        salt3m1errwave = np.unique(salt3m1errwave)
        
        for plotphase,i,plotphasestr in zip([-5,0,10],range(3),['-5','+0','+10']):

            spacing = 0.5
            int_salt3m0 = interp2d(salt3m0wave,salt3m0phase,salt3m0flux)
            int_salt3m0err = interp2d(salt3m0errwave,salt3m0errphase,salt3m0fluxerr)
            salt3m0flux_0 = int_salt3m0(salt3m0wave,plotphase)
            salt3m0fluxerr_0 = int_salt3m0err(salt3m0wave,plotphase)

            if i == 0: ax1.plot(salt3m0wave,salt3m0flux_0+spacing*i,color=color,label=label)
            else: ax1.plot(salt3m0wave,salt3m0flux_0+spacing*i,color=color)
            ax1.fill_between(salt3m0wave,
                             salt3m0flux_0-np.sqrt(salt3m0fluxerr_0)+spacing*i,
                             salt3m0flux_0+np.sqrt(salt3m0fluxerr_0)+spacing*i,
                             color=color,alpha=0.5)
            ax1.set_xlim(xlimits)
            ax1.set_ylim([0,1.35])
            
            ax1.text(xlimits[1]-100,spacing*(i+0.2),'%s'%plotphasestr,ha='right')


            spacing = 0.15
            int_salt3m1 = interp2d(salt3m1wave,salt3m1phase,salt3m1flux)
            int_salt3m1err = interp2d(salt3m1errwave,salt3m1errphase,salt3m1fluxerr)
            salt3m1flux_0 = int_salt3m1(salt3m1wave,plotphase)
            salt3m1fluxerr_0 = int_salt3m1err(salt3m1wave,plotphase)

            ax2.plot(salt3m1wave,salt3m1flux_0+spacing*i,color=color)
            ax2.fill_between(salt3m1wave,
                             salt3m1flux_0-np.sqrt(salt3m1fluxerr_0)+spacing*i,
                             salt3m1flux_0+np.sqrt(salt3m1fluxerr_0)+spacing*i,
                             color=color,alpha=0.5)
            ax2.set_xlim(xlimits)
            ax2.set_ylim([-0.05,0.39])
            
            ax2.text(xlimits[1]-100,spacing*(i+0.2),'%s'%plotphasestr,ha='right')

        ax1.legend(loc='upper right',bbox_to_anchor=(0.82,1.4))

    ax3 = fig.add_subplot(gs[2, 0:4])
    ax4 = fig.add_subplot(gs[2, 4])
    ax3.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
    ax4.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)


    med_lowz,med_highz = [],[]
    for filename,salt3text,color in zip(['SALT2mu_SALT3.fitres','SALT2mu_SALT3_NOSPEC.fitres'],
                                        ['SALT3','SALT3 (no spectral timeseries)'],['b','r']):
        
        with open(filename) as fin:
            for line in fin:
                if line.startswith('#  alpha0'): salt3alpha,salt3alphaerr = float(line.split()[3]),float(line.split()[5].replace('\n',''))
                if line.startswith('#  beta0'): salt3beta,salt3betaerr = float(line.split()[3]),float(line.split()[5].replace('\n',''))
                if line.startswith('#  sigint'): salt3sigint = float(line.split()[3])
        
        # avg chi2/SN
        fr3 = txtobj(filename,fitresheader=True)

        # hubble diagram w/ bins
        fr3 = getmu.getmu(fr3,salt2alpha=salt3alpha,salt2beta=salt3beta,sigint=0.0)
        fr3 = getmu.mkcuts(fr3,salt2alpha=salt3alpha,salt2beta=salt3beta)

        mn3,md3,std3 = sigma_clipped_stats(fr3.mures)
        iOut3 = len(np.where(((fr3.mures < md3-3*std3) | (fr3.mures > md3+3*std3)) & (fr3.zCMB > 0.015))[0])/float(len(fr3.CID))

        med_chi3 = np.median(fr3.FITCHI2/fr3.NDOF)

        zbins = np.logspace(np.log10(0.01),np.log10(1.0),20)
        salt3mubins = binned_statistic(fr3.zCMB,fr3.mures-np.median(fr3.mures[fr3.zCMB > 0.01]),
                                       bins=zbins,statistic='mean').statistic
        salt3muerrbins = binned_statistic(fr3.zCMB,fr3.mures-np.median(fr3.mures[fr3.zCMB > 0.01]),
                                          bins=zbins,statistic=errfnc).statistic

        ax3.axhline(0,lw=2,color='k')
        ax3.errorbar(fr3.zCMB,fr3.mures-np.median(fr3.mures[fr3.zCMB > 0.01]),yerr=fr3.muerr,fmt='D',color=color,alpha=0.1)
        ax3.errorbar((zbins[1:]+zbins[:-1])/2.,salt3mubins,yerr=salt3muerrbins,fmt='D-',color=color)
        ax3.set_xscale('log')
        ax3.xaxis.set_major_formatter(NullFormatter())
        ax3.xaxis.set_minor_formatter(NullFormatter())
        ax3.xaxis.set_ticks([0.01,0.02,0.05,0.1,0.3,0.7])
        ax3.xaxis.set_ticklabels(['0.01','0.02','0.05','0.1','0.3','0.7'])

        muresbins = np.linspace(-1,1,40)
        ax4.hist(fr3.mures[fr3.zCMB > 0.01]-np.median(fr3.mures[fr3.zCMB > 0.01]),bins=muresbins,
                 orientation='horizontal',color=color,alpha=0.5)

        med_lowz += [np.median(fr3.mures[(fr3.zCMB > 0.01) & (fr3.zCMB < 0.15)])]
        med_highz += [np.median(fr3.mures[(fr3.zCMB > 0.15)])]
        
#        ax3.text(0.3,0.97,fr"""{salt3text}
#    $\alpha = {salt3alpha:.3f}\pm{salt3alphaerr:.3f}$
#    $\beta = {salt3beta:.2f}\pm{salt3betaerr:.2f}$
#    $\sigma_{{int}}$ = {salt3sigint:.3f}
#    $med. \chi^2_{{\nu}}$ = {med_chi3:.2f}""",
#                 ha='left',va='top',color=color,transform=ax1.transAxes,bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.5})

        #ax4.text(0.97,0.03,f"""SALT3 RMS = {np.std(fr3.mures[fr3.zCMB > 0.015]):.3f}
    #{iOut3*len(fr3.CID):.0f} outliers ({iOut3*100:.1f}%)""",transform=ax2.transAxes,ha='right',va='bottom',color=colordict['SALT3'])
        ax4.yaxis.tick_right()
        ax3.set_ylim([-0.5,0.5])
        ax4.set_ylim([-0.5,0.5])
        ax3.set_xlabel('$z_{CMB}$',fontsize=15)
        ax3.set_ylabel('$\mu-\mu_{\Lambda CDM}$',fontsize=15)
        ax4.xaxis.set_ticks([])
    ax3.text(0.03,0.85,'$\Delta\mu(z > 0.15) - \Delta\mu(z < 0.15)) = %.3f$'%((med_highz[0]-med_lowz[0]) - (med_highz[1]-med_lowz[1])),transform=ax3.transAxes,
             bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.5})
    ax4.yaxis.tick_right()
    ax1.set_xlabel('Wavelength ($\AA$)')
    ax2.set_xlabel('Wavelength ($\AA$)')
    ax1.set_ylabel('$M_0$ flux')
    ax2.set_ylabel('$M_1$ flux')
    
    import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    main()
