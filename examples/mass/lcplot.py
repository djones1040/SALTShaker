#!/usr/bin/env python
# D. Jones - 12/28/21

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.ion()
plt.rcParams['figure.figsize'] = (12,6)
import os
from SALT3HostSource import SALT3HostSource
import sncosmo
from sncosmo.models import SALT3Source
import copy

class lcplot:
    def __init__(self):
        pass

    def main(self,modelname='SALT3.fixedpars_'):
        # let's plot UBVRI light curves for high vs. low-mass
        # models, and also show residuals

        # first use GridSpec to set up the layout
        fig = plt.figure()
        gs = GridSpec(5, 5, figure=fig)
        gs.update(wspace=0.0, hspace=0)

        ax1 = fig.add_subplot(gs[0:3, 0])
        ax2 = fig.add_subplot(gs[0:3, 1])
        ax3 = fig.add_subplot(gs[0:3, 2])
        ax4 = fig.add_subplot(gs[0:3, 3])
        ax5 = fig.add_subplot(gs[0:3, 4])

        #ax6 = fig.add_subplot(gs[7:10, 0])
        #ax7 = fig.add_subplot(gs[7:10, 1])
        #ax8 = fig.add_subplot(gs[7:10, 2])
        #ax9 = fig.add_subplot(gs[7:10, 3])
        #ax10 = fig.add_subplot(gs[7:10, 4])

        # resid panels
        ax1resid = fig.add_subplot(gs[3, 0])
        ax2resid = fig.add_subplot(gs[3, 1])
        ax3resid = fig.add_subplot(gs[3, 2])
        ax4resid = fig.add_subplot(gs[3, 3])
        ax5resid = fig.add_subplot(gs[3, 4])

        #ax6resid = fig.add_subplot(gs[10, 0])
        #ax7resid = fig.add_subplot(gs[10, 1])
        #ax8resid = fig.add_subplot(gs[10, 2])
        #ax9resid = fig.add_subplot(gs[10, 3])
        #ax10resid = fig.add_subplot(gs[10, 4])

        # random stuff
        for ax,filt in zip([ax1,ax2,ax3,ax4,ax5],'UBVRI'):
            ax.set_title(f'${filt}$')
        #for ax,filt in zip([ax6,ax7,ax8,ax9,ax10],'UBVRI'):
        #    ax.set_title(f'${filt}$')
        for ax,filt in zip([ax1resid,ax2resid,ax3resid,ax4resid,ax5resid],'UBVRI'):
            ax.set_xlabel(f'Phase')
        #for ax,filt in zip([ax6resid,ax7resid,ax8resid,ax9resid,ax10resid],'UBVRI'):
        #    ax.set_xlabel(f'Phase')

        ax1.set_ylabel('Flux')
        #ax6.set_ylabel('Flux')
        #ax3.text(0.5,1.4,'Low-Mass Hosts',transform=ax3.transAxes,fontsize=15,ha='center',va='center')
        #ax8.text(0.5,1.4,'High-Mass Hosts',transform=ax8.transAxes,fontsize=15,ha='center',va='center')


        # get the different SED models
        #phase,wave,flux = np.loadtxt('output/salt3_template_0.dat')
        #mphase,mwave,mflux = np.loadtxt('output/salt3_template_host.dat')
        #k21phase,k21wave,k21flux = np.loadtxt(os.path.expandvars('$SNDATA_ROOT/models/SALT3/SALT3.K21/salt3_template_0.dat'))

        filtdict = {'SDSS':['sdss%s'%s for s in  'ugri']+['desz'],'Bessell':['bessell%s'%s +('x' if s=='u' else '')for s in  'ubvri']}
        filters=filtdict['Bessell']
    
        zpsys='AB'
    
        #salt3host = SALT3HostSource(
        #    modeldir='output',
        #    m0file='salt3_template_0.dat',
        #    m1file='salt3_template_1.dat',
        #    mhostfile='salt3_template_host.dat',
        #    clfile='salt3_color_correction.dat',
        #    cdfile='salt3_color_dispersion.dat',
        #    lcrv00file='salt3_lc_variance_0.dat',
        #    lcrv11file='salt3_lc_variance_1.dat',
        #    lcrv01file='salt3_lc_covariance_01.dat')
        salt3lowmasshost = SALT3Source(
            modeldir=f'{modelname}LowMass',
            m0file='salt3_template_0.dat',
            m1file='salt3_template_1.dat',
            clfile='salt3_color_correction.dat',
            cdfile='salt3_color_dispersion.dat',
            lcrv00file='salt3_lc_variance_0.dat',
            lcrv11file='salt3_lc_variance_1.dat',
            lcrv01file='salt3_lc_covariance_01.dat')
        salt3lowmasshostmodel = sncosmo.Model(salt3lowmasshost)
        salt3highmasshost = SALT3Source(
            modeldir=f'{modelname}HighMass',
            m0file='salt3_template_0.dat',
            m1file='salt3_template_1.dat',
            clfile='salt3_color_correction.dat',
            cdfile='salt3_color_dispersion.dat',
            lcrv00file='salt3_lc_variance_0.dat',
            lcrv11file='salt3_lc_variance_1.dat',
            lcrv01file='salt3_lc_covariance_01.dat')
        salt3highmasshostmodel = sncosmo.Model(salt3highmasshost)

        
        salt3k21source = SALT3Source(
            modeldir='/Users/David/Dropbox/research/SALTShaker/examples/SALT3TRAIN_K21_PUBLIC/SALT3.Fragilistic', #os.path.expandvars('$SNDATA_ROOT/models/SALT3/SALT3.K21'),
            m0file='salt3_template_0.dat',
            m1file='salt3_template_1.dat',
            clfile='salt3_color_correction.dat',
            cdfile='salt3_color_dispersion.dat',
            lcrv00file='salt3_lc_variance_0.dat',
            lcrv11file='salt3_lc_variance_1.dat',
            lcrv01file='salt3_lc_covariance_01.dat')
        salt3k21 = sncosmo.Model(salt3k21source)

        
        salt3lowmasshostmodel.set(z=0.0)
        salt3lowmasshostmodel.set(x0=1)
        salt3lowmasshostmodel.set(t0=0)
        salt3lowmasshostmodel.set(c=0)

        salt3highmasshostmodel.set(z=0.0)
        salt3highmasshostmodel.set(x0=1)
        salt3highmasshostmodel.set(t0=0)
        salt3highmasshostmodel.set(c=0)

        
        salt3k21.set(z=0.0)
        salt3k21.set(x0=1)
        salt3k21.set(t0=0)
        salt3k21.set(c=0)


        plotmjd = np.linspace(-10, 50,121)

        handles=[]
        for flt,ax,axresid in zip(filters,[ax1,ax2,ax3,ax4,ax5],[ax1resid,ax2resid,ax3resid,ax4resid,ax5resid]):
            # the direct check that the errors make sense
            #plotmjd = np.array([0])
            #salt3lmhostfluxerr = np.sqrt(salt3lowmasshostmodel._source._bandflux_rvar_single(sncosmo.get_bandpass(flt),plotmjd))*salt3lmhostflux
            #salt3handle2=ax.fill_between(plotmjd,salt3lmhostflux-salt3lmhostfluxerr,salt3lmhostflux+salt3lmhostfluxerr,color='r',alpha=0.1)
            #import pdb; pdb.set_trace()

            
            #salt3flux = salt3k21.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            #salt3lowmasshostmodel.set(xhost=0)
            salt3flux,salt3fluxerr = salt3k21.bandfluxcov(flt, plotmjd,zp=27.5,zpsys='AB')
            salt3fluxerr = np.sqrt(np.diag(salt3fluxerr))
            #salt3handle=ax.plot(plotmjd,salt3flux,color='b',label='SALT3.K21')
            
            #salt3lowmasshostmodel.set(xhost=-0.5)
            salt3lmhostflux,salt3lmhostfluxerr = salt3lowmasshostmodel.bandfluxcov(flt, plotmjd,zp=27.5,zpsys='AB')
            salt3lmhostfluxerr = np.sqrt(np.diag(salt3lmhostfluxerr))
            salt3handle=ax.plot(plotmjd,salt3lmhostflux,color='b',label='SALT3.LowMass')
            salt3handle2=ax.fill_between(plotmjd,salt3lmhostflux-salt3lmhostfluxerr,salt3lmhostflux+salt3lmhostfluxerr,color='b',alpha=0.5)

            #salt3handle2=ax.fill_between(plotmjd,salt3lmhostflux-salt3fluxerr,salt3lmhostflux+salt3fluxerr,color='b',alpha=0.5)
            
            #salt3lmresidhandle=axresid.plot(plotmjd,(salt3hostflux-salt3flux)/salt3flux,color='b') #,label='SALT3.LowMass')

            salt3hmhostflux,salt3hmhostfluxerr = salt3highmasshostmodel.bandfluxcov(flt, plotmjd,zp=27.5,zpsys='AB')
            salt3hmhostfluxerr = np.sqrt(np.diag(salt3hmhostfluxerr))
            salt3handle=ax.plot(plotmjd,salt3hmhostflux,color='r',label='SALT3.HighMass')
            salt3handle2=ax.fill_between(plotmjd,salt3hmhostflux-salt3hmhostfluxerr,salt3hmhostflux+salt3hmhostfluxerr,color='r',alpha=0.5)
            
            salt3residhandle=axresid.plot(plotmjd,(salt3hmhostflux-salt3lmhostflux)/salt3flux,color='0.5') #,label='SALT3.LowMass')
            lower = (salt3hmhostflux-salt3lmhostflux)/salt3flux - np.sqrt(salt3hmhostfluxerr**2.+salt3lmhostfluxerr**2.)/salt3flux
            upper = (salt3hmhostflux-salt3lmhostflux)/salt3flux + np.sqrt(salt3hmhostfluxerr**2.+salt3lmhostfluxerr**2.)/salt3flux
            salt3residhandle2=axresid.fill_between(plotmjd,lower,upper,color='0.5',alpha=0.5) #,label='SALT3.LowMass')
            
            print(f'{flt}, high mass mag = {-2.5*np.log10(salt3lmhostflux[plotmjd==0][0])+27.5}')
            print(f'{flt}, low mass mag = {-2.5*np.log10(salt3hmhostflux[plotmjd==0][0])+27.5}')            
            
            ax.set_yticks([])
            ax.axhline(0,color='k')
            ax.plot([plotmjd.min(),plotmjd.max()],[0,0],'k--')
            axresid.plot([plotmjd.min(),plotmjd.max()],[0,0],'k--')
            axresid.set_ylim([-0.2,0.2])
            if ax != ax1: axresid.yaxis.set_ticklabels([])
            else:
                axresid.set_ylabel('Percent Difference')

        ax1.legend(prop={'size':8},loc='upper right')
                
        #for flt,ax,axresid in zip(filters,[ax6,ax7,ax8,ax9,ax10],[ax6resid,ax7resid,ax8resid,ax9resid,ax10resid]):

            #salt3flux = salt3k21.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            #salt3highmasshostmodel.set(xhost=0)
        #    salt3flux = salt3k21.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
        #    salt3handle=ax.plot(plotmjd,salt3flux,color='b',label='SALT3.K21')

            #salt3highmasshostmodel.set(xhost=0.5)
        #    salt3hostflux = salt3highmasshostmodel.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
        #    salt3handle=ax.plot(plotmjd,salt3hostflux,color='r',label='SALT3.LowMass')

        #    salt3residhandle=axresid.plot(plotmjd,(salt3hostflux-salt3flux)/salt3flux,color='r') #,label='SALT3.LowMass')
            
        #    ax.set_yticks([])
        #    ax.axhline(0,color='k')
        #    ax.plot([plotmjd.min(),plotmjd.max()],[0,0],'k--')
        #    axresid.plot([plotmjd.min(),plotmjd.max()],[0,0],'k--')
        #    axresid.set_ylim([-0.1,0.1])

        #    if ax != ax6: axresid.yaxis.set_ticklabels([])
        #    else:
        #        axresid.set_ylabel('resid.')
            
        import pdb; pdb.set_trace()

    def colors(self):
        
        fig = plt.figure()
        gs = GridSpec(5, 3, figure=fig)
        gs.update(wspace=0.0, hspace=0)

        ax1 = fig.add_subplot(gs[0:3, 0])
        ax2 = fig.add_subplot(gs[0:3, 1])
        ax3 = fig.add_subplot(gs[0:3, 2])

        # resid panels
        ax1resid = fig.add_subplot(gs[3, 0])
        ax2resid = fig.add_subplot(gs[3, 1])
        ax3resid = fig.add_subplot(gs[3, 2])

        for ax,filt1,filt2 in zip([ax1,ax2,ax3],'BVRI','VRI'):
            ax.set_title(f'${filt1}-{filt2}$')
        for ax,filt in zip([ax1resid,ax2resid,ax3resid],'UBVRI'):
            ax.set_xlabel(f'Phase')

        

        salt3lowmasshost = SALT3Source(
            modeldir='SALT3.LowMass',
            m0file='salt3_template_0.dat',
            m1file='salt3_template_1.dat',
            clfile='salt3_color_correction.dat',
            cdfile='salt3_color_dispersion.dat',
            lcrv00file='salt3_lc_variance_0.dat',
            lcrv11file='salt3_lc_variance_1.dat',
            lcrv01file='salt3_lc_covariance_01.dat')
        salt3lowmasshostmodel = sncosmo.Model(salt3lowmasshost)
        salt3highmasshost = SALT3Source(
            modeldir='SALT3.HighMass',
            m0file='salt3_template_0.dat',
            m1file='salt3_template_1.dat',
            clfile='salt3_color_correction.dat',
            cdfile='salt3_color_dispersion.dat',
            lcrv00file='salt3_lc_variance_0.dat',
            lcrv11file='salt3_lc_variance_1.dat',
            lcrv01file='salt3_lc_covariance_01.dat')
        salt3highmasshostmodel = sncosmo.Model(salt3highmasshost)

        
        salt3k21source = SALT3Source(
            modeldir='output',#os.path.expandvars('$SNDATA_ROOT/models/SALT3/SALT3.K21'),
            m0file='salt3_template_0.dat',
            m1file='salt3_template_1.dat',
            clfile='salt3_color_correction.dat',
            cdfile='salt3_color_dispersion.dat',
            lcrv00file='salt3_lc_variance_0.dat',
            lcrv11file='salt3_lc_variance_1.dat',
            lcrv01file='salt3_lc_covariance_01.dat')
        salt3k21 = sncosmo.Model(salt3k21source)

        
        salt3lowmasshostmodel.set(z=0.0)
        salt3lowmasshostmodel.set(x0=1)
        salt3lowmasshostmodel.set(t0=0)
        salt3lowmasshostmodel.set(c=0)

        salt3highmasshostmodel.set(z=0.0)
        salt3highmasshostmodel.set(x0=1)
        salt3highmasshostmodel.set(t0=0)
        salt3highmasshostmodel.set(c=0)

        
        salt3k21.set(z=0.0)
        salt3k21.set(x0=1)
        salt3k21.set(t0=0)
        salt3k21.set(c=0)


        plotmjd = np.linspace(-10, 50,121)

        handles=[]
        
        filtdict = {'SDSS':['sdss%s'%s for s in  'ugri']+['desz'],'Bessell':['bessell%s'%s +('x' if s=='u' else '')for s in  'bvri']}
        filters=filtdict['Bessell']
        for flt1,flt2,ax,axresid in zip(filters,filters[1:],[ax1,ax2,ax3],[ax1resid,ax2resid,ax3resid]):

            #salt3flux = salt3k21.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            #salt3lowmasshostmodel.set(xhost=0)
            #salt3flux = salt3k21.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')

            #salt3handle=ax.plot(plotmjd,salt3flux,color='b',label='SALT3.K21')

            #salt3lowmasshostmodel.set(xhost=-0.5)
            #import pdb; pdb.set_trace()
            salt3lmhostmag1 = salt3lowmasshostmodel.bandmag(flt1, 'AB', plotmjd)#,zp=27.5,zpsys='AB')
            salt3lmhostflux1,salt3lmhostfluxerr1 = salt3lowmasshostmodel.bandfluxcov(flt1, plotmjd,zp=27.5,zpsys='AB')
            salt3lmhostfluxerr1 = np.sqrt(np.diag(salt3lmhostfluxerr1))
            salt3lmhostmagerr1 = 1.086*salt3lmhostfluxerr1/salt3lmhostflux1
            
            salt3lmhostmag2 = salt3lowmasshostmodel.bandmag(flt2, 'AB', plotmjd)#,zp=27.5)#,zpsys='AB')
            salt3lmhostflux2,salt3lmhostfluxerr2 = salt3lowmasshostmodel.bandfluxcov(flt2, plotmjd,zp=27.5,zpsys='AB')
            salt3lmhostfluxerr2 = np.sqrt(np.diag(salt3lmhostfluxerr2))
            salt3lmhostmagerr2 = 1.086*salt3lmhostfluxerr2/salt3lmhostflux2

            salt3handle=ax.plot(plotmjd,salt3lmhostmag1-salt3lmhostmag2,color='b',label='SALT3.LowMass')
            color = salt3lmhostmag1-salt3lmhostmag2
            colorerr = np.sqrt(salt3lmhostmagerr1**2.+salt3lmhostmagerr2**2.)
            salt3handle2=ax.fill_between(plotmjd,color-colorerr,color+colorerr,color='b',alpha=0.5)
            
            #salt3lmresidhandle=axresid.plot(plotmjd,(salt3hostflux-salt3flux)/salt3flux,color='b') #,label='SALT3.LowMass')

            salt3hmhostmag1 = salt3highmasshostmodel.bandmag(flt1, 'AB', plotmjd)#,zp=27.5),zpsys='AB')
            salt3hmhostflux1,salt3hmhostfluxerr1 = salt3lowmasshostmodel.bandfluxcov(flt1, plotmjd,zp=27.5,zpsys='AB')
            salt3hmhostfluxerr1 = np.sqrt(np.diag(salt3hmhostfluxerr1))
            salt3hmhostmagerr1 = 1.086*salt3hmhostfluxerr1/salt3hmhostflux1

            salt3hmhostmag2 = salt3highmasshostmodel.bandmag(flt2, 'AB', plotmjd)#,zp=27.5,zpsys='AB')
            salt3hmhostflux2,salt3hmhostfluxerr2 = salt3lowmasshostmodel.bandfluxcov(flt2, plotmjd,zp=27.5,zpsys='AB')
            salt3hmhostfluxerr2 = np.sqrt(np.diag(salt3hmhostfluxerr2))
            salt3hmhostmagerr2 = 1.086*salt3hmhostfluxerr2/salt3hmhostflux2
            color = salt3hmhostmag1-salt3hmhostmag2
            colorerr = np.sqrt(salt3hmhostmagerr1**2.+salt3hmhostmagerr2**2.)
            salt3handle=ax.plot(plotmjd,salt3hmhostmag1-salt3hmhostmag2,color='r',label='SALT3.HighMass')
            salt3handle2=ax.fill_between(plotmjd,color-colorerr,color+colorerr,color='r',alpha=0.5)

            colordiff = (salt3hmhostmag1-salt3hmhostmag2)-(salt3lmhostmag1-salt3lmhostmag2)
            colordifferr = np.sqrt(salt3hmhostmagerr1**2.+salt3hmhostmagerr2**2.+salt3lmhostmagerr1**2.+salt3lmhostmagerr2**2.)
            salt3residhandle=axresid.plot(plotmjd,(salt3hmhostmag1-salt3hmhostmag2)-(salt3lmhostmag1-salt3lmhostmag2),color='0.5')
            salt3residhandle2=axresid.fill_between(plotmjd,colordiff-colordifferr,colordiff+colordifferr,color='0.5',alpha=0.5)
            
            #print(f'{flt}, high mass mag = {-2.5*np.log10(salt3lmhostflux[plotmjd==0][0])+27.5}')
            #print(f'{flt}, low mass mag = {-2.5*np.log10(salt3hmhostflux[plotmjd==0][0])+27.5}')            
            
            #ax.set_yticks([])
            ax.set_ylim([-0.7,1.2])
            ax.axhline(0,color='k')
            ax.plot([plotmjd.min(),plotmjd.max()],[0,0],'k--')
            axresid.plot([plotmjd.min(),plotmjd.max()],[0,0],'k--')
            axresid.set_ylim([-0.1,0.1])
            if ax != ax1:
                axresid.yaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
            else:
                axresid.set_ylabel('Resid.')
        ax1.set_ylabel('Color (mag)')
        ax1.legend(prop={'size':13},loc='lower right')
        plt.ion(); plt.show()
        import pdb; pdb.set_trace()
        
    def m1diff(self,modelname='SALT3.fixedpars_'):

        # first use GridSpec to set up the layout
        fig = plt.figure()
        gs = GridSpec(5, 5, figure=fig)
        gs.update(wspace=0.0, hspace=0)

        ax1 = fig.add_subplot(gs[0:3, 0])
        ax2 = fig.add_subplot(gs[0:3, 1])
        ax3 = fig.add_subplot(gs[0:3, 2])
        ax4 = fig.add_subplot(gs[0:3, 3])
        ax5 = fig.add_subplot(gs[0:3, 4])

        # random stuff
        for ax,filt in zip([ax1,ax2,ax3,ax4,ax5],'UBVRI'):
            ax.set_title(f'${filt}$')

        ax1.set_ylabel('Flux')


        filtdict = {'SDSS':['sdss%s'%s for s in  'ugri']+['desz'],'Bessell':['bessell%s'%s +('x' if s=='u' else '')for s in  'ubvri']}
        filters=filtdict['Bessell']
    
        zpsys='AB'
    
        salt3lowmasshost = SALT3Source(
            modeldir=f'{modelname}LowMass',
            m0file='salt3_template_0.dat',
            m1file='salt3_template_1.dat',
            clfile='salt3_color_correction.dat',
            cdfile='salt3_color_dispersion.dat',
            lcrv00file='salt3_lc_variance_0.dat',
            lcrv11file='salt3_lc_variance_1.dat',
            lcrv01file='salt3_lc_covariance_01.dat')
        salt3lowmasshostmodel = sncosmo.Model(salt3lowmasshost)
        salt3highmasshost = SALT3Source(
            modeldir=f'{modelname}HighMass',
            m0file='salt3_template_0.dat',
            m1file='salt3_template_1.dat',
            clfile='salt3_color_correction.dat',
            cdfile='salt3_color_dispersion.dat',
            lcrv00file='salt3_lc_variance_0.dat',
            lcrv11file='salt3_lc_variance_1.dat',
            lcrv01file='salt3_lc_covariance_01.dat')
        salt3highmasshostmodel = sncosmo.Model(salt3highmasshost)

        
        salt3k21source = SALT3Source(
            modeldir='/Users/David/Dropbox/research/SALTShaker/examples/SALT3TRAIN_K21_PUBLIC/SALT3.Fragilistic', #os.path.expandvars('$SNDATA_ROOT/models/SALT3/SALT3.K21'),
            m0file='salt3_template_0.dat',
            m1file='salt3_template_1.dat',
            clfile='salt3_color_correction.dat',
            cdfile='salt3_color_dispersion.dat',
            lcrv00file='salt3_lc_variance_0.dat',
            lcrv11file='salt3_lc_variance_1.dat',
            lcrv01file='salt3_lc_covariance_01.dat')
        salt3k21 = sncosmo.Model(salt3k21source)

        
        salt3lowmasshostmodel.set(z=0.0)
        salt3lowmasshostmodel.set(x0=1)
        salt3lowmasshostmodel.set(t0=0)
        salt3lowmasshostmodel.set(c=0)

        salt3highmasshostmodel.set(z=0.0)
        salt3highmasshostmodel.set(x0=1)
        salt3highmasshostmodel.set(t0=0)
        salt3highmasshostmodel.set(c=0)

        
        salt3k21.set(z=0.0)
        salt3k21.set(x0=1)
        salt3k21.set(t0=0)
        salt3k21.set(c=0)


        plotmjd = np.linspace(-10, 50,121)

        handles=[]
        for flt,ax in zip(filters,[ax1,ax2,ax3,ax4,ax5]):
            salt3k21.set(x1=0)
            salt3fluxbase = salt3k21.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            salt3k21.set(x1=1)
            salt3fluxx1 = salt3k21.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')

            salt3lowmasshostmodel.set(x1=0)
            salt3lmhostfluxbase = salt3lowmasshostmodel.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            salt3lowmasshostmodel.set(x1=1)
            salt3lmhostfluxx1 = salt3lowmasshostmodel.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')

            salt3hmhostfluxbase = salt3highmasshostmodel.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            
            diff_host = salt3lmhostfluxx1-salt3lmhostfluxbase
            diff_k21 = salt3fluxx1-salt3fluxbase
            diff_host2 = salt3hmhostfluxbase - salt3lmhostfluxbase
            ax.plot(plotmjd,(diff_host-diff_k21)/salt3fluxbase,color='k',ls='--',label=r'$(M_{1,{\rm host}}-M_{1,{\rm K21}})/M_0$')
            ax.plot(plotmjd,diff_host2/salt3fluxbase,color='k',label=r'$M_{\rm host}/M_0$')
            
            ax.axhline(0,color='k')
            ax.plot([plotmjd.min(),plotmjd.max()],[0,0],'k--')
            ax.set_xlabel('Phase')
            ax.set_ylim([-0.15,0.15])
            if not ax == ax1:
                ax.yaxis.set_ticklabels([])
                
        ax1.set_ylabel(r'Fractional Difference') #$(M_{1,{\rm host}}-M_{1,{\rm K21}})/M_0$')
        ax1.legend()
        
        
        import pdb; pdb.set_trace()

        
if __name__ == "__main__":
    lc = lcplot()
    lc.main()
    #lc.colors()
    #lc.m1diff()
