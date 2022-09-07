#!/usr/bin/env python
import numpy as np
import pylab as plt
from matplotlib import colors
import sncosmo
import argparse
from saltshaker.util import snana
from astropy.table import Table
import astropy.units as u
from saltshaker.util.synphot import synphot
from scipy.interpolate import interp1d
from sncosmo.constants import HC_ERG_AA
from saltshaker.initfiles import init_rootdir
from saltshaker.training.init_hsiao import synphotB
from sncosmo.salt2utils import SALT2ColorLaw
import os
from SALT3HostSource import SALT3HostSource

plt.rcParams['font.size'] = 10
_SCALE_FACTOR = 1e-12

def overPlotSynthPhotByComponent(outfile,salt3dir,filterset='SDSS',
         m0file='salt3_template_0.dat',
         m1file='salt3_template_1.dat',
         clfile='salt3_color_correction.dat',
         cdfile='salt3_color_dispersion.dat',
         errscalefile='salt3_lc_dispersion_scaling.dat',
         lcrv00file='salt3_lc_variance_0.dat',
         lcrv11file='salt3_lc_variance_1.dat',
         lcrv01file='salt3_lc_covariance_01.dat'):
    
    plt.clf()



    filtdict = {'SDSS':['sdss%s'%s for s in  'ugri']+['desz'],'Bessell':['bessell%s'%s +('x' if s=='u' else '')for s in  'ubvri']}
    filters=filtdict[filterset]
    
    zpsys='AB'
    
    salt2model = sncosmo.Model(source='salt2')
    salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,
                                m1file=m1file,
                                clfile=clfile,cdfile=cdfile,
                                errscalefile=errscalefile,
                                lcrv00file=lcrv00file,
                                lcrv11file=lcrv11file,
                                lcrv01file=lcrv01file)
    salt3model =  sncosmo.Model(salt3)
    
    salt2model.set(z=0.0)
    salt2model.set(x0=1)
    salt2model.set(t0=0)
    salt2model.set(c=0)
    
    salt3model.set(z=0.0)
    salt3model.set(x0=1)
    salt3model.set(t0=0)
    salt3model.set(c=0)

    plotmjd = np.linspace(-20, 50,100)
    
#   fig = plt.figure(figsize=)
    fig,(m0axes,m1axes,familyaxes)=plt.subplots(3,len(filters),sharex=True,figsize=(7,3.4),gridspec_kw={'hspace':0})
#   m0axes = [fig.add_subplot(3,len(filters),1+i) for i in range(len(filters))]
#   m1axes = [fig.add_subplot(3,len(filters),len(filters)+ 1+i,sharex=ax) for i,ax in enumerate(m0axes)]
#   familyaxes = [fig.add_subplot(3,len(filters),2*len(filters)+ 1+i,sharex=ax) for i,ax in enumerate(m0axes)]
    xmin,xmax=-2,2
    handles=[]
    for flt,ax0,ax1 in zip(filters,m0axes,m1axes):
        
            salt2model.set(x1=1)
            salt3model.set(x1=1)
            
            try:
                salt2stretchedflux = salt2model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
                salt2model.set(x1=0)
                salt2flux = salt2model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
                salt2handle=ax0.plot(plotmjd,salt2flux,color='b',label='SALT2')
                ax1.plot(plotmjd,salt2stretchedflux-salt2flux,color='b',label='SALT2')
            except: pass

            try:
                salt3stretchedflux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
                salt3model.set(x1=0)

                salt3flux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
                salt3handle=ax0.plot(plotmjd,salt3flux,color='r',label='SALT3')
                ax1.plot(plotmjd,salt3stretchedflux-salt3flux,color='r',label='SALT3')
            except: pass

            ax0.set_yticks([])
            ax1.set_yticks([])

            ax0.axhline(0,color='k')
            ax0.plot([plotmjd.min(),plotmjd.max()],[0,0],'k--')
            ax1.plot([plotmjd.min(),plotmjd.max()],[0,0],'k--')
        
            title=flt
            if 'bessell' in title:
                title= 'Bessell '+ flt[len('bessell')].upper()
            ax0.set_title(title)

    xmin,xmax=-2,2
    norm=colors.Normalize(vmin=xmin,vmax=xmax)
    cmap=plt.get_cmap('RdBu')
    #line = plt.Line2D([0,1],[275./422]*2, transform=fig.transFigure, color="black")

    for flt,ax in zip(filters,familyaxes):
        
        for x1 in np.linspace(xmin,xmax,100,True):
            salt3model.set(x1=x1)
            color=cmap(norm(x1))
            try:
                salt3flux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
                ax.plot(plotmjd,salt3flux,color=color,label='SALT3',linewidth=0.1)
            except: pass
            
            ax.set_yticks([])


            #ax.set_xlim([-30,55])
    sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    sm._A=[]
    fig.subplots_adjust(right=0.8,bottom=0.15,left=0.05)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.35])
    familyaxes[0].set_ylabel('SALT3 Flux')
    for ax in familyaxes: ax.set_xlim(plotmjd.min(),plotmjd.max())
    fig.colorbar(sm,cax=cbar_ax)
    cbar_ax.set_ylabel('Stretch  ($x_1$ parameter)')

    #axes[0].legend()
    
    m0axes[0].set_ylabel('M0 Flux')
    m1axes[0].set_ylabel('M1 Flux')
    fig.text(0.5,0.04,'Time since peak (days)',ha='center')
    fig.legend(salt2handle+salt3handle,['SALT2.JLA','SALT3.K21'],loc=(.825,.75))
    #axes[0].legend()

    plt.savefig(outfile)
    plt.close(fig)

def overPlotSynthPhotByComponentWithMass(
        outfile,salt3dir,filterset='SDSS',
        m0file='salt3_template_0.dat',
        m1file='salt3_template_1.dat',
        mhostfile='salt3_template_host.dat',
        clfile='salt3_color_correction.dat',
        cdfile='salt3_color_dispersion.dat',
        errscalefile='salt3_lc_dispersion_scaling.dat',
        lcrv00file='salt3_lc_variance_0.dat',
        lcrv11file='salt3_lc_variance_1.dat',
        lcrv01file='salt3_lc_covariance_01.dat'):
    
    plt.clf()


    filtdict = {'SDSS':['sdss%s'%s for s in  'ugri']+['desz'],'Bessell':['bessell%s'%s +('x' if s=='u' else '')for s in  'ubvri']}
    filters=filtdict[filterset]
    
    zpsys='AB'
    
    salt2model = sncosmo.Model(source='salt2')
    #salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,
    #                            m1file=m1file,
    #                            clfile=clfile,cdfile=cdfile,
    #                            errscalefile=errscalefile,
    #                            lcrv00file=lcrv00file,
    #                            lcrv11file=lcrv11file,
    #                            lcrv01file=lcrv01file)

    salt3 = SALT3HostSource(modeldir=salt3dir,m0file=m0file,
                            m1file=m1file,
                            mhostfile=mhostfile,
                            clfile=clfile,cdfile=cdfile,
                            lcrv00file=lcrv00file,
                            lcrv11file=lcrv11file,
                            lcrv01file=lcrv01file)
    salt3model =  sncosmo.Model(salt3)
    
    salt2model.set(z=0.0)
    salt2model.set(x0=1)
    salt2model.set(t0=0)
    salt2model.set(c=0)
    
    salt3model.set(z=0.0)
    salt3model.set(x0=1)
    salt3model.set(t0=0)
    salt3model.set(c=0)

    plotmjd = np.linspace(-10, 50,100)
    
    fig,(m0axes,m1axes,mhostaxes,familyaxes)=plt.subplots(4,len(filters),sharex=True,figsize=(7,3.4),gridspec_kw={'hspace':0})
    
    xmin,xmax=-2,2
    handles=[]
    for flt,ax0,ax1,ax2 in zip(filters,m0axes,m1axes,mhostaxes):
        
            salt2model.set(x1=1)
            salt3model.set(x1=1)
            
            try:
                salt2stretchedflux = salt2model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
                salt2model.set(x1=0)
                salt2flux = salt2model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
                salt2handle=ax0.plot(plotmjd,salt2flux,color='b',label='SALT2')
                ax1.plot(plotmjd,salt2stretchedflux-salt2flux,color='b',label='SALT2')
            except: pass

            #try:
            salt3stretchedflux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            salt3model.set(x1=0)

            salt3flux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            salt3handle=ax0.plot(plotmjd,salt3flux,color='r',label='SALT3')
            ax1.plot(plotmjd,salt3stretchedflux-salt3flux,color='r',label='SALT3')
#            import pdb; pdb.set_trace()
            #except: pass

            salt3model.set(xhost=0.5)
            
            #try:
            salt3hostflux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            salt3model.set(xhost=0)

            salt3flux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            salt3handle=ax0.plot(plotmjd,salt3flux,color='r',label='SALT3')
            ax2.plot(plotmjd,salt3hostflux-salt3flux,color='r',label='SALT3')
#            import pdb; pdb.set_trace()
            #except:
            #    pass
            
            ax0.set_yticks([])
            ax1.set_yticks([])
            ax2.set_yticks([])
            
            ax0.axhline(0,color='k')
            ax0.plot([plotmjd.min(),plotmjd.max()],[0,0],'k--')
            ax1.plot([plotmjd.min(),plotmjd.max()],[0,0],'k--')
            ax2.plot([plotmjd.min(),plotmjd.max()],[0,0],'k--')
            
            title=flt
            if 'bessell' in title:
                title= 'Bessell '+ flt[len('bessell')].upper()
            ax0.set_title(title)

    xmin,xmax=-2,2
    norm=colors.Normalize(vmin=xmin,vmax=xmax)
    cmap=plt.get_cmap('RdBu')


    for flt,ax in zip(filters,familyaxes):
        
        for x1 in np.linspace(xmin,xmax,100,True):
            salt3model.set(x1=x1)
            color=cmap(norm(x1))
            try:
                salt3flux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
                ax.plot(plotmjd,salt3flux,color=color,label='SALT3',linewidth=0.1)
            except: pass
            
            ax.set_yticks([])


            #ax.set_xlim([-30,55])
    sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    sm._A=[]
    fig.subplots_adjust(right=0.8,bottom=0.15,left=0.05)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.35])
    familyaxes[0].set_ylabel('SALT3 Flux')
    for ax in familyaxes: ax.set_xlim(plotmjd.min(),plotmjd.max())
    fig.colorbar(sm,cax=cbar_ax)
    cbar_ax.set_ylabel('Stretch  ($x_1$ parameter)')


    
    m0axes[0].set_ylabel('M0 Flux')
    m1axes[0].set_ylabel('M1 Flux')
    mhostaxes[0].set_ylabel('Mhost Flux')
    fig.text(0.5,0.04,'Time since peak (days)',ha='center')
    fig.legend(salt2handle+salt3handle,['SALT2.JLA','SALT3.K21'],loc=(.825,.75))


    plt.savefig(outfile)
    plt.close(fig)

    
def overPlotSynthPhotByComponentCustom(
        outfile,salt3dir,filterset='SDSS',
        m0file='salt3_template_0.dat',
        m1file='salt3_template_1.dat',
        clfile='salt3_color_correction.dat',
        cdfile='salt3_color_dispersion.dat',
        errscalefile='salt3_lc_dispersion_scaling.dat',
        lcrv00file='salt3_lc_variance_0.dat',
        lcrv11file='salt3_lc_variance_1.dat',
        lcrv01file='salt3_lc_covariance_01.dat',
        ufilt='$SNDATA_ROOT/filters/SDSS/SDSS_web2001/u.dat',
        gfilt='$SNDATA_ROOT/filters/SDSS/SDSS_web2001/g.dat',
        rfilt='$SNDATA_ROOT/filters/SDSS/SDSS_web2001/r.dat',
        ifilt='$SNDATA_ROOT/filters/SDSS/SDSS_web2001/i.dat',
        zfilt='$SNDATA_ROOT/filters/SDSS/SDSS_web2001/z.dat'):
    
    plt.clf()


    # read everything in
    salt2dir = os.path.expandvars('$SNDATA_ROOT/models/SALT2/SALT2.JLA-B14')
    salt3phase,salt3wave,salt3flux = np.genfromtxt('%s/%s'%(salt3dir,m0file),unpack=True)
    salt3m1phase,salt3m1wave,salt3m1flux = np.genfromtxt('%s/%s'%(salt3dir,m1file),unpack=True)
    salt2phase,salt2wave,salt2flux = np.genfromtxt('{}/salt2_template_0.dat'.format(salt2dir),unpack=True)
    salt2m1phase,salt2m1wave,salt2m1flux = np.genfromtxt('{}/salt2_template_1.dat'.format(salt2dir),unpack=True)

    salt3flux = salt3flux.reshape([len(np.unique(salt3phase)),len(np.unique(salt3wave))])
    salt3m1flux = salt3m1flux.reshape([len(np.unique(salt3phase)),len(np.unique(salt3wave))])
    salt3phase = np.unique(salt3phase)
    salt3wave = np.unique(salt3wave)

    salt2m0flux = salt2flux.reshape([len(np.unique(salt2phase)),len(np.unique(salt2wave))])
    salt2flux = salt2flux.reshape([len(np.unique(salt2phase)),len(np.unique(salt2wave))])
    salt2m1flux = salt2m1flux.reshape([len(np.unique(salt2phase)),len(np.unique(salt2wave))])
    salt2phase = np.unique(salt2phase)
    salt2wave = np.unique(salt2wave)
    int1dm0 = interp1d(salt3phase,salt3flux,axis=0,fill_value='extrapolate')
    int1dm1 = interp1d(salt3phase,salt3m1flux,axis=0,fill_value='extrapolate')
    int1ds2m0 = interp1d(salt2phase,salt2flux,axis=0,fill_value='extrapolate')
    int1ds2m1 = interp1d(salt2phase,salt2m1flux,axis=0,fill_value='extrapolate')
    

    filtdict = {'SDSS':['sdss%s'%s for s in  'ugri']+['desz'],'Bessell':['bessell%s'%s +('x' if s=='u' else '')for s in  'ubvri']}
    filters=filtdict[filterset]
    
    zpsys='AB'
    
    salt2model = sncosmo.Model(source='salt2')
    salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,
                                m1file=m1file,
                                clfile=clfile,cdfile=cdfile,
                                errscalefile=errscalefile,
                                lcrv00file=lcrv00file,
                                lcrv11file=lcrv11file,
                                lcrv01file=lcrv01file)
    salt3model =  sncosmo.Model(salt3)
    
    salt2model.set(z=0.025)
    salt2model.set(x0=1)
    salt2model.set(t0=0)
    salt2model.set(c=0)
    
    salt3model.set(z=0.025)
    salt3model.set(x0=1)
    salt3model.set(t0=0)
    salt3model.set(c=0)

    plotmjd = np.linspace(-20, 55,100)
    
    fig = plt.figure(figsize=(15, 5))
    m0axes = [fig.add_subplot(2,len(filters),1+i) for i in range(len(filters))]
    m1axes = [fig.add_subplot(2,len(filters),len(filters)+ 1+i,sharex=ax) for i,ax in enumerate(m0axes)]
    xmin,xmax=-2,2
    for flt,ax0,ax1,fltfile in zip(filters,m0axes,m1axes,[ufilt,gfilt,rfilt,ifilt]):

        salt3m0flux = int1dm0(plotmjd)
        salt3m1flux = int1dm1(plotmjd)
        salt2m0flux = int1ds2m0(plotmjd)
        salt2m1flux = int1ds2m1(plotmjd)
        
        filtwave,filttrans = np.loadtxt(os.path.expandvars(fltfile),unpack=True)
        g = (salt3wave >= filtwave[0]) & (salt3wave <= filtwave[-1])  # overlap range
        pbspl = np.interp(salt3wave[g],filtwave,filttrans)
        pbspl *= salt3wave[g]
        deltawave = salt3wave[g][1]-salt3wave[g][0]
        denom = np.sum(pbspl)*deltawave
        salt3m0synflux=np.sum(pbspl[np.newaxis,:]*salt3m0flux[:,g],axis=1)*deltawave/HC_ERG_AA/denom
        salt3m1synflux=np.sum(pbspl[np.newaxis,:]*salt3m1flux[:,g],axis=1)*deltawave/HC_ERG_AA/denom
        g = (salt2wave >= filtwave[0]) & (salt2wave <= filtwave[-1])  # overlap range
        pbspl = np.interp(salt2wave[g],filtwave,filttrans)
        pbspl *= salt2wave[g]
        deltawave = salt2wave[g][1]-salt2wave[g][0]
        denom = np.sum(pbspl)*deltawave
        salt2m0synflux=np.sum(pbspl[np.newaxis,:]*salt2m0flux[:,g],axis=1)*deltawave/HC_ERG_AA/denom
        salt2m1synflux=np.sum(pbspl[np.newaxis,:]*salt2m1flux[:,g],axis=1)*deltawave/HC_ERG_AA/denom
        
        salt2model.set(x1=1)
        salt3model.set(x1=1)
        salt2stretchedflux = salt2model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
        salt3stretchedflux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')#*\

        salt2model.set(x1=0)
        salt3model.set(x1=0)
        salt2flux = salt2model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
        salt3flux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')#*\

        #ax0.set_yticks([])
        #ax1.set_yticks([])
        #ax0.plot(plotmjd,salt2flux,color='b',label='SALT2')
        #ax0.plot(plotmjd,salt3flux,color='r',label='SALT3')
        ax0.plot(plotmjd,salt2m0synflux,color='b',label='SALT2 Syn')
        ax0.plot(plotmjd,salt3m0synflux,color='r',label='SALT3 Syn')
        #ax1.plot(plotmjd,salt2stretchedflux-salt2flux,color='b',label='SALT2')
        #ax1.plot(plotmjd,salt3stretchedflux-salt3flux,color='r',label='SALT3')
        ax1.plot(plotmjd,salt2m1synflux,color='b',label='SALT2 Syn')
        ax1.plot(plotmjd,salt3m1synflux,color='r',label='SALT3 Syn')    
        title=flt
        if 'bessell' in title:
            title= 'Bessell '+ flt[len('bessell')].upper()
        ax0.set_title(title,fontsize=20)
        #ax.set_xlim([-30,55])
        
    fig.subplots_adjust(right=0.8,bottom=0.15,left=0.05)
    m0axes[0].set_ylabel('M0 Flux',fontsize=20)
    m1axes[0].set_ylabel('M1 Flux',fontsize=20)
    
    fig.text(0.5,0.04,'Time since peak (days)',ha='center',fontsize=20)
    #axes[0].legend()
    plt.savefig(outfile)
    plt.close(fig)
    

def plotSynthPhotOverStretchRange(outfile,salt3dir,filterset='SDSS',
         m0file='salt3_template_0.dat',
         m1file='salt3_template_1.dat',
         clfile='salt3_color_correction.dat',
         cdfile='salt3_color_dispersion.dat',
         errscalefile='salt3_lc_dispersion_scaling.dat',
         lcrv00file='salt3_lc_variance_0.dat',
         lcrv11file='salt3_lc_variance_1.dat',
         lcrv01file='salt3_lc_covariance_01.dat',includeSALT2=True):
    
    plt.clf()



    filtdict = {'SDSS':['sdss%s'%s for s in  'ugri']+['desz'],'Bessell':['bessell%s'%s +('x' if s=='u' else '')for s in 'ubvri']}
    filters=filtdict[filterset]
    
    zpsys='AB'
    
    salt2model = sncosmo.Model(source='salt2')
    salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,
                                m1file=m1file,
                                clfile=clfile,cdfile=cdfile,
                                errscalefile=errscalefile,
                                lcrv00file=lcrv00file,
                                lcrv11file=lcrv11file,
                                lcrv01file=lcrv01file)
    salt3model =  sncosmo.Model(salt3)
    
    salt2model.set(z=0)
    salt2model.set(x0=1)
    salt2model.set(t0=0)
    salt2model.set(c=0)
    
    salt3model.set(z=0)
    salt3model.set(x0=1)
    salt3model.set(t0=0)
    salt3model.set(c=0)

    plotmjd = np.linspace(-20, 55,100)
    
    fig = plt.figure(figsize=(15, 5))
    salt3axes = [fig.add_subplot(1+includeSALT2,len(filters),1+i) for i in range(len(filters))]
    if includeSALT2: salt2axes = [fig.add_subplot(2,len(filters),len(filters)+ 1+i,sharex=salt3ax) for i,salt3ax in enumerate(salt3axes)]
    else:
        salt2axes=[None for ax in salt3axes]    
    xmin,xmax=-2,2
    norm=colors.Normalize(vmin=xmin,vmax=xmax)
    cmap=plt.get_cmap('RdBu')
    for flt,ax2,ax3 in zip(filters,salt2axes,salt3axes):
        
        for x1 in np.linspace(xmin,xmax,100,True):
            salt2model.set(x1=x1)
            salt3model.set(x1=x1)
            color=cmap(norm(x1))
            if includeSALT2: 
                try:
                    salt2flux = salt2model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
                    ax2.plot(plotmjd,salt2flux,color=color,label='SALT2',linewidth=0.1)
                except: pass
            try:
                salt3flux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
                ax3.plot(plotmjd,salt3flux,color=color,label='SALT3',linewidth=0.1)
            except: pass
            
            if includeSALT2: ax2.set_yticks([])
            ax3.set_yticks([])


            title=flt
            if 'bessell' in title:
                title= 'Bessell '+ flt[len('bessell')].upper()
            ax3.set_title(title,fontsize=20)
            #ax.set_xlim([-30,55])
    sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    sm._A=[]
    fig.subplots_adjust(right=0.8,bottom=0.15,left=0.05)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    if includeSALT2: salt2axes[0].set_ylabel('SALT2 Flux',fontsize=20)
    salt3axes[0].set_ylabel('SALT3 Flux',fontsize=20)

    fig.colorbar(sm,cax=cbar_ax)
    cbar_ax.set_ylabel('Stretch  ($x_1$ parameter)',fontsize=20)
    
    fig.text(0.5,0.04,'Time since peak (days)',ha='center',fontsize=20)
    #axes[0].legend()
    plt.savefig(outfile)
    plt.close(fig)

def plotSynthPhotOverStretchRangeNIR(
        outfile,salt3dir,filterset='CSP',
        m0file='salt3_template_0.dat',
        m1file='salt3_template_1.dat',
        clfile='salt3_color_correction.dat',
        cdfile='salt3_color_dispersion.dat',
        errscalefile='salt3_lc_dispersion_scaling.dat',
        lcrv00file='salt3_lc_variance_0.dat',
        lcrv11file='salt3_lc_variance_1.dat',
        lcrv01file='salt3_lc_covariance_01.dat',includeSALT2=False):
    
    plt.clf()



    filtdict = {'SDSS':['sdss%s'%s for s in  'ugri']+['desz'],'Bessell':['bessell%s'%s +('x' if s=='u' else '')for s in 'ubvri'],
                'CSP':['csp%s'%s for s in ['ys','js']]}
    filters=filtdict[filterset]
    
    zpsys='AB'
    
    salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,
                                m1file=m1file,
                                clfile=clfile,cdfile=cdfile,
                                errscalefile=errscalefile,
                                lcrv00file=lcrv00file,
                                lcrv11file=lcrv11file,
                                lcrv01file=lcrv01file)
    salt3model =  sncosmo.Model(salt3)
        
    salt3model.set(z=0)
    salt3model.set(x0=1)
    salt3model.set(t0=0)
    salt3model.set(c=0)

    plotmjd = np.linspace(-20, 55,100)
    
    fig = plt.figure(figsize=(15, 5))
    salt3axes = [fig.add_subplot(1,len(filters),1+i) for i in range(len(filters))]
    xmin,xmax=-2,2
    norm=colors.Normalize(vmin=xmin,vmax=xmax)
    cmap=plt.get_cmap('RdBu')
    for flt,ax3 in zip(filters,salt3axes):
        
        for x1 in np.linspace(xmin,xmax,100,True):
            salt3model.set(x1=x1)
            color=cmap(norm(x1))
            salt3flux = salt3model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            ax3.plot(plotmjd,salt3flux,color=color,label='SALT3',linewidth=0.1)
            
            ax3.set_yticks([])


            title=flt
            if 'bessell' in title:
                title= 'Bessell '+ flt[len('bessell')].upper()
            ax3.set_title(title,fontsize=20)
            #ax.set_xlim([-30,55])
    sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    sm._A=[]
    fig.subplots_adjust(right=0.8,bottom=0.15,left=0.05)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    salt3axes[0].set_ylabel('SALT3 Flux',fontsize=20)

    fig.colorbar(sm,cax=cbar_ax)
    cbar_ax.set_ylabel('Stretch  ($x_1$ parameter)',fontsize=20)
    
    fig.text(0.5,0.04,'Time since peak (days)',ha='center',fontsize=20)
    #axes[0].legend()
    plt.savefig(outfile)
    plt.close(fig)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot synthetic photometry for a range of x1 values')
    parser.add_argument('salt3dir',type=str,help='File with supernova fit parameters')
    parser.add_argument('outdir',type=str,nargs='?',default=None,help='File with supernova fit parameters')
    parser.add_argument('--filterset',type=str,nargs='?',default='SDSS',help='File with supernova fit parameters')
    parser=parser.parse_args()
    args=vars(parser)
    if parser.outdir is None:
        parser.outdir=(parser.salt3dir)
    #plotSynthPhotOverStretchRange('{}/synthphotrange.pdf'.format(parser.outdir),parser.salt3dir,parser.filterset)
    #overPlotSynthPhotByComponent('{}/synthphotoverplot.pdf'.format(parser.outdir),parser.salt3dir,parser.filterset)
    #plotSynthPhotOverStretchRangeNIR('{}/synthphotrangeNIR.pdf'.format(parser.outdir),parser.salt3dir) 

    overPlotSynthPhotByComponentWithMass('{}/synthphotoverplotmass.pdf'.format(parser.outdir),parser.salt3dir,parser.filterset)
