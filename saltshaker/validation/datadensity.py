import unittest,os
from scipy import sparse,stats
import numpy as np

from matplotlib import ticker as mtick,colors

import os
import matplotlib.pyplot as plt
import sys

from mpl_toolkits.axes_grid1 import make_axes_locatable
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

labelsize=25
def getfwhm(kcordict,survey,filt):
    wavesabovehalfmax=kcordict[survey][filt]['filtwave'][kcordict[survey][filt]['filttrans']>(kcordict[survey][filt]['filttrans'].max()/2)]
    return wavesabovehalfmax.min(),wavesabovehalfmax.max()

def getspecdensity(phasebins,wavebins,datadict):
    neffRaw=np.zeros((phasebins.size-1,wavebins.size-1))

    # wavebins=wavebins[:-1],wavebins[1:]
    # phasebins=phasebins[:-1],phasebins[1:]

    for sn in (datadict.keys()):
        photdata = datadict[sn].photdata
        specdata = datadict[sn].specdata
        survey = datadict[sn].survey
        z = datadict[sn].zHelio
        tpkoff=0
        #For each spectrum, add one point to each bin for every spectral measurement in that bin
    # 			spectime-=time.time()
        for k in specdata.keys():
            # weight by ~mag err?
            err=specdata[k].fluxerr/specdata[k].flux
            snr=specdata[k].flux/specdata[k].fluxerr
            restWave=specdata[k].wavelength/(1+z)
            err=err[(restWave>wavebins[0])&(restWave<wavebins[-1])]
            snr=snr[(restWave>wavebins[0])&(restWave<wavebins[-1])]
            flux=specdata[k].flux[(restWave>wavebins[0])&(restWave<wavebins[-1])]
            fluxerr=specdata[k].fluxerr[(restWave>wavebins[0])&(restWave<wavebins[-1])]
            restWave=restWave[(restWave>wavebins[0])&(restWave<wavebins[-1])]

            phase=(specdata[k].tobs+tpkoff)/(1+z)
            if phase<phasebins[0]:
                phaseIndex=0
            elif phase>phasebins[-1]:
                phaseIndex=-1
            else:
                phaseIndex= np.where( (phase>=phasebins[:-1]) & (phase<phasebins[1:]))[0][0]


            #neffNoWeight = np.histogram(restWave,wavebins)[0]
            #snr = ss.binned_statistic(restWave,snr,bins=wavebins,statistic='sum').statistic
            #neffNoWeight = neffNoWeight*snr**2./900 #[snr < 5] = 0.0
            spec_cov = stats.binned_statistic(
                restWave,flux/flux.max()/len(flux),
                bins=wavebins,statistic='sum').statistic
            # HACK
            neffRaw[phaseIndex,:]+=(spec_cov>0)

            #neffNoWeight/np.median(neffNoWeight[(wavebins[1:] < wavebins.min()+500) |
            #																(wavebins[1:] > wavebins.min()-500)])
            #np.histogram(restWave,wavebins)[0]
    return neffRaw


def getphotdensity(phasebins,wavebins,datadict,kcordict,countlightcurves=False):
    neffphot=np.zeros((phasebins.size-1,wavebins.size-1))
    lambdaeffarr=[]
    for sn in (datadict.keys()):
        photdata = datadict[sn].photdata
        survey = datadict[sn].survey
        z = datadict[sn].zHelio
        tpkoff=0
        #For each spectrum, add one point to each bin for every spectral measurement in that bin
    # 			spectime-=time.time()
        for flt in (photdata):

            restphase=(photdata[flt].tobs)/(1+z)
            fwhm=getfwhm(kcordict,survey,flt)
            lambdaeff=kcordict[survey][flt]['lambdaeff']
            photcount =  (fwhm[1]>(1+z)*wavebins[:-1] )& (fwhm[0]<(1+z)*wavebins[1:])
            fltcount=np.zeros(shape=neffphot.shape)
            for phase in restphase:
                if phase<phasebins[0]:
                    phaseIndex=0
                elif phase>phasebins[-1]:
                    phaseIndex=-1
                else:
                    phaseIndex= np.where( (phase>=phasebins[:-1]) & (phase<phasebins[1:]))[0][0]
                fltcount[phaseIndex,:]+=photcount
            
            neffphot+=(fltcount>0) if countlightcurves else fltcount
            lambdaeffarr+=[lambdaeff/(1+z)]
            
    return neffphot
    
    
def datadensityplot(filename,phasebins,wavebins,datadict,kcordict,photscale=None,specscale=None):
    fig,axes=plt.subplots(2,2,sharex=True,sharey=True,squeeze=True,figsize=np.array([4,6])*7/6,)
    labelsize=11
    ticksize=8
    axes,coloraxes=[x[0] for x in axes], [x[1] for x in axes]
    photdensity,specdensity=getphotdensity(phasebins,wavebins,datadict,kcordict),getspecdensity(phasebins,wavebins,datadict)
    for name,ax,colorax,datacomb in [('spec', axes[1],coloraxes[1],specdensity), ('phot',axes[0],coloraxes[0],photdensity)]:
        if( name=='spec') and not ( specscale is None):
            top=specscale
        elif( name=='phot') and not ( photscale is None):
            top=photscale
        else:
            top=datacomb.max()
        i=0
        plt.sca(ax)
        scale=0,top
        colorax.axis('off')
        #normalize=colors.SymLogNorm(linthresh=scale[1]**.5, linscale=1, vmin=scale[0], vmax=scale[1], base=10)
        normalize=colors.Normalize(0,top)
        image=plt.imshow(datacomb,extent=[0,5,5,0],norm=normalize,cmap=discontmap,interpolation='nearest')
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
    for ax in axes:
        ticklocs=np.linspace(0,5,4,True)
        ax.set_xticks(ticklocs)
        ax.set_xticklabels(labels=[f'{tick*(wavebins[-1]-wavebins[0])/(ticklocs[-1])+wavebins[0]:.0f}' for tick in ticklocs],fontsize=ticksize,rotation=-45)



    axes[-1].set_xlabel('Wavelength ($\mathrm{\AA}$)',fontsize=labelsize)
    axes[0].set_ylabel('Phase (days)',fontsize=labelsize,verticalalignment='top')
    axes[0].yaxis.set_label_coords(-0.5,0)
    for ax in axes:
        ticklocs=np.linspace(0,5,5,True)
        ax.set_yticks(ticklocs)
        ax.set_yticklabels(labels=[f'{tick*(phasebins[-1]-phasebins[0])/(ticklocs[-1])+phasebins[0]:.0f}' for tick in ticklocs],fontsize=ticksize)

    #plt.subplots_adjust(top=.9,bottom=.15,hspace=.1)

    #plt.tight_layout()

    #     plt.ylabel('Phase (days)',fontsize=labelsize)
    #     plt.xlabel('Wavelength ($\mathrm{\AA}$)',fontsize=labelsize)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    return photdensity,specdensity
