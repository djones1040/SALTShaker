#!/usr/bin/env python
# D. Jones - 8/5/20
import pylab as plt
from matplotlib.gridspec import GridSpec
from saltshaker.util import snana
from saltshaker.validation.ValidateSpectra import flux
import numpy as np
plt.ion()
import glob
import os
from txtobj import txtobj
from scipy.special import factorial
from scipy.stats import binned_statistic
from scipy.optimize import minimize, least_squares
parname,parval = np.loadtxt('output/salt3_parameters.dat',unpack=True,dtype=str,skiprows=1)
snid,pkmjd = np.loadtxt('SALT3_PKMJD_INIT.LIST',unpack=True,dtype=str,usecols=[0,1])
parval,pkmjd = parval.astype(float),pkmjd.astype(float)
plt.rcParams['figure.figsize'] = (10,10)
colordict = {'SALT2':'#009988', # teal
             'SALT3':'#EE7733',  # orange
}


def main():

    fig = plt.figure()
    gs_top = GridSpec(5, 3, figure=fig,hspace=0,wspace=0,top=0.9)
    gs = GridSpec(5, 3, figure=fig,hspace=0,top=0.75,bottom=0.1)
    #gs_bottom = GridSpec(5, 3, figure=fig,hspace=0,top=0.95)
    axes = []
    for i in range(3):
        for j in range(3):
            axes += [fig.add_subplot(gs_top[i, j])]
            #if i != 0: 
            #    axes[-1].xaxis.set_ticklabels([])
            if i != 2:
                axes[-1].xaxis.set_ticklabels([])
            else:
                axes[-1].xaxis.set_ticks([4000,5000,6000,7000,8000])
                
            if j == 0: axes[-1].set_ylabel('Flux')
            if i == 2: axes[-1].set_xlabel('Wavelength ($\mathrm{\AA}$)')
            axes[-1].set_xlim([3000,9000])
            axes[-1].yaxis.set_ticks([])
            
    for ax in axes:
        ax.tick_params(top="on",bottom="on",left="off",right="off",direction="inout",length=8, width=1.5)
    axfound = fig.add_subplot(gs[3,:])
    axall = fig.add_subplot(gs[4,:])
    axall.set_xlabel('Wavelength ($\mathrm{\AA}$)')
    axall.set_ylabel('Resid (%)')
    axfound.set_ylabel('Resid (%)')
    axall.text(0.5,0.9,'All Spectra',transform=axall.transAxes,ha='center',va='center')
    axfound.text(0.5,0.9,'Foundation',transform=axfound.transAxes,ha='center',va='center')
    axall.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
    axfound.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
    axfound.xaxis.set_ticklabels([])
    
    foundfiles = glob.glob('SALT3_training_data/Foundation_DR1_spec*.txt')
    allfiles = np.loadtxt('SALT3_training_data/SALT3_training.LIST',unpack=True,dtype=str)
    allfiles = np.array([f'SALT3_training_data/{a}' for a in allfiles])
    count = 0
    for i,ax in enumerate(axes):
        for f in foundfiles[count:]:
            sn = snana.SuperNova(f)
            count += 1
            if len(list(sn.SPECTRA.keys())) and len(parval[parname == f'specx0_{sn.SNID}_0']): break
        wave = (sn.SPECTRA[0]['LAMMIN']+sn.SPECTRA[0]['LAMMAX'])/2.
        dataflux = sn.SPECTRA[0]['FLAM']
        zHel = float(sn.REDSHIFT_HELIO.split('+-')[0])
        restwave=wave/(1+zHel)

        x0,x1,c = parval[parname == f'specx0_{sn.SNID}_0'][0],parval[parname == f'x1_{sn.SNID}'][0],parval[parname == f'c_{sn.SNID}'][0]
        t0 = pkmjd[sn.SNID == snid][0]
        
        coeffs=parval[parname=='specrecal_{}_{}'.format(sn.SNID,0)]
        pow=coeffs.size-np.arange(coeffs.size)
        recalCoord=(wave-np.mean(wave))/2500
        drecaltermdrecal=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
        recalexp=np.exp((drecaltermdrecal*coeffs[np.newaxis,:]).sum(axis=1))

        uncalledModel = flux('output',sn.SPECTRA[0]['SPECTRUM_MJD']-t0,wave,
                             zHel,x0,x1,c,mwebv=float(sn.MWEBV.split()[0]),fill_value=None)
        modelflux = uncalledModel*recalexp

        ax.text(0.5,0.85,f"{sn.SNID}, phase={sn.SPECTRA[0]['SPECTRUM_MJD']-t0:.1f}",
                transform=ax.transAxes,ha='center',va='center',bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.5},zorder=1000)
        ax.plot(wave,dataflux,'k-',label='data')
        ax.plot(wave,uncalledModel,'-',color='0.7',label='SALT3 Model spectrum\n(no calibration)')
        if len(coeffs): ax.plot(wave,modelflux,'-',
                                color=colordict['SALT3'],label='SALT3 Model spectrum\n(recalibrated)',zorder=500)

        #ax.set_xlim(wave.min(),wave.max())

        ax.set_ylim(0,dataflux.max()*1.25)
        #ax.set_xlabel('Wavelength $\AA$')
        #ax.set_ylabel('Flux')
        
    axes[1].legend(loc='upper center',prop={'size':12},bbox_to_anchor=(0.5,1.5),ncol=3)
    
    # now for everything
    # gonna be slow.....
    # SALT3
    wavebins = np.linspace(2000,11000,180)
    resids_full = np.array([])
    count = 0
    for i in range(600):
        for f in foundfiles[count:]:
            sn = snana.SuperNova(f)
            count += 1
            if len(list(sn.SPECTRA.keys())) and len(parval[parname == f'specx0_{sn.SNID}_0']): break
            if count >= len(foundfiles): break
        if count >= len(foundfiles): break
        
        wave = (sn.SPECTRA[0]['LAMMIN']+sn.SPECTRA[0]['LAMMAX'])/2.
        dataflux = sn.SPECTRA[0]['FLAM']
        zHel = float(sn.REDSHIFT_HELIO.split('+-')[0])
        restwave=wave/(1+zHel)

        x0,x1,c = parval[parname == f'specx0_{sn.SNID}_0'][0],parval[parname == f'x1_{sn.SNID}'][0],parval[parname == f'c_{sn.SNID}'][0]
        t0 = pkmjd[sn.SNID == snid][0]
        
        coeffs=parval[parname=='specrecal_{}_{}'.format(sn.SNID,0)]
        pow=coeffs.size-np.arange(coeffs.size)
        recalCoord=(wave-np.mean(wave))/2500
        drecaltermdrecal=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
        recalexp=np.exp((drecaltermdrecal*coeffs[np.newaxis,:]).sum(axis=1))

        uncalledModel = flux('output',sn.SPECTRA[0]['SPECTRUM_MJD']-t0,wave,
                             zHel,x0,x1,c,mwebv=float(sn.MWEBV.split()[0]),fill_value=None)
        modelflux = uncalledModel*recalexp
        resids = binned_statistic(restwave,dataflux-modelflux,bins=wavebins,statistic='median').statistic/np.max(modelflux)
        if not len(resids_full):
            resids_full = resids[:]
        else:
            resids_full = np.vstack((resids_full,resids))

    # SALT2
    fr = txtobj('fit_output/Foundation_DR1_SALT2.FITRES.TEXT',fitresheader=True)
    resids_full_salt2 = np.array([])
    count = 0
    for i in range(600):
        for f in foundfiles[count:]:
            sn = snana.SuperNova(f)
            count += 1
            if len(list(sn.SPECTRA.keys())) and len(fr.x0[fr.CID == sn.SNID]): break
            if count >= len(foundfiles): break
        if count >= len(foundfiles): break
        
        wave = (sn.SPECTRA[0]['LAMMIN']+sn.SPECTRA[0]['LAMMAX'])/2.
        dataflux = sn.SPECTRA[0]['FLAM']
        zHel = float(sn.REDSHIFT_HELIO.split('+-')[0])
        restwave=wave/(1+zHel)

        x0,x1,c = fr.x0[fr.CID == sn.SNID][0],fr.x1[fr.CID == sn.SNID][0],fr.c[fr.CID == sn.SNID][0]
        t0 = fr.PKMJD[fr.CID == sn.SNID][0]
        
        coeffs=parval[parname=='specrecal_{}_{}'.format(sn.SNID,0)]
        pow=coeffs.size-np.arange(coeffs.size)
        recalCoord=(wave-np.mean(wave))/2500
        drecaltermdrecal=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
        uncalledModel = flux(os.path.expandvars('$SNDATA_ROOT/models/SALT2/SALT2.JLA-B14'),sn.SPECTRA[0]['SPECTRUM_MJD']-t0,wave,
                             zHel,x0,x1,c,mwebv=float(sn.MWEBV.split()[0]),fill_value=None)
        #if np.max(restwave) > 9200: import pdb; pdb.set_trace()
        
        def recalpars(x):
            recalexp=np.exp((drecaltermdrecal*x[1:][np.newaxis,:]).sum(axis=1))
            return x[0]*uncalledModel*recalexp - dataflux
        #import pdb; pdb.set_trace()
        try:
            md = least_squares(recalpars,[np.median(dataflux)/np.median(uncalledModel)]+list(coeffs))
            recalexp=np.exp((drecaltermdrecal*md.x[1:][np.newaxis,:]).sum(axis=1))
        except:
            continue
            
        modelflux = md.x[0]*uncalledModel*recalexp
        resids = binned_statistic(restwave,dataflux-modelflux,bins=wavebins,statistic='median').statistic/np.max(modelflux)

        if not len(resids_full_salt2):
            resids_full_salt2 = resids[:]
        else:
            resids_full_salt2 = np.vstack((resids_full_salt2,resids))

    axfound.plot((wavebins[1:]+wavebins[:-1])/2.,np.nanmedian(resids_full_salt2,axis=0),color=colordict['SALT2'])
    axfound.fill_between((wavebins[1:]+wavebins[:-1])/2.,np.nanmedian(resids_full_salt2,axis=0)-np.nanstd(resids_full_salt2,axis=0),
                         np.nanmedian(resids_full_salt2,axis=0)+np.nanstd(resids_full_salt2,axis=0),color=colordict['SALT2'],alpha=0.3)
            
    axfound.plot((wavebins[1:]+wavebins[:-1])/2.,np.nanmedian(resids_full,axis=0),color=colordict['SALT3'])
    axfound.fill_between((wavebins[1:]+wavebins[:-1])/2.,np.nanmedian(resids_full,axis=0)-np.nanstd(resids_full,axis=0),
                         np.nanmedian(resids_full,axis=0)+np.nanstd(resids_full,axis=0),color=colordict['SALT3'],alpha=0.3)
    axfound.axhline(0,color='k',lw=2)
    axfound.set_xlim([2000,11000])

            
    # now for everything
    count = 0
    for i in range(600):
        for f in allfiles[count:]:
            if 'Foundation' in f:
                count += 1
                continue
            sn = snana.SuperNova(f)
            count += 1
            if len(list(sn.SPECTRA.keys())) and len(parval[parname == f'specx0_{sn.SNID}_0']): break
            if count >= len(allfiles): break
        if count >= len(allfiles): break
        
        wave = (sn.SPECTRA[0]['LAMMIN']+sn.SPECTRA[0]['LAMMAX'])/2.
        dataflux = sn.SPECTRA[0]['FLAM']
        zHel = float(sn.REDSHIFT_HELIO.split('+-')[0])
        restwave=wave/(1+zHel)

        x0,x1,c = parval[parname == f'specx0_{sn.SNID}_0'][0],parval[parname == f'x1_{sn.SNID}'][0],parval[parname == f'c_{sn.SNID}'][0]
        t0 = pkmjd[sn.SNID == snid][0]
        
        coeffs=parval[parname=='specrecal_{}_{}'.format(sn.SNID,0)]
        pow=coeffs.size-np.arange(coeffs.size)
        recalCoord=(wave-np.mean(wave))/2500
        drecaltermdrecal=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
        recalexp=np.exp((drecaltermdrecal*coeffs[np.newaxis,:]).sum(axis=1))

        uncalledModel = flux('output',sn.SPECTRA[0]['SPECTRUM_MJD']-t0,wave,
                             zHel,x0,x1,c,mwebv=float(sn.MWEBV.split()[0]),fill_value=None)
        modelflux = uncalledModel*recalexp
        resids = binned_statistic(restwave,dataflux-modelflux,bins=wavebins,statistic='median').statistic/np.max(modelflux)

        if not len(resids_full):
            resids_full = resids[:]
        else:
            resids_full = np.vstack((resids_full,resids))

    # SALT2
    fr = txtobj('fit_output/JLA_TRAINING_SALT2.FITRES',fitresheader=True)
    resids_full_salt2 = np.array([])
    count = 0
    for i in range(600):
        for f in allfiles[count:]:
            if 'Foundation' in f:
                count += 1
                continue
            sn = snana.SuperNova(f)
            count += 1
            if len(list(sn.SPECTRA.keys())) and len(fr.x0[fr.CID == sn.SNID]): break
            if count >= len(allfiles): break
        if count >= len(allfiles): break
        
        wave = (sn.SPECTRA[0]['LAMMIN']+sn.SPECTRA[0]['LAMMAX'])/2.
        dataflux = sn.SPECTRA[0]['FLAM']
        zHel = float(sn.REDSHIFT_HELIO.split('+-')[0])
        restwave=wave/(1+zHel)

        x0,x1,c = fr.x0[fr.CID == sn.SNID][0],fr.x1[fr.CID == sn.SNID][0],fr.c[fr.CID == sn.SNID][0]
        t0 = fr.PKMJD[fr.CID == sn.SNID][0]
        
        coeffs=parval[parname=='specrecal_{}_{}'.format(sn.SNID,0)]
        pow=coeffs.size-np.arange(coeffs.size)
        recalCoord=(wave-np.mean(wave))/2500
        drecaltermdrecal=((recalCoord)[:,np.newaxis] ** (pow)[np.newaxis,:]) / factorial(pow)[np.newaxis,:]
        uncalledModel = flux(os.path.expandvars('$SNDATA_ROOT/models/SALT2/SALT2.JLA-B14'),sn.SPECTRA[0]['SPECTRUM_MJD']-t0,wave,
                             zHel,x0,x1,c,mwebv=float(sn.MWEBV.split()[0]),fill_value=None)

        
        def recalpars(x):
            recalexp=np.exp((drecaltermdrecal*x[1:][np.newaxis,:]).sum(axis=1))
            return x[0]*uncalledModel*recalexp - dataflux
        #import pdb; pdb.set_trace()
        try:
            md = least_squares(recalpars,[np.median(dataflux)/np.median(uncalledModel)]+list(coeffs))
            recalexp=np.exp((drecaltermdrecal*md.x[1:][np.newaxis,:]).sum(axis=1))
        except:
            continue
            
        modelflux = md.x[0]*uncalledModel*recalexp
        
        resids = binned_statistic(restwave,dataflux-modelflux,bins=wavebins,statistic='median').statistic/np.max(modelflux)
        if not len(resids_full_salt2):
            resids_full_salt2 = resids[:]
        else:
            resids_full_salt2 = np.vstack((resids_full_salt2,resids))
    
            
    axall.plot((wavebins[1:]+wavebins[:-1])/2.,np.nanmedian(resids_full_salt2,axis=0),color=colordict['SALT2'])
    axall.fill_between((wavebins[1:]+wavebins[:-1])/2.,np.nanmedian(resids_full_salt2,axis=0)-np.nanstd(resids_full_salt2,axis=0),
                         np.nanmedian(resids_full_salt2,axis=0)+np.nanstd(resids_full_salt2,axis=0),color=colordict['SALT2'],alpha=0.3)
            
    axall.plot((wavebins[1:]+wavebins[:-1])/2.,np.nanmedian(resids_full,axis=0),color=colordict['SALT3'])
    axall.fill_between((wavebins[1:]+wavebins[:-1])/2.,np.nanmedian(resids_full,axis=0)-np.nanstd(resids_full,axis=0),
                         np.nanmedian(resids_full,axis=0)+np.nanstd(resids_full,axis=0),color=colordict['SALT3'],alpha=0.3)
    axall.axhline(0,color='k',lw=2)
    axall.set_xlim([2000,11000])
    axfound.set_ylim([-0.4,0.4])
    axall.set_ylim([-0.4,0.4])
    
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
