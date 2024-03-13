#!/usr/bin/env python

import numpy as np
import pylab as plt
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
import extinction
import copy
from astropy.io import fits
from saltshaker.initfiles import init_rootdir as salt2dir
_SCALE_FACTOR = 1e-12

#filtdict = {'b':'cspb','c':'cspv3014','d':'cspr','e':'cspi'}
filtdict = {'J':'J','H':'H',
            'Y':'Y',
            'a':'Jrc2',
            'b':'Jrc1',
            'c':'Ydw',
            'd':'Jdw',
            'e':'Hdw',
            'f':'J2m',
            'g':'H2m',
            'l':'Ks2m',
            'm':'JANDI',
            'n':'HANDI',
            'o':'F125W',
            'p':'F160W'}

def main(outfile,lcfile,salt3dir,
         m0file='salt3_template_0.dat',
         m1file='salt3_template_1.dat',
         clfile='salt2_color_correction.dat',
         cdfile='salt2_color_dispersion.dat',
         errscalefile='salt2_lc_dispersion_scaling.dat',
         lcrv00file='salt2_lc_relative_variance_0.dat',
         lcrv11file='salt2_lc_relative_variance_1.dat',
         lcrv01file='salt2_lc_relative_covariance_01.dat',
         x0 = None, x1 = None, c = None, t0 = None,
         fitx1=False,fitc=False,bandpassdict=None):
    bandpassdict = None
    
    plt.clf()

    fitparams_salt3 = []
    if not t0: fitparams_salt3 += ['t0']
    if not x0: fitparams_salt3 += ['x0']
    if not x1 and fitx1: fitparams_salt3 += ['x1']
    if not c and fitc: fitparams_salt3 += ['c']

    sn = snana.SuperNova(lcfile)
    try:
        sn.REDSHIFT_HELIO = float(sn.REDSHIFT_HELIO.split('+-')[0])
    except:
        ### might be a float already
        pass
    try: sn.FLT = sn.FLT.astype('U20')
    except: sn.FLT = sn.BAND.astype('U20')
    
    if bandpassdict:
        bandlist = []
        for k in bandpassdict.keys():
            band = sncosmo.Bandpass(
                bandpassdict[k]['filtwave'],
                bandpassdict[k]['filttrans'],
                wave_unit=u.angstrom,name=k)
            sncosmo.register(band, k, force=True)
    else:
        for i in range(len(sn.FLT)):
            if sn.FLT[i] in filtdict.keys():
                sn.FLT[i] = filtdict[sn.FLT[i]]
            elif sn.FLT[i] in 'griz':
                sn.FLT[i] = 'sdss%s'%sn.FLT[i]
            elif sn.FLT[i].lower() == 'v':
                sn.FLT[i] = 'swope2::v'
            else:
                sn.FLT[i] = 'csp%s'%sn.FLT[i].lower()

    zpsys='AB'
    data = Table([sn.MJD,sn.FLT,sn.FLUXCAL,sn.FLUXCALERR,
                  np.array([27.5]*len(sn.MJD)),np.array([zpsys]*len(sn.MJD))],
                 names=['mjd','band','flux','fluxerr','zp','zpsys'],
                 meta={'t0':sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]})
    
    flux = sn.FLUXCAL
    salt2model = sncosmo.Model(source='salt2')
    hsiaomodel = sncosmo.Model(source='hsiao')
    salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,
                                m1file=m1file,
                                clfile=clfile,cdfile=cdfile,
                                errscalefile=errscalefile,
                                lcrv00file=lcrv00file,
                                lcrv11file=lcrv11file,
                                lcrv01file=lcrv01file)
    salt3model =  sncosmo.Model(salt3)
    salt3model.set(z=sn.REDSHIFT_HELIO)
    fitparams_salt2=['t0', 'x0', 'x1', 'c']
    salt2model.set(z=sn.REDSHIFT_HELIO)
    result_salt2, fitted_salt2_model = sncosmo.fit_lc(data, salt2model, fitparams_salt2)
    fitparams_hsiao = ['t0','amplitude']
    hsiaomodel.set(z=sn.REDSHIFT_HELIO)
    result_hsiao, fitted_hsiao_model = sncosmo.fit_lc(data, hsiaomodel, fitparams_hsiao)

    salt3model.set(z=sn.REDSHIFT_HELIO)
    if x0: salt3model.set(x0=x0)
    if t0: salt3model.set(t0=t0)
    if x1: salt3model.set(x1=x1)
    if c: salt3model.set(c=c)
    if len(fitparams_salt3):
        result_salt3, fitted_salt3_model = sncosmo.fit_lc(data, salt3model, fitparams_salt3)
    else:
        fitted_salt3_model = salt3model
    plotmjd = np.linspace(sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]-20,
                          sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]+55,100)
    
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    for flt,i,ax in zip(np.unique(sn.FLT),range(3),[ax1,ax2,ax3]):
        try:
            hsiaoflux = fitted_hsiao_model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            salt2flux = fitted_salt2_model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
            salt3flux = fitted_salt3_model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')#*\
                #10**(-0.4*bandpassdict[flt]['zpoff'])*10**(0.4*bandpassdict[flt]['stdmag'])
        except:
            print('Warning : error for band %s'%flt)
            continue
        ax.plot(plotmjd,hsiaoflux,color='C0',
                label='Hsiao, $\chi^2_{red} = %.1f$'%(
                    result_hsiao['chisq']/result_hsiao['ndof']))
        ax.plot(plotmjd,salt2flux,color='C1',
                label='SALT2, $\chi^2_{red} = %.1f$'%(
                    result_salt2['chisq']/result_salt2['ndof']))
        if len(fitparams_salt3):
            ax.plot(plotmjd,salt3flux,color='C2',
                    label='SALT3, $\chi^2_{red} = %.1f$'%(
                        result_salt3['chisq']/result_salt3['ndof']))
        else:
            ax.plot(plotmjd,salt3flux*1e10,color='C2',
                    label='SALT3')

        ax.errorbar(sn.MJD[sn.FLT == flt],sn.FLUXCAL[sn.FLT == flt],
                    yerr=sn.FLUXCALERR[sn.FLT == flt],
                    fmt='o',label=sn.SNID,color='k')
        ax.set_title(flt)
        ax.set_xlim([sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]-30,
                     sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]+55])
        ax.set_ylim([-np.max(sn.FLUXCAL)*1/20.,np.max(sn.FLUXCAL)*1.1])
#       import pdb; pdb.set_trace()
    ax1.legend()
    plt.savefig(outfile)
    plt.show()
    plt.close(fig)

def customfilt(outfile,lcfile,salt3dir,
               m0file='salt3_template_0.dat',
               m1file='salt3_template_1.dat',
               clfile='salt3_color_correction.dat',
               cdfile='salt3_color_dispersion.dat',
               errscalefile='salt3_lc_dispersion_scaling.dat',
               lcrv00file='salt3_lc_variance_0.dat',
               lcrv11file='salt3_lc_variance_1.dat',
               lcrv01file='salt3_lc_covariance_01.dat',
               flatnu='flatnu.dat',
               x0 = None, x1 = None, c = None, t0 = None,
               fitx1=False,fitc=False,bandpassdict=None,
               n_components=1, ax1=None, ax2=None, ax3=None, ax4=None, ax5=None,
               saltdict={},n_colorpars=4,snid=None):

    salt2_chi2tot,salt3_chi2tot = 0,0
    
    if not ax1:
        plt.clf()
    
    fitparams_salt3 = []
    if not t0: fitparams_salt3 += ['t0']
    if not x0: fitparams_salt3 += ['x0']
    if not x1 and fitx1: fitparams_salt3 += ['x1']
    if not c and fitc: fitparams_salt3 += ['c']
    
    if not '.fits' in lcfile.lower():
        sn = snana.SuperNova(lcfile)
        try:
            sn.REDSHIFT_HELIO = float(sn.REDSHIFT_HELIO.split('+-')[0])
        except:
            pass
        if type(sn.MWEBV)==str: sn.MWEBV = sn.MWEBV.split()[0]
    else:
        sn = snana.SuperNova(snid=snid,headfitsfile=lcfile,photfitsfile=lcfile.replace('_HEAD.FITS','_PHOT.FITS'),
                             specfitsfile=None,readspec=False)
        survey = fits.getval( lcfile, 'SURVEY')
        if 'SUBSURVEY' in sn.__dict__.keys() and not (len(np.unique(sn.SUBSURVEY))==1 and survey.strip()==np.unique(sn.SUBSURVEY)[0].strip()) \
           and sn.SUBSURVEY.strip() != '':
            sn.SURVEY = f"{survey}({sn.SUBSURVEY})"
        else:
            sn.SURVEY = survey

    try: sn.FLT = sn.FLT.astype('U20')
    except: sn.FLT = sn.BAND.astype('U20')

    zpsys='AB'
    if 'PEAKMJD' in sn.__dict__.keys():
        data = Table(rows=None,names=['mjd','band','flux','fluxerr','zp','zpsys'],
                     dtype=('f8','S1','f8','f8','f8','U5'),
                     meta={'t0':sn.PEAKMJD})
    else:
        data = Table(rows=None,names=['mjd','band','flux','fluxerr','zp','zpsys'],
                     dtype=('f8','S1','f8','f8','f8','U5'),
                     meta={'t0':sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]})

    for flt in np.unique(sn.FLT):
        filtwave = bandpassdict[sn.SURVEY][flt]['filtwave']
        filttrans = bandpassdict[sn.SURVEY][flt]['filttrans']

        band = sncosmo.Bandpass(
            filtwave,
            filttrans,
            wave_unit=u.angstrom,name=flt)
        sncosmo.register(band, force=True)

    sysdict = {}
    for m,flt,flx,flxe in zip(sn.MJD,sn.FLT,sn.FLUXCAL,sn.FLUXCALERR):
        if bandpassdict[sn.SURVEY][flt]['magsys'] == 'BD17': sys = 'bd17'
        elif bandpassdict[sn.SURVEY][flt]['magsys'] == 'AB': sys = 'ab'
        else: sys = 'vega'
        if bandpassdict[sn.SURVEY][flt]['lambdaeff']/(1+sn.REDSHIFT_HELIO) > 2800 and \
           bandpassdict[sn.SURVEY][flt]['lambdaeff']/(1+sn.REDSHIFT_HELIO) < 9000 and\
           '-u' not in bandpassdict[sn.SURVEY][flt]['fullname']:
            data.add_row((m,flt,flx*10**(0.4*bandpassdict[sn.SURVEY][flt]['primarymag']),
                          flxe*10**(0.4*bandpassdict[sn.SURVEY][flt]['primarymag']),
                          27.5,sys)) #+bandpassdict[sn.SURVEY][flt]['zpoff']
                
                
        sysdict[flt] = sys
        
    flux = sn.FLUXCAL
    salt2source = sncosmo.SALT2Source(modeldir=salt2dir)
    dust = sncosmo.F99Dust()
    salt2model = sncosmo.Model(salt2source,effects=[dust],effect_names=['mw'],effect_frames=['obs'])
    hsiaomodel = sncosmo.Model(source='hsiao')
    salt2model = sncosmo.Model(source='salt2',effects=[dust],effect_names=['mw'],effect_frames=['obs'])
    
    if not len(list(saltdict.keys())):
        salt3phase,salt3wave,salt3flux = np.genfromtxt('%s/%s'%(salt3dir,m0file),unpack=True)
        salt3m1phase,salt3m1wave,salt3m1flux = np.genfromtxt('%s/%s'%(salt3dir,m1file),unpack=True)
        salt2phase,salt2wave,salt2flux = np.genfromtxt('{}/salt2_template_0.dat'.format(salt2dir),unpack=True)
        salt2m1phase,salt2m1wave,salt2m1flux = np.genfromtxt('{}/salt2_template_1.dat'.format(salt2dir),unpack=True)

        salt3flux = salt3flux.reshape([len(np.unique(salt3phase)),len(np.unique(salt3wave))])
        salt3m1flux = salt3m1flux.reshape([len(np.unique(salt3phase)),len(np.unique(salt3wave))])
        salt3phase = np.unique(salt3phase)*(1+sn.REDSHIFT_HELIO)
        salt3wave = np.unique(salt3wave)*(1+sn.REDSHIFT_HELIO)

        salt2m0flux = salt2flux.reshape([len(np.unique(salt2phase)),len(np.unique(salt2wave))])
        salt2flux = salt2flux.reshape([len(np.unique(salt2phase)),len(np.unique(salt2wave))])
        salt2m1flux = salt2m1flux.reshape([len(np.unique(salt2phase)),len(np.unique(salt2wave))])
        salt2phase = np.unique(salt2phase)*(1+sn.REDSHIFT_HELIO)
        salt2wave = np.unique(salt2wave)*(1+sn.REDSHIFT_HELIO)

    else:
        salt3phase,salt3wave,salt3flux = copy.deepcopy(saltdict['salt3phase']),copy.deepcopy(saltdict['salt3wave']),copy.deepcopy(saltdict['salt3flux'])
        salt3m1phase,salt3m1wave,salt3m1flux = copy.deepcopy(saltdict['salt3m1phase']),copy.deepcopy(saltdict['salt3m1wave']),copy.deepcopy(saltdict['salt3m1flux'])
        salt2phase,salt2wave,salt2m0flux = copy.deepcopy(saltdict['salt2phase']),copy.deepcopy(saltdict['salt2wave']),copy.deepcopy(saltdict['salt2m0flux'])
        salt2m1phase,salt2m1wave,salt2m1flux = copy.deepcopy(saltdict['salt2m1phase']),copy.deepcopy(saltdict['salt2m1wave']),copy.deepcopy(saltdict['salt2m1flux'])
        salt3phase *= 1+sn.REDSHIFT_HELIO; salt3wave *= 1+sn.REDSHIFT_HELIO
        salt2phase *= 1+sn.REDSHIFT_HELIO; salt2wave *= 1+sn.REDSHIFT_HELIO

        salt2phase_tmp,salt2wave_tmp,salt2flux_tmp = np.genfromtxt('{}/salt2_template_0.dat'.format(salt2dir),unpack=True)
        #salt2m0flux = salt2flux.reshape([len(np.unique(salt2phase)),len(np.unique(salt2wave))])
        salt2flux_tmp = salt2flux_tmp.reshape([len(np.unique(salt2phase_tmp)),len(np.unique(salt2wave_tmp))])
        #salt2m1phase,salt2m1wave,salt2m1flux = np.genfromtxt('{}/salt2_template_1.dat'.format(salt2dir),unpack=True)
        #salt2phase = np.interp(salt3wave,salt2wave,salt2phase)
        #salt2flux = np.interp(salt3wave,salt2wave,salt2flux)
        from scipy.interpolate import interp1d
        int1dwave = interp1d(np.unique(salt2wave_tmp),salt2flux_tmp,axis=1,fill_value="extrapolate")
        salt2m0flux_tmp = int1dwave(salt3wave)
        int1dphase = interp1d(np.unique(salt2phase_tmp),salt2m0flux_tmp,axis=0,fill_value="extrapolate")
        salt2flux_tmp = int1dphase(np.unique(salt3phase))

        
    # color laws
    #try:
    with open('%s/%s'%(salt3dir,clfile)) as fin:
        lines = fin.readlines()
    if len(lines):
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n','')
        colorlaw_salt3_coeffs = np.array(lines[1:n_colorpars[0]+1]).astype('float')
        salt3_colormin = float(lines[n_colorpars[0]+2].split()[1])
        salt3_colormax = float(lines[n_colorpars[0]+3].split()[1])

        salt3colorlaw = SALT2ColorLaw([salt3_colormin,salt3_colormax],colorlaw_salt3_coeffs)
    #except:
#       pass
    salt2colorlaw = SALT2ColorLaw([2800,7000], [-0.504294,0.787691,-0.461715,0.0815619])

    #print(np.sum(salt2flux_tmp[20,:]),np.sum(salt3flux[20,:]))
    if n_components == 1: salt3flux = x0*salt3flux
    elif n_components == 2: salt3flux = x0*(salt3flux + x1*salt3m1flux)
    if c:
        salt3flux *= 10. ** (-0.4 * salt3colorlaw(salt3wave/(1+sn.REDSHIFT_HELIO)) * c)
    salt3flux *= _SCALE_FACTOR

    #if 'SIM_SALT2x0' in sn.__dict__.keys():
    #   salt2flux = sn.SIM_SALT2x0*(salt2m0flux*_SCALE_FACTOR + (sn.SIM_SALT2x1)*salt2m1flux*_SCALE_FACTOR) * \
    #               10. ** (-0.4 * salt2colorlaw(salt2wave/(1+float(sn.SIM_REDSHIFT_HELIO))) * float(sn.SIM_SALT2c))
    #   try: mwextcurve = 10**(-0.4*extinction.fitzpatrick99(salt2wave,float(sn.MWEBV.split()[0])*3.1))
    #   except: mwextcurve = 10**(-0.4*extinction.fitzpatrick99(salt2wave,sn.MWEBV*3.1))
    #   salt2flux *= mwextcurve[np.newaxis,:]
    #else:
    salt2model.set(z=sn.REDSHIFT_HELIO,mwebv=sn.MWEBV)
    fitparams = ['t0', 'x0', 'x1', 'c']

    try:
        result, fitted_model = sncosmo.fit_lc(
            data, salt2model, fitparams,
            bounds={'t0':(t0-10, t0+10),
                    'z':(0.0,0.7),'x1':(-3,3),'c':(-0.3,0.3)})
        has_salt2 = True
    except: has_salt2 = False
    salt3 = sncosmo.SALT2Source(modeldir=salt3dir,m0file=m0file,
                                m1file=m1file,
                                clfile=clfile,cdfile=cdfile,
                                errscalefile=errscalefile,
                                lcrv00file=lcrv00file,
                                lcrv11file=lcrv11file,
                                lcrv01file=lcrv01file)
    
    plotmjd = np.linspace(t0-20,
                          t0+55,200)
    
    if not ax1:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(151)
        ax2 = fig.add_subplot(152)
        ax3 = fig.add_subplot(153)
        ax4 = fig.add_subplot(154)
        ax5 = fig.add_subplot(155)
        
    int1d = interp1d(salt3phase,salt3flux,axis=0,fill_value='extrapolate')
    for flt,i,ax in zip(np.unique(sn.FLT),range(5),[ax1,ax2,ax3,ax4,ax5]):
        phase=plotmjd-t0
        salt3fluxnew = int1d(phase)
        try: mwextcurve = 10**(-0.4*extinction.fitzpatrick99(salt3wave,float(sn.MWEBV)*3.1))
        except: mwextcurve = 10**(-0.4*extinction.fitzpatrick99(salt3wave,sn.MWEBV*3.1))
        salt3fluxnew *= mwextcurve[np.newaxis,:]

        filtwave = bandpassdict[sn.SURVEY][flt]['filtwave']
        filttrans = bandpassdict[sn.SURVEY][flt]['filttrans']

        g = (salt3wave >= filtwave[0]) & (salt3wave <= filtwave[-1])  # overlap range
        pbspl = np.interp(salt3wave[g],filtwave,filttrans)
        pbspl *= salt3wave[g]
        deltawave = salt3wave[g][1]-salt3wave[g][0]
        denom = np.sum(pbspl)*deltawave
        salt3synflux=np.sum(pbspl[np.newaxis,:]*salt3fluxnew[:,g],axis=1)*deltawave/HC_ERG_AA/denom
        salt3synflux *= 10**(0.4*bandpassdict[sn.SURVEY][flt]['stdmag'])*10**(0.4*27.5)/(1+sn.REDSHIFT_HELIO)
        

        iFLT = (sn.FLT == flt) & (sn.MJD-t0 > -20) & (sn.MJD-t0 < 50)
        chi2_salt3 = np.sum((sn.FLUXCAL[iFLT]-\
                             np.interp(sn.MJD[iFLT],plotmjd,salt3synflux))**2./sn.FLUXCALERR[iFLT]**2.)
        chi2red_salt3 = chi2_salt3/(len(sn.FLUXCAL[(sn.FLT == flt) & (sn.MJD-t0 > -20) & (sn.MJD-t0 < 50)])-3)
        ax.plot(plotmjd-t0,salt3synflux,color='C2',
                label='SALT3, $x_0$ = %8.5e, \nx1=%.2f, z=%.3f\nc=%.3f\n$\chi_{red}^2=%.1f$'%(
                    x0,x1,sn.REDSHIFT_HELIO,c,chi2red_salt3))
        salt3_chi2tot += chi2_salt3
        if bandpassdict[sn.SURVEY][flt]['lambdaeff']/(1+sn.REDSHIFT_HELIO) > 2800 and \
           bandpassdict[sn.SURVEY][flt]['lambdaeff']/(1+sn.REDSHIFT_HELIO) < 9000:

            try:
                if has_salt2:
                    chi2 = np.sum((sn.FLUXCAL[iFLT]-fitted_model.bandflux(
                        flt, sn.MJD[iFLT], zp=27.5-bandpassdict[sn.SURVEY][flt]['primarymag'],zpsys=sysdict[flt]))**2./sn.FLUXCALERR[iFLT]**2.)
                    chi2red = chi2/(len(sn.FLUXCAL[iFLT])-3)
                    ax.plot(plotmjd-t0,fitted_model.bandflux(
                        flt, plotmjd, zp=27.5-bandpassdict[sn.SURVEY][flt]['primarymag'],zpsys=sysdict[flt]),color='C1',
                            label='SALT2; $x_1 = %.2f$, $c = %.2f$,\n$\chi_{red}^2 = %.1f$'%(
                                result['parameters'][3],result['parameters'][4],chi2red))
                    salt2_chi2tot += chi2
            except ValueError: pass

        ax.errorbar(sn.MJD[sn.FLT == flt]-t0,sn.FLUXCAL[sn.FLT == flt],
                    yerr=sn.FLUXCALERR[sn.FLT == flt],
                    fmt='.',color='k')
        
        ax.set_title('%s, %s'%(sn.SNID,bandpassdict[sn.SURVEY][flt]['fullname']),pad=2)
        try:
            if 'PEAKMJD' in sn.__dict__.keys():
                ax.set_xlim([sn.PEAKMJD-30,
                             sn.PEAKMJD+55])
            else:
                iMax = np.where(sn.FLUXCAL == np.max(sn.FLUXCAL))[0]
                if len(iMax) > 1:
                    iMax = iMax[0]

                ax.set_xlim([-20,50])
        except:
            import pdb; pdb.set_trace()
        ax.set_ylim([-np.max(sn.FLUXCAL)*1/20.,np.max(sn.FLUXCAL)*1.1])

        #if flt == 'c': import pdb; pdb.set_trace()
        #
        ax.legend(prop={'size':5})
    if 'SIM_SALT2x0' in sn.__dict__.keys():

        ax2.text(0.5,0.2,'SN %s\nsimulated $x_0$ = %8.5e, \n$x_1$ = %.2f, $c$ = %.2f, $z$ = %.2f'%(
            sn.SNID,sn.SIM_SALT2x0,sn.SIM_SALT2x1,sn.SIM_SALT2c,sn.SIM_REDSHIFT_HELIO),
                 ha='center',va='center',transform=ax2.transAxes,fontsize=7,
                 bbox={'facecolor':'1.0','edgecolor':'1.0','alpha':0.7,'pad':0})


    for ax in [ax1,ax2,ax3,ax4,ax5]:
        ax.set_xlabel('Phase')
        ax.set_xlim([-20,50])
    ax1.set_ylabel('Flux')
    ax2.yaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])
    ax4.yaxis.set_ticklabels([])
    ax5.yaxis.set_ticklabels([])
    #import pdb; pdb.set_trace()
    return salt2_chi2tot,salt3_chi2tot
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot lightcurves from SALT3 model against SALT2 model, Hsiao model, and data')
    parser.add_argument('lcfile',type=str,help='File with supernova fit parameters')
    parser.add_argument('salt3dir',type=str,help='File with supernova fit parameters')
    parser.add_argument('outfile',type=str,nargs='?',default=None,help='File with supernova fit parameters')
    parser=parser.parse_args()
    args=vars(parser)
    if parser.outfile is None:
        sn = snana.SuperNova(parser.lcfile)
        args['outfile']='lccomp_%s.png'%sn.SNID
    main(**args)
