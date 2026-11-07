#!/usr/bin/env python

import numpy as np
import pylab as plt
import sncosmo
import argparse
from saltshaker.util import snana
from saltshaker.util.salt3_sncosmo import colorlawfromfile
from astropy.table import Table
import astropy.units as u
from saltshaker.util.synphot import synphot
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from sncosmo.constants import HC_ERG_AA
from saltshaker.initfiles import init_rootdir
from saltshaker.training.init_templates import synphotB
import extinction
import copy
from astropy.io import fits
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
         clfile='salt3_color_correction.dat',
         cdfile='salt3_color_dispersion.dat',
         errscalefile='salt3_lc_dispersion_scaling.dat',
         lcrv00file='salt3_lc_variance_0.dat',
         lcrv11file='salt3_lc_variance_1.dat',
         lcrv01file='salt3_lc_covariance_01.dat',
         x0 = None, x1 = None, c = None, t0 = None,
         fitx1=False,fitc=False,bandpassdict=None):
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
    ax1.legend()
    plt.savefig(outfile)
    plt.show()
    plt.close(fig)

def customfilt(outfile,lcfile,salt3dir,
               m0file='salt3_template_0.dat',
               m1file='salt3_template_1.dat',
               m2file='salt3_template_2.dat',
               clfile='salt3_color_correction.dat',
               x0 = None, x1 = None, x2 = None, c = None, t0 = None,
               fitx1=False,fitc=False,bandpassdict=None,
               n_components=1, ax1=None, ax2=None, ax3=None, ax4=None, ax5=None,
               saltdict={},snid=None,modelsource='salt3'):

    comparison_chi2tot,salt3_chi2tot = 0,0
    
    if not ax1:
        plt.clf()
    
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
    fitbands = set()
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
            fitbands.add(flt)
                
                
        sysdict[flt] = sys
        
    flux = sn.FLUXCAL
    dust = sncosmo.F99Dust()
    comparisonmodel = sncosmo.Model(source=modelsource,effects=[dust],effect_names=['mw'],effect_frames=['obs'])
    comparisonname = ' '.join(str(x) for x in (comparisonmodel.source.name,comparisonmodel.source.version) if x)
    
    if not len(list(saltdict.keys())):
        salt3phase,salt3wave,salt3flux = np.genfromtxt('%s/%s'%(salt3dir,m0file),unpack=True)
        salt3m1flux = np.genfromtxt('%s/%s'%(salt3dir,m1file),unpack=True)[2]
        salt3flux = salt3flux.reshape([len(np.unique(salt3phase)),len(np.unique(salt3wave))])
        salt3m1flux = salt3m1flux.reshape(salt3flux.shape)
        if n_components == 3:
            salt3m2flux = np.genfromtxt('%s/%s'%(salt3dir,m2file),unpack=True)[2].reshape(salt3flux.shape)
        salt3phase = np.unique(salt3phase)*(1+sn.REDSHIFT_HELIO)
        salt3wave = np.unique(salt3wave)*(1+sn.REDSHIFT_HELIO)
    else:
        salt3phase,salt3wave,salt3flux = copy.deepcopy(saltdict['salt3phase']),copy.deepcopy(saltdict['salt3wave']),copy.deepcopy(saltdict['salt3flux'])
        salt3m1flux = copy.deepcopy(saltdict['salt3m1flux'])
        if n_components == 3:
            salt3m2flux = copy.deepcopy(saltdict['salt3m2flux'])
        salt3phase *= 1+sn.REDSHIFT_HELIO; salt3wave *= 1+sn.REDSHIFT_HELIO

    # color laws
    salt3colorlaw = colorlawfromfile('%s/%s'%(salt3dir,clfile))

    comparisonmodel.set(z=sn.REDSHIFT_HELIO,mwebv=sn.MWEBV)
    fitparams = ['t0']+list(comparisonmodel.source.param_names)
    # both models are refit to each SN within the same bounds
    fitbounds = {'t0':(t0-10, t0+10),'x0':(0,np.inf),'x1':(-3,3),'x2':(-3,3),'c':(-0.3,0.3)}

    try:
        result, fitted_model = sncosmo.fit_lc(data, comparisonmodel, fitparams,
                                              bounds={k:fitbounds[k] for k in fitparams
                                                      if k in ['t0','x1','c']})
        has_comparison = True
    except: has_comparison = False
    plotmjd = np.linspace(t0-20,
                          t0+55,200)
    
    if not ax1:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(151)
        ax2 = fig.add_subplot(152)
        ax3 = fig.add_subplot(153)
        ax4 = fig.add_subplot(154)
        ax5 = fig.add_subplot(155)
        
    try: mwextcurve = 10**(-0.4*extinction.fitzpatrick99(salt3wave,float(sn.MWEBV)*3.1))
    except: mwextcurve = 10**(-0.4*extinction.fitzpatrick99(salt3wave,sn.MWEBV*3.1))
    components = [salt3flux,salt3m1flux]
    if n_components == 3: components += [salt3m2flux]
    if fitc or c: salt3colorcurve = salt3colorlaw(salt3wave/(1+sn.REDSHIFT_HELIO))

    def bandlightcurve(flt,pars):
        filtwave = bandpassdict[sn.SURVEY][flt]['filtwave']
        filttrans = bandpassdict[sn.SURVEY][flt]['filttrans']

        g = (salt3wave >= filtwave[0]) & (salt3wave <= filtwave[-1])  # overlap range
        pbspl = np.interp(salt3wave[g],filtwave,filttrans)
        pbspl *= salt3wave[g]
        deltawave = salt3wave[g][1]-salt3wave[g][0]
        denom = np.sum(pbspl)*deltawave
        surface = sum(coeff*component[:,g] for coeff,component in
                      zip((1,pars['x1'],pars['x2'])[:n_components],components))*pars['x0']*_SCALE_FACTOR
        surface = surface*mwextcurve[g]
        if pars['c']:
            surface = surface*10. ** (-0.4 * salt3colorcurve[g] * pars['c'])
        synflux = np.sum(pbspl[np.newaxis,:]*surface,axis=1)*deltawave/HC_ERG_AA/denom
        synflux *= 10**(0.4*bandpassdict[sn.SURVEY][flt]['stdmag'])*10**(0.4*27.5)/(1+sn.REDSHIFT_HELIO)
        return interp1d(salt3phase,synflux,fill_value='extrapolate')

    # refit the trained model over the same points as the comparison, starting from the training values
    salt3pars = {'t0':t0,'x0':x0,'x1':x1 or 0,'x2':x2 or 0,'c':c or 0}
    salt3fitparams = ['t0','x0']+['x1']*fitx1+['x2']*(n_components == 3)+['c']*fitc
    fitpoints = []
    for flt in fitbands:
        iFLT = (sn.FLT == flt) & (sn.MJD-t0 > -20) & (sn.MJD-t0 < 50)
        fitpoints += [(flt,sn.MJD[iFLT],sn.FLUXCAL[iFLT],sn.FLUXCALERR[iFLT])]
    def residuals(values):
        trial = {**salt3pars,**dict(zip(salt3fitparams,values))}
        return np.concatenate([(flx-bandlightcurve(flt,trial)(mjd-trial['t0']))/flxe
                               for flt,mjd,flx,flxe in fitpoints])
    if sum(len(mjd) for _,mjd,_,_ in fitpoints):
        lower,upper = np.array([fitbounds[k] for k in salt3fitparams]).T
        start = np.clip([salt3pars[k] for k in salt3fitparams],lower,upper)
        salt3result = least_squares(residuals,start,bounds=(lower,upper),
                                    x_scale=[{'t0':1,'x0':x0}.get(k,1) for k in salt3fitparams])
        salt3pars.update(zip(salt3fitparams,salt3result.x))

    for flt,i,ax in zip(np.unique(sn.FLT),range(5),[ax1,ax2,ax3,ax4,ax5]):
        salt3lc = bandlightcurve(flt,salt3pars)
        iFLT = (sn.FLT == flt) & (sn.MJD-t0 > -20) & (sn.MJD-t0 < 50)
        chi2_salt3 = np.sum((sn.FLUXCAL[iFLT]-salt3lc(sn.MJD[iFLT]-salt3pars['t0']))**2./sn.FLUXCALERR[iFLT]**2.)
        chi2red_salt3 = chi2_salt3/(len(sn.FLUXCAL[(sn.FLT == flt) & (sn.MJD-t0 > -20) & (sn.MJD-t0 < 50)])-3)
        salt3label = 'Trained, $x_0$ = %8.5e, \nx1=%.2f, z=%.3f\nc=%.3f, $\\Delta t_0$=%.2f\n$\\chi_{red}^2=%.1f$'%(
            salt3pars['x0'],salt3pars['x1'],sn.REDSHIFT_HELIO,salt3pars['c'],salt3pars['t0']-t0,chi2red_salt3)
        if n_components == 3: salt3label = salt3label.replace(', z=',', x2=%.2f, z='%salt3pars['x2'])
        ax.plot(plotmjd-t0,salt3lc(plotmjd-salt3pars['t0']),color='C2',label=salt3label)
        if bandpassdict[sn.SURVEY][flt]['lambdaeff']/(1+sn.REDSHIFT_HELIO) > 2800 and \
           bandpassdict[sn.SURVEY][flt]['lambdaeff']/(1+sn.REDSHIFT_HELIO) < 9000:

            try:
                if has_comparison:
                    chi2 = np.sum((sn.FLUXCAL[iFLT]-fitted_model.bandflux(
                        flt, sn.MJD[iFLT], zp=27.5-bandpassdict[sn.SURVEY][flt]['primarymag'],zpsys=sysdict[flt]))**2./sn.FLUXCALERR[iFLT]**2.)
                    chi2red = chi2/(len(sn.FLUXCAL[iFLT])-3)
                    comparisonpars = ', '.join('%s = %.2f'%(name,fitted_model.get(name))
                                               for name in fitparams if name in ['x1','c'])
                    ax.plot(plotmjd-t0,fitted_model.bandflux(
                        flt, plotmjd, zp=27.5-bandpassdict[sn.SURVEY][flt]['primarymag'],zpsys=sysdict[flt]),color='C1',
                            label='%s; %s\n$\Delta t_0$=%.2f, $\chi_{red}^2 = %.1f$'%(
                                comparisonname,comparisonpars,fitted_model.get('t0')-t0,chi2red))
                    comparison_chi2tot += chi2
                    salt3_chi2tot += chi2_salt3
            except ValueError: pass

        ax.errorbar(sn.MJD[sn.FLT == flt]-t0,sn.FLUXCAL[sn.FLT == flt],
                    yerr=sn.FLUXCALERR[sn.FLT == flt],
                    fmt='.',color='k')
        
        ax.set_title('%s, %s'%(sn.SNID,bandpassdict[sn.SURVEY][flt]['fullname']),pad=2)
        ax.set_ylim([-np.max(sn.FLUXCAL)*1/20.,np.max(sn.FLUXCAL)*1.1])

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
    return comparison_chi2tot,salt3_chi2tot
    
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
