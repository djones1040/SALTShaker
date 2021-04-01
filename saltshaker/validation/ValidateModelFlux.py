#!/usr/bin/env python
# wraps training.saltfit.modelValsForSN to
# make sure training code is getting sensible
# SALT values

from saltshaker.training.base import TrainSALTBase
from saltshaker.training.TrainSALT import TrainSALT
from saltshaker.util import readutils
from saltshaker.training import saltfit
import configparser
import numpy as np
from sncosmo.salt2utils import SALT2ColorLaw
import extinction
from scipy.interpolate import interp1d
from saltshaker.util.synphot import synphot
from sncosmo.constants import HC_ERG_AA
import astropy.table as at
from astropy.table import Table
from saltshaker.util.salt3_sncosmo import SALT3Source
import sncosmo
import astropy.units as u
import pylab as plt
plt.ion()
_SCALE_FACTOR = 1e-12


class SALT3Model:

    def __init__(self,configfile):
        ts=TrainSALT()

        config = configparser.ConfigParser()
        config.read(configfile)
        user_parser = ts.add_user_options(usage='',config=config)
        user_options = user_parser.parse_known_args()[0]

        trainingconfig = configparser.ConfigParser()
        trainingconfig.read(user_options.trainingconfig)

        training_parser = ts.add_training_options(
            usage='',config=trainingconfig)
        training_options = training_parser.parse_known_args(namespace=user_options)[0]
        ts.options=training_options
        
        self.kcordict=readutils.rdkcor(ts.surveylist,ts.options)

        if ts.options.binspec:
            binspecres = ts.options.binspecres
        else:
            binspecres = None

        ts.kcordict=self.kcordict
        datadict = readutils.rdAllData(ts.options.snlists,ts.options.estimate_tpk,ts.kcordict,
                                       dospec=ts.options.dospec,KeepOnlySpec=ts.options.keeponlyspec,
                                       peakmjdlist=ts.options.tmaxlist,waverange=ts.options.waverange,
                                       binspecres=binspecres,snparlist=ts.options.snparlist)
        datadict = ts.mkcuts(datadict)
        ts.options.initbfilt='Bessell90_B.dat'
        ts.options.outputdir='output'
        ts.options.resume_from_outputdir=True

        parlist,guess,phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc = ts.initialParameters(datadict)
        saltfitkwargs = ts.get_saltkw(phaseknotloc,waveknotloc,errphaseknotloc,errwaveknotloc)
        X0=guess.copy()
        
        saltfitkwargs['fitting_sequence']='all'
        saltfitkwargs['phaseoutres']=.2
        saltfitkwargs.pop('fitTpkOff')
        self.resids = saltfit.GaussNewton(guess,datadict,parlist,fix_salt2modelpars=False,**saltfitkwargs,fitTpkOff=True)
        #X0=self.resids.priors.satisfyDefinitions(X0,self.resids.SALTModel(X0))

        self.datadict = datadict
        self.ts = ts
        self.X = X0
        self.parlist = parlist
        
    def bandflux(self,snparams,band,snid):

        X = self.X.copy()
        # params: t0,x0,x1,c
        X[self.parlist==f'tpkoff_{snid}']=self.datadict[snid]['tpk']-snparams[0]
        X[self.parlist==f'x0_{snid}']=snparams[1]
        X[self.parlist==f'x1_{snid}']=snparams[2]
        X[self.parlist==f'c_{snid}']=snparams[3]
        
        photmodel=self.resids.modelvalsforSN(X,snid,{},np.zeros(X.size,dtype=bool))[0]
        return photmodel[band]['modelflux']*10**(-0.4*0.27),np.sqrt(photmodel[band]['modelvariance'])*10**(-0.4*0.27)

    def snanaflux(self):
        flux,fluxerr = np.array([]),np.array([])
        with open('tmp') as fin:
            for line in fin:
                flux = np.append(flux,float(line.split('Fmodel=')[-1].split('+-')[0]))
                fluxerr = np.append(fluxerr,float(line.split('+-')[-1].split()[0]))
        return flux,fluxerr

    
class SALT3ModelFiles:
    
    def __init__(self,configfile):

        ts=TrainSALT()
        config = configparser.ConfigParser()
        config.read(configfile)
        user_parser = ts.add_user_options(usage='',config=config)
        user_options = user_parser.parse_known_args()[0]

        trainingconfig = configparser.ConfigParser()
        trainingconfig.read(user_options.trainingconfig)

        training_parser = ts.add_training_options(
            usage='',config=trainingconfig)
        training_options = training_parser.parse_known_args(namespace=user_options)[0]
        ts.options=training_options

        
        self.kcordict=readutils.rdkcor(ts.surveylist,ts.options)

        self.salt3dir = training_options.outputdir

        salt3phase,salt3wave,salt3flux = np.genfromtxt(f'{self.salt3dir}/salt3_template_0.dat',unpack=True)
        salt3m1phase,salt3m1wave,salt3m1flux = np.genfromtxt(f'{self.salt3dir}/salt3_template_1.dat',unpack=True)

        salt3m0varphase,salt3m0varwave,salt3m0var = np.genfromtxt(f'{self.salt3dir}/salt3_lc_variance_0.dat',unpack=True)
        salt3m1varphase,salt3m1varwave,salt3m1var = np.genfromtxt(f'{self.salt3dir}/salt3_lc_variance_1.dat',unpack=True)
        salt3covarphase,salt3covarwave,salt3covar = np.genfromtxt(f'{self.salt3dir}/salt3_lc_covariance_01.dat',unpack=True)
        self.cdispwave,self.cdisp = np.genfromtxt(f'{self.salt3dir}/salt3_color_dispersion.dat',unpack=True)
        
        self.salt3phase = np.unique(salt3phase)
        self.salt3wave = np.unique(salt3wave)
        self.salt3flux = salt3flux.reshape([len(self.salt3phase),len(self.salt3wave)])
        self.salt3m1flux = salt3m1flux.reshape([len(self.salt3phase),len(self.salt3wave)])

        salt3m0varphase = np.unique(salt3m0varphase)
        salt3m0varwave = np.unique(salt3m0varwave)
        salt3m0var = salt3m0var.reshape([len(salt3m0varphase),len(salt3m0varwave)])
        salt3m1var = salt3m1var.reshape([len(salt3m0varphase),len(salt3m0varwave)])
        salt3covar = salt3covar.reshape([len(salt3m0varphase),len(salt3m0varwave)])

        int1dm0phasevar = interp1d(salt3m0varphase,salt3m0var,axis=0,fill_value='extrapolate')
        salt3m0var = int1dm0phasevar(self.salt3phase)
        int1dm0wavevar = interp1d(salt3m0varwave,salt3m0var,axis=1,fill_value='extrapolate')
        self.salt3m0var = int1dm0wavevar(self.salt3wave)        
        int1dm1phasevar = interp1d(salt3m0varphase,salt3m1var,axis=0,fill_value='extrapolate')
        salt3m1var = int1dm1phasevar(self.salt3phase)
        int1dm1wavevar = interp1d(salt3m0varwave,salt3m1var,axis=1,fill_value='extrapolate')
        self.salt3m1var = int1dm1wavevar(self.salt3wave)
        int1dcophasevar = interp1d(salt3m0varphase,salt3covar,axis=0,fill_value='extrapolate')
        salt3covar = int1dcophasevar(self.salt3phase)
        int1dcowavevar = interp1d(salt3m0varwave,salt3covar,axis=1,fill_value='extrapolate')
        self.salt3covar = int1dcowavevar(self.salt3wave)
        
        self.n_colorpars = ts.options.n_colorpars

        self.datadict = readutils.rdAllData(
            ts.options.snlists,ts.options.estimate_tpk,self.kcordict,
            dospec=ts.options.dospec,KeepOnlySpec=ts.options.keeponlyspec,
            peakmjdlist=ts.options.tmaxlist,waverange=ts.options.waverange,
            binspecres=ts.options.binspecres,snparlist=ts.options.snparlist)

        for k in self.kcordict.keys():
            if k == 'default': continue
            for k2 in self.kcordict[k].keys():
                if k2 not in ['primarywave','snflux','BD17','filtwave','AB','Vega']:
                    if self.kcordict[k][k2]['magsys'] == 'AB': primarykey = 'AB'
                    elif self.kcordict[k][k2]['magsys'] == 'Vega': primarykey = 'Vega'
                    elif self.kcordict[k][k2]['magsys'] == 'VEGA': primarykey = 'Vega'
                    elif self.kcordict[k][k2]['magsys'] == 'BD17': primarykey = 'BD17'

                    self.kcordict[k][k2]['stdmag'] = synphot(
                        self.kcordict[k]['primarywave'],
                        self.kcordict[k][primarykey],
                        filtwave=self.kcordict[k]['filtwave'],
                        filttp=self.kcordict[k][k2]['filttrans'],
                        zpoff=0) - self.kcordict[k][k2]['primarymag']

        return

    def bandflux(self,snparams,band,snid):

        salt3phase = self.salt3phase*(1+self.datadict[snid]['zHelio'])
        salt3wave = self.salt3wave*(1+self.datadict[snid]['zHelio'])
        
        t0,x0,x1,c = snparams

        with open(f'{self.salt3dir}/salt3_color_correction.dat') as fin:
            lines = fin.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n','')
        colorlaw_salt3_coeffs = np.array(lines[1:self.n_colorpars+1]).astype('float')
        salt3_colormin = float(lines[self.n_colorpars+2].split()[1])
        salt3_colormax = float(lines[self.n_colorpars+3].split()[1])

        salt3colorlaw = SALT2ColorLaw([salt3_colormin,salt3_colormax],colorlaw_salt3_coeffs)

        # flux
        salt3flux = x0*(self.salt3flux + x1*self.salt3m1flux)
        salt3finteg = self.salt3flux + x1*self.salt3m1flux
        salt3flux *= 10. ** (-0.4 * salt3colorlaw(salt3wave/(1+self.datadict[snid]['zHelio'])) * c)
        salt3flux *= _SCALE_FACTOR
        salt3finteg *= _SCALE_FACTOR
        
        # fluxerr
        salt3fluxerr = x0*np.sqrt(self.salt3m0var + 2.0*x1*self.salt3covar + x1**2.*self.salt3m1var)
        salt3fluxerrtmp = np.sqrt(self.salt3m0var + 2.0*x1*self.salt3covar + x1**2.*self.salt3m1var)
        salt3fluxerr *= 10. ** (-0.4 * salt3colorlaw(salt3wave/(1+self.datadict[snid]['zHelio'])) * c)
        salt3fluxerr *= _SCALE_FACTOR

        int1d = interp1d(salt3phase,salt3flux,axis=0,fill_value='extrapolate')
        int1derr = interp1d(salt3phase,salt3fluxerr,axis=0,fill_value='extrapolate')
        int1derrtmp = interp1d(salt3phase,salt3fluxerrtmp,axis=0,fill_value='extrapolate')
        int1df = interp1d(salt3phase,salt3finteg,axis=0,fill_value='extrapolate')
        phase=self.datadict[snid]['photdata']['mjd'][self.datadict[snid]['photdata']['filt'] == band]-t0
        print(phase)
        salt3fluxnew = int1d(phase)
        salt3fluxerrnew = int1derr(phase)
        salt3fluxerrtmp = int1derrtmp(phase)
        salt3fintegtmp = int1df(phase)
        
        mwextcurve = 10**(-0.4*extinction.fitzpatrick99(salt3wave,self.datadict[snid]['MWEBV']*3.1))
        salt3fluxnew *= mwextcurve[np.newaxis,:]
        salt3fluxerrnew *= mwextcurve[np.newaxis,:]
        
        filtwave = self.kcordict[self.datadict[snid]['survey']]['filtwave']
        filttrans = self.kcordict[self.datadict[snid]['survey']][band]['filttrans']

        g = (salt3wave >= filtwave[0]) & (salt3wave <= filtwave[-1])  # overlap range
        pbspl = np.interp(salt3wave[g],filtwave,filttrans)
        pbspl *= salt3wave[g]
        deltawave = salt3wave[g][1]-salt3wave[g][0]
        denom = np.sum(pbspl)*deltawave
        salt3synflux=np.sum(pbspl[np.newaxis,:]*salt3fluxnew[:,g],axis=1)*deltawave/HC_ERG_AA/denom
        salt3fintegtmp = np.sum(pbspl[np.newaxis,:]*salt3fintegtmp[:,g],axis=1)*deltawave/HC_ERG_AA/denom

        salt3synflux_denom = salt3synflux/(10. ** (-0.4 * salt3colorlaw(np.atleast_1d(self.kcordict[self.datadict[snid]['survey']][band]['lambdaeff']/(1+self.datadict[snid]['zHelio']))) * c))/10**(-0.4*extinction.fitzpatrick99(np.atleast_1d(self.kcordict[self.datadict[snid]['survey']][band]['lambdaeff']).astype(float),self.datadict[snid]['MWEBV']*3.1))

        
        salt3synfluxerr = np.array([np.interp(self.kcordict[self.datadict[snid]['survey']][band]['lambdaeff'],salt3wave[g],salt3fluxerrnew[i,g]) for i,p in enumerate(phase)])
        salt3synfluxerrtmp = np.array([np.interp(self.kcordict[self.datadict[snid]['survey']][band]['lambdaeff'],salt3wave[g],salt3fluxerrtmp[i,g]) for i,p in enumerate(phase)])
        #import pdb; pdb.set_trace()
        salt3synfluxerr /= HC_ERG_AA

        salt3synflux *= 10**(0.4*self.kcordict[self.datadict[snid]['survey']][band]['stdmag'])*10**(0.4*27.5)/(1+self.datadict[snid]['zHelio'])
        salt3synfluxerr *= 10**(0.4*self.kcordict[self.datadict[snid]['survey']][band]['stdmag'])*10**(0.4*27.5)//(1+self.datadict[snid]['zHelio'])


        # SALT3 model flux from finteg
        salt3modelflux_full = salt3fintegtmp*x0*10**(-0.4*extinction.fitzpatrick99(np.atleast_1d(self.kcordict[self.datadict[snid]['survey']][band]['lambdaeff']).astype(float),self.datadict[snid]['MWEBV']*3.1))/(1+self.datadict[snid]['zHelio'])*10. ** (-0.4 * salt3colorlaw(np.atleast_1d(self.kcordict[self.datadict[snid]['survey']][band]['lambdaeff']/(1+self.datadict[snid]['zHelio']))) * c)*10**(0.4*self.kcordict[self.datadict[snid]['survey']][band]['stdmag'])*10**(0.4*27.5)
        # SALT3 model error from salt3synfluxerrtmp
        salt3modelerr_full = salt3synfluxerrtmp*10. ** (-0.4 * salt3colorlaw(np.atleast_1d(self.kcordict[self.datadict[snid]['survey']][band]['lambdaeff']/(1+self.datadict[snid]['zHelio']))) * c)*10**(0.4*self.kcordict[self.datadict[snid]['survey']][band]['stdmag'])*10**(0.4*27.5)*10**(-0.4*extinction.fitzpatrick99(np.atleast_1d(self.kcordict[self.datadict[snid]['survey']][band]['lambdaeff']).astype(float),self.datadict[snid]['MWEBV']*3.1))/(1+self.datadict[snid]['zHelio'])*x0/HC_ERG_AA/1e12

        
        # SNANA SALT3 does
        # fractional error = (varM0 + x1^2*varM1 + 2*x1*covarM0M1)/integ(M0+x1*M1)
        # in this code that's salt3synfluxerrtmp/salt3fintegtmp
        return salt3synflux*10**(-0.4*0.27),salt3synfluxerr*10**(-0.4*0.27) #/salt3synflux/HC_ERG_AA/1e12 #*salt3synfluxerrtmp # SNANA factor
#tmp/salt3synflux_denom

class TestSNCosmo:
    
    def __init__(self,configfile):

        ts=TrainSALT()
        config = configparser.ConfigParser()
        config.read(configfile)
        user_parser = ts.add_user_options(usage='',config=config)
        user_options = user_parser.parse_known_args()[0]

        trainingconfig = configparser.ConfigParser()
        trainingconfig.read(user_options.trainingconfig)

        training_parser = ts.add_training_options(
            usage='',config=trainingconfig)
        training_options = training_parser.parse_known_args(namespace=user_options)[0]
        ts.options=training_options
        self.salt3dir = training_options.outputdir
        
        self.kcordict=readutils.rdkcor(ts.surveylist,ts.options)

        self.datadict = readutils.rdAllData(
            ts.options.snlists,ts.options.estimate_tpk,self.kcordict,
            dospec=ts.options.dospec,KeepOnlySpec=ts.options.keeponlyspec,
            peakmjdlist=ts.options.tmaxlist,waverange=ts.options.waverange,
            binspecres=ts.options.binspecres,snparlist=ts.options.snparlist)

        for k in self.kcordict.keys():
            if k == 'default': continue
            for k2 in self.kcordict[k].keys():
                if k2 not in ['primarywave','snflux','BD17','filtwave','AB','Vega']:
                    if self.kcordict[k][k2]['magsys'] == 'AB': primarykey = 'AB'
                    elif self.kcordict[k][k2]['magsys'] == 'Vega': primarykey = 'Vega'
                    elif self.kcordict[k][k2]['magsys'] == 'VEGA': primarykey = 'Vega'
                    elif self.kcordict[k][k2]['magsys'] == 'BD17': primarykey = 'BD17'

                    self.kcordict[k][k2]['stdmag'] = synphot(
                        self.kcordict[k]['primarywave'],
                        self.kcordict[k][primarykey],
                        filtwave=self.kcordict[k]['filtwave'],
                        filttp=self.kcordict[k][k2]['filttrans'],
                        zpoff=0) - self.kcordict[k][k2]['primarymag']

        return


    def dofit(self,snparams,snid):
        plt.close('all')
        # astropy table from SALT3 datadict
        data = Table(rows=None,names=['mjd','band','flux','fluxerr','zp','zpsys'],
                     dtype=('f8','S1','f8','f8','f8','U5'),
                     meta={'t0':self.datadict[snid]['tpk']})
        for flt in np.unique(self.datadict[snid]['photdata']['filt']):
            filtwave = self.kcordict[self.datadict[snid]['survey']]['filtwave']
            filttrans = self.kcordict[self.datadict[snid]['survey']][flt]['filttrans']

            band = sncosmo.Bandpass(
                filtwave,
                filttrans,
                wave_unit=u.angstrom,name=flt)
            sncosmo.register(band, force=True)

        sysdict = {}
        photdata = self.datadict[snid]['photdata']
        for m,flt,flx,flxe in zip(photdata['mjd'],photdata['filt'],photdata['fluxcal'],photdata['fluxcalerr']):
            if self.kcordict[self.datadict[snid]['survey']][flt]['magsys'] == 'BD17': sys = 'bd17'
            elif self.kcordict[self.datadict[snid]['survey']][flt]['magsys'] == 'AB': sys = 'ab'
            else: sys = 'vega'
            if self.kcordict[self.datadict[snid]['survey']][flt]['lambdaeff']/(1+self.datadict[snid]['zHelio']) > 2800 and \
               self.kcordict[self.datadict[snid]['survey']][flt]['lambdaeff']/(1+self.datadict[snid]['zHelio']) < 9000 and\
               '-u' not in self.kcordict[self.datadict[snid]['survey']][flt]['fullname']:
                data.add_row((m,flt,flx*10**(0.4*self.kcordict[self.datadict[snid]['survey']][flt]['primarymag']),
                              flxe*10**(0.4*self.kcordict[self.datadict[snid]['survey']][flt]['primarymag']),
                              27.5,sys))
            sysdict[flt] = sys
        
        # do the fit with sncosmo
        salt3source = SALT3Source(modeldir=self.salt3dir)
        dust = sncosmo.F99Dust()
        salt3model = sncosmo.Model(salt3source,effects=[dust],effect_names=['mw'],effect_frames=['obs'])

        salt3model.set(z=self.datadict[snid]['zHelio'],mwebv=self.datadict[snid]['MWEBV'])
        t0 = self.datadict[snid]['tpk']
        fitparams = ['t0', 'x0', 'x1', 'c']
        if not len(snparams):
            result, fitted_model = sncosmo.fit_lc(
                data, salt3model, fitparams,
                bounds={'t0':(t0-10, t0+10),
                        'z':(0.0,0.7),'x1':(-4,4),'c':(-0.3,0.3)})
        else:
            salt3model.set(t0=snparams[0],x0=snparams[1]*10**(-0.4*0.27),x1=snparams[2],c=snparams[3])
            fitted_model = salt3model
            
        # plot the fit and model errors
        ax1,ax2,ax3,ax4 = plt.subplot(221),plt.subplot(222),plt.subplot(223),plt.subplot(224)
        plotmjd = np.linspace(t0-20,
                              t0+55,200)
        for flt,i,ax in zip(np.unique(self.datadict[snid]['photdata']['filt']),range(4),[ax1,ax2,ax3,ax4]):
            modelflux,modelerr = fitted_model.bandfluxcov(
                flt, plotmjd, zp=27.5-self.kcordict[self.datadict[snid]['survey']][flt]['primarymag'],zpsys=sysdict[flt])
            modelerr = np.sqrt(np.diag(modelerr))

            #modelerr = np.sqrt(salt3source._bandflux_rvar_single(flt,plotmjd-t0))
            ax.fill_between(plotmjd-t0,modelflux-modelerr,modelflux+modelerr,color='C1',alpha=0.5)
            ax.errorbar(photdata['mjd'][photdata['filt'] == flt]-t0,photdata['fluxcal'][photdata['filt'] == flt],
                        yerr=photdata['fluxcalerr'][photdata['filt'] == flt],fmt='o',color='k')
            modelflux_chi2,modelerr_chi2 = fitted_model.bandfluxcov(
                flt, photdata['mjd'][photdata['filt'] == flt], zp=27.5-self.kcordict[self.datadict[snid]['survey']][flt]['primarymag'],zpsys=sysdict[flt])
            modelerr_chi2 = np.sqrt(np.diag(modelerr_chi2))
            
            chi2 = np.sum((modelflux_chi2-photdata['fluxcal'][photdata['filt'] == flt])**2./(modelerr_chi2**2.+photdata['fluxcalerr'][photdata['filt'] == flt]**2.))
            if not len(snparams):
                ax.plot(plotmjd-t0,modelflux,color='C1',
                    label=f"""SALT2; $x_1 = {result['parameters'][3]:.2f}$, 
$c = {result['parameters'][4]:.3f}$""")
            else:
                ax.plot(plotmjd-t0,modelflux,color='C1',
                        label=f"""SALT2; $x_1 = {snparams[2]:.2f}$, 
$c = {snparams[3]:.3f}$""")
            ax.set_title(f"{flt}, $\chi^2={chi2:.3f}$")
            #import pdb; pdb.set_trace()
        ax1.legend()
        # plot the model forced to input snparams w/ model errors

        import pdb; pdb.set_trace()
        
