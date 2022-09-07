#!/usr/bin/env python
# D. Jones - 12/15/21

import os
import glob
import numpy as np
from saltshaker.util.txtobj import txtobj
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from saltshaker.util import getmu
from scipy.stats import binned_statistic
import cosmo
from scipy.optimize import minimize
from astropy.stats import sigma_clipped_stats
import warnings

_nmltmpl = f"""
  &SNLCINP
     PRIVATE_DATA_PATH = '{os.getcwd()}/SALT3TRAIN_J21'

     VERSION_PHOTOMETRY = '<data_version>'
     KCOR_FILE         = '<kcor>'

     NFIT_ITERATION = 3
     INTERP_OPT     = 1

     SNTABLE_LIST = 'FITRES LCPLOT(text:key)'
     TEXTFILE_PREFIX  = '<outfile>'
     
     LDMP_SNFAIL = T
     USE_MWCOR = F
     USE_MINOS = F

     H0_REF   = 70.0
     OLAM_REF =  0.70
     OMAT_REF =  0.30
     W0_REF   = -1.00

     SNCID_LIST    =  0
     CUTWIN_CID    =  0, 20000000
     !SNCCID_LIST   =  <cidlist>
     SNCCID_IGNORE =  

     cutwin_redshift   = 0.001, 2.0
     cutwin_Nepoch    =  1

     RV_MWCOLORLAW = 3.1
     OPT_MWCOLORLAW = 99
     OPT_MWEBV = 3
     MWEBV_SCALE = 1.00
     MWEBV_SHIFT = 0.0
     FUDGE_MAG_ERROR = 
     MXLC_PLOT = 10000

     MAGOBS_SHIFT_PRIMARY = ' '
     EPCUT_SNRMIN = ''
     ABORT_ON_NOEPOCHS = F

  &END
  &FITINP

     FITMODEL_NAME  = '<model>'
    
     PRIOR_MJDSIG        = 5.0
     PRIOR_LUMIPAR_RANGE = -5.0, 5.0

     OPT_XTMW_ERR = 1
     OPT_COVAR_FLUX = 0
     TREST_REJECT  = -15.0, 45.0
     NGRID_PDF     = 0

     FUDGEALL_ITER1_MAXFRAC = 0.02
     FILTLIST_FIT = '<filtlist>'

  &END
"""


class masshubbleresids():
    def __init__(self):
        pass

    def main(self):
        # edit the output model to give high- and low-mass results
        #self.edit_model()
        
        # run the LC fitting
        # and SALT2mu to give hubble residuals
        self.snlc_fit()
        
        # make some comparison plots
        #self.model_table()
        #self.plot_hrs_threepanel()

        # measure the mass step, w/ and w/o SALT2mu

    def edit_model_list(self):

        for errs in [True,False]:
            for modeldir,modelname,host in zip(
                    ['SALT3.host_bootstrap','SALT3.host_fixedpars_bootstrap','SALT3.nohost_bootstrap'],
                    ['SALT3Models/SALT3.Host','SALT3Models/SALT3.HostFixed','SALT3Models/SALT3.NoHost'],
                    [True,True,False]):

                if errs:
                    modelnamestr = modelname
                else:
                    modelnamestr = modelname+'NoErrs'
                self.edit_model(modeldir=modeldir,modelname=modelnamestr,errs=errs,host=host)
                
    def edit_model(self,modeldir='SALT3.host_fixedpars_bootstrap',modelname='SALT3.',errs=True,host=True):

        if not os.path.exists(f'{modelname}HighMass'):
            os.makedirs(f'{modelname}HighMass')
        if not os.path.exists(f'{modelname}LowMass'):
            os.makedirs(f'{modelname}LowMass')

        #os.system(f'cp salt3_template_0.dat.gz ')

        # copy to HighMass dir
        os.system(f'cp {modeldir}/SALT3.INFO {modelname}HighMass/')
        os.system(f'cp {modeldir}/salt3_color_correction.dat {modelname}HighMass/')
        os.system(f'cp {modeldir}/salt3_color_dispersion.dat {modelname}HighMass/')

        # copy to LowMass Dir
        os.system(f'cp {modeldir}/SALT3.INFO {modelname}LowMass/')
        os.system(f'cp {modeldir}/salt3_color_correction.dat {modelname}LowMass/')
        os.system(f'cp {modeldir}/salt3_color_dispersion.dat {modelname}LowMass/')

        # copy color dispersion or file full of zeros
        if errs:
            os.system(f'cp {modeldir}/salt3_color_dispersion.dat {modelname}HighMass/')
            os.system(f'cp {modeldir}/salt3_color_dispersion.dat {modelname}LowMass/')
        else:
            wave,disp = np.loadtxt(f'{modeldir}/salt3_color_dispersion.dat',unpack=True)
            for filename in [f"{modelname}HighMass/salt3_color_dispersion.dat",f"{modelname}LowMass/salt3_color_dispersion.dat"]:
                with open(f"{modelname}LowMass/salt3_color_dispersion.dat","w") as fout:
                    for w,d in zip(wave,disp):
                        print(f"{w:.1f} 0.0",file=fout)
            
        # now edit the M0 component for high vs. low-mass SNe
        m0phase,m0wave,m0flux = np.loadtxt(f"{modeldir}/salt3_template_0.dat",unpack=True)
        m1phase,m1wave,m1flux = np.loadtxt(f"{modeldir}/salt3_template_1.dat",unpack=True)
        m0varphase,m0varwave,m0varflux = np.loadtxt(f"{modeldir}/salt3_lc_variance_0.dat",unpack=True)
        m1varphase,m1varwave,m1varflux = np.loadtxt(f"{modeldir}/salt3_lc_variance_1.dat",unpack=True)
        m01covarphase,m01covarwave,m01covarflux = np.loadtxt(f"{modeldir}/salt3_lc_covariance_01.dat",unpack=True)
        if host:
            mhostphase,mhostwave,mhostflux = np.loadtxt(f"{modeldir}/salt3_template_host.dat",unpack=True)
            mhostvarphase,mhostvarwave,mhostvarflux = np.loadtxt(f"{modeldir}/salt3_lc_variance_host.dat",unpack=True)
            m0hostcovarphase,m0hostcovarwave,m0hostcovarflux = np.loadtxt(f"{modeldir}/salt3_lc_covariance_0host.dat",unpack=True)
        
        # model flux = x0*(M0 + x1*M1 + xhost*Mhost)
        # xhost is +0.5 for high-mass
        # xhost is -0.5 for low-mass
        if host:
            m0highmassflux = m0flux + mhostflux*0.5
            m0lowmassflux= m0flux - mhostflux*0.5
            m0highmassvarflux = m0varflux + 0.5**2.*mhostvarflux + 2*0.5*m0hostcovarflux*np.sqrt(m0varflux)*np.sqrt(mhostvarflux)
            m0lowmassvarflux = m0varflux + 0.5**2.*mhostvarflux - 2*0.5*m0hostcovarflux*np.sqrt(m0varflux)*np.sqrt(mhostvarflux)
        else:
            m0highmassflux = m0flux; m0lowmassflux = m0flux
            m0highmassvarflux = m0varflux; m0lowmassvarflux = m0varflux
            
        from saltshaker.training.init_hsiao import synphotB, synphotBflux
        Bfilt = '../../saltshaker/initfiles/Bessell90_B.dat'
        refWave,refFlux=np.loadtxt('../../saltshaker/initfiles/flatnu.dat',unpack=True)
        
        # we have to make sure M1 has a B-band flux of zero
        m0Bflux = synphotBflux(m0wave[m0phase==0],m0lowmassflux[m0phase==0],0,0,Bfilt)
        m1Bflux = synphotBflux(m0wave[m0phase==0],m1flux[m0phase==0],0,0,Bfilt)
        ratio =m1Bflux/m0Bflux
        m1lowmassflux = m1flux - ratio*m0lowmassflux
        m1lowmassvarflux = m1varflux - 2*ratio*m01covarflux+ratio**2*m0varflux
        m01lowmasscovarflux = m01covarflux - m0varflux*ratio

        
        m0Bflux = synphotBflux(m0wave[m0phase==0],m0highmassflux[m0phase==0],0,0,Bfilt)
        m1Bflux = synphotBflux(m0wave[m0phase==0],m1flux[m0phase==0],0,0,Bfilt)
        ratio =m1Bflux/m0Bflux
        m1highmassflux = m1flux - ratio*m0highmassflux
        m1highmassvarflux = m1varflux - 2*ratio*m01covarflux+ratio**2*m0varflux
        m01highmasscovarflux = m01covarflux - m0varflux*ratio

        # now fix M0 mag
        flux_adj_highmass = 10**(-0.4*(-19.49+(synphotB(refWave,refFlux,0,0,Bfilt)-synphotB(m0wave[m0phase==0],m0highmassflux[m0phase==0],0,0,Bfilt))))
        flux_adj_lowmass = 10**(-0.4*(-19.49+(synphotB(refWave,refFlux,0,0,Bfilt)-synphotB(m0wave[m0phase==0],m0lowmassflux[m0phase==0],0,0,Bfilt))))
#        import pdb; pdb.set_trace()
        # now correct so B-band mag is same
        m0highmassflux *= flux_adj_highmass
        m0highmassvarflux *= flux_adj_highmass**2.
        m1highmassflux = m1highmassflux*flux_adj_highmass
        m1highmassvarflux = m1highmassvarflux*flux_adj_highmass**2.
        m01highmasscovarflux = m01highmasscovarflux*flux_adj_highmass**2.

        m0lowmassflux *= flux_adj_lowmass
        m0lowmassvarflux *= flux_adj_lowmass**2.
        m1lowmassflux = m1lowmassflux*flux_adj_lowmass
        m1lowmassvarflux = m1lowmassvarflux*flux_adj_lowmass**2.
        m01lowmasscovarflux = m01lowmasscovarflux*flux_adj_lowmass**2.

        if not errs:
            m0highmassvarflux[:] = 0.0
            m0lowmassvarflux[:] = 0.0
            m1highmassvarflux[:] = 0.0
            m1lowmassvarflux[:] = 0.0
            m01highmasscovarflux[:] = 0.0
            m01lowmasscovarflux[:] = 0.0
        
        with open(f'{modelname}HighMass/salt3_template_0.dat','w') as fout:
            for p,w,m in zip(m0phase,m0wave,m0highmassflux):
                print(f'{p:.1f} {w:.2f} {m:8.15e}',file=fout)
        with open(f'{modelname}LowMass/salt3_template_0.dat','w') as fout:
            for p,w,m in zip(m0phase,m0wave,m0lowmassflux):
                print(f'{p:.1f} {w:.2f} {m:8.15e}',file=fout)
        with open(f'{modelname}HighMass/salt3_template_1.dat','w') as fout:
            for p,w,m in zip(m0phase,m0wave,m1highmassflux):
                print(f'{p:.1f} {w:.2f} {m:8.15e}',file=fout)
        with open(f'{modelname}LowMass/salt3_template_1.dat','w') as fout:
            for p,w,m in zip(m0phase,m0wave,m1lowmassflux):
                print(f'{p:.1f} {w:.2f} {m:8.15e}',file=fout)
        with open(f'{modelname}HighMass/salt3_lc_variance_0.dat','w') as fout:
            for p,w,m in zip(m0phase,m0wave,m0highmassvarflux):
                print(f'{p:.1f} {w:.2f} {m:8.15e}',file=fout)
        with open(f'{modelname}LowMass/salt3_lc_variance_0.dat','w') as fout:
            for p,w,m in zip(m0phase,m0wave,m0lowmassvarflux):
                print(f'{p:.1f} {w:.2f} {m:8.15e}',file=fout)
        with open(f'{modelname}HighMass/salt3_lc_variance_1.dat','w') as fout:
            for p,w,m in zip(m0phase,m0wave,m1highmassvarflux):
                print(f'{p:.1f} {w:.2f} {m:8.15e}',file=fout)
        with open(f'{modelname}LowMass/salt3_lc_variance_1.dat','w') as fout:
            for p,w,m in zip(m0phase,m0wave,m1lowmassvarflux):
                print(f'{p:.1f} {w:.2f} {m:8.15e}',file=fout)
        with open(f'{modelname}HighMass/salt3_lc_covariance_01.dat','w') as fout:
            for p,w,m in zip(m0phase,m0wave,m01highmasscovarflux):
                print(f'{p:.1f} {w:.2f} {m:8.15e}',file=fout)
        with open(f'{modelname}LowMass/salt3_lc_covariance_01.dat','w') as fout:
            for p,w,m in zip(m0phase,m0wave,m01lowmasscovarflux):
                print(f'{p:.1f} {w:.2f} {m:8.15e}',file=fout)

                
        return


    def snlc_fit(self,clobber=False,dofit=True,dosalt2mu=True):
        # let's just fit everything with both SN models and then concat the FITRES files
        # only keeping high vs. low-mass for different ones

        # tmp flag
        clobber = True
        do_snlcfit = True
        # surveys, filters, kcors
        all_versions = ['SALT3TRAIN_J21_Hamuy1996','SALT3TRAIN_J21_Riess1999','SALT3TRAIN_J21_Jha2006',
                        'SALT3TRAIN_J21_Hicken2009','SALT3TRAIN_J21_CfA4p1','SALT3TRAIN_J21_CfA4p2',
                        'SALT3TRAIN_J21_CSPDR3','SALT3TRAIN_J21_OTHER_LOWZ','SALT3TRAIN_J21_Foundation_DR1']
        all_kcors = ['kcor/SALT3TRAIN_Fragilistic/kcor_Hamuy1996.fits',
                     'kcor/SALT3TRAIN_Fragilistic/kcor_Riess1999.fits',
                     'kcor/SALT3TRAIN_Fragilistic/kcor_Jha2006.fits',
                     'kcor/SALT3TRAIN_Fragilistic/kcor_Hicken2009.fits',
                     'kcor/SALT3TRAIN_Fragilistic/kcor_CfA4p1.fits',
                     'kcor/SALT3TRAIN_Fragilistic/kcor_CfA4p2.fits',
                     'kcor/SALT3TRAIN_Fragilistic/kcor_CSPDR3.fits',
                     'kcor/SALT3TRAIN_Fragilistic/kcor_OTHER_LOWZ.fits',
                     'kcor/SALT3TRAIN_Fragilistic/kcor_Foundation_DR1.fits']
        all_filtlists = ['BVRI','BVRI','BVRI','ABIRVbcdehjkluvwxLC','DEFG','PQWT','AuvxLCw','ABIRVhjkltuvLC','griz']
        
        # do the fitting
        for model in ['K21','SALT3Models/SALT3.HostFixed','SALT3Models/SALT3.Host','SALT3Models/SALT3.NoHost',
                      'SALT3Models/SALT3.HostNoErrs','SALT3Models/SALT3.HostFixedNoErrs','SALT3Models/SALT3.NoHostNoErrs']:
            if dofit:
                prefix = model.split('.')[-1]
                if do_snlcfit:
                    if clobber:
                        os.system(f"rm fitres/*")
                    for massmodel in ['HighMass','LowMass']:
                        for version,kcor,filtlist,outfile in zip(all_versions,all_kcors,all_filtlists,all_versions):

                            nmltext = _nmltmpl.replace('<data_version>',version).\
                                      replace('<kcor>',kcor).\
                                      replace('<filtlist>',filtlist).\
                                      replace('<outfile>','fitres/'+outfile+'_'+model.split('/')[-1]+massmodel)
                            if model != 'K21' and model != 'JLA-B14':
                                nmltext = nmltext.replace('<model>',os.getcwd()+'/'+model+massmodel)
                            elif model == 'K21':
                                nmltext = nmltext.replace('<model>','SALT3.K21') #model)
                            else:
                                nmltext = nmltext.replace('<model>','SALT2.JLA-B14')
                            with open('tmp.nml','w') as fout:
                                print(nmltext,file=fout)

                            os.system(f'snlc_fit.exe tmp.nml')

                        #import pdb; pdb.set_trace()
                # now catenate everything together but only high- vs. low-mass
                spi = txtobj('SALT3_PARS_INIT_HOSTMASS.LIST')

                for ext_single,ext_comb,linestart in zip(['','.LCPLOT.TEXT'],['.FITRES.TEXT','.LCPLOT.TEXT'],['SN','OBS']):
                    fitres_files_highmass = glob.glob(f'fitres/*HighMass'+ext_single)
                    with open(f'fitres_combined/{prefix}_HighMass_Combined'+ext_comb,'w') as fout:
                        pass
                    for i,ff in enumerate(fitres_files_highmass):
                        with open(f'fitres_combined/{prefix}_HighMass_Combined'+ext_comb,'a') as fout:
                            print(ff)
                            if 'Combined' in ff and i == 0: raise RuntimeError('bleh!')
                            if 'Combined' in ff: continue
                            with open(ff) as fin:
                                for line in fin:
                                    line = line.replace('\n','')
                                    if line.startswith('VARNAMES') and i == 0:
                                        print(line,file=fout)
                                    elif line.startswith(f'{linestart}:'):
                                        snid = line.split()[1]
                                        if snid in spi.SNID:
                                            iMass = spi.SNID == snid
                                            if spi.xhost[iMass] > 0:
                                                print(line,file=fout)

                    fitres_files_lowmass = glob.glob('fitres/*LowMass'+ext_single)
                    with open(f'fitres_combined/{prefix}_LowMass_Combined'+ext_comb,'w') as fout:
                        for i,ff in enumerate(fitres_files_lowmass):
                            if 'Combined' in ff and i == 0: raise RuntimeError('bleh!')
                            if 'Combined' in ff: continue
                            with open(ff) as fin:
                                for line in fin:
                                    line = line.replace('\n','')
                                    if line.startswith('VARNAMES') and i == 0:
                                        print(line,file=fout)
                                    elif line.startswith(f'{linestart}:'):
                                        snid = line.split()[1]
                                        if snid in spi.SNID:
                                            iMass = spi.SNID == snid
                                            if spi.xhost[iMass] < 0:
                                                print(line,file=fout)

                    with open(f'fitres_combined/{prefix}_AllMass_Combined'+ext_comb,'w') as fout:
                        for i,ff in enumerate([f'fitres_combined/{prefix}_LowMass_Combined'+ext_comb,f'fitres_combined/{prefix}_HighMass_Combined'+ext_comb]):
                            with open(ff) as fin:
                                for line in fin:
                                    line = line.replace('\n','')
                                    if line.startswith('VARNAMES') and i == 0:
                                        print(line,file=fout)
                                    elif line.startswith(f'{linestart}:'):
                                        print(line,file=fout)

            if dosalt2mu:
                # now it's SALT2mu time
                prefix = model.split('.')[-1]
                os.system(f'SALT2mu.exe SALT2mu.default file=fitres_combined/{prefix}_HighMass_Combined.FITRES.TEXT prefix=fitres_combined/{prefix}_HighMass_Combined_SALT2mu')
                os.system(f'SALT2mu.exe SALT2mu.default file=fitres_combined/{prefix}_LowMass_Combined.FITRES.TEXT prefix=fitres_combined/{prefix}_LowMass_Combined_SALT2mu')
                os.system(f'SALT2mu.exe SALT2mu.default file=fitres_combined/{prefix}_AllMass_Combined.FITRES.TEXT prefix=fitres_combined/{prefix}_AllMass_Combined_SALT2mu')

        return

    def plot_hrs_threepanel(self):
        plt.rcParams['figure.figsize'] = (12,8)

        # matplotlib boredom
        plt.subplots_adjust(wspace=0)
        fig = plt.figure()
        gs = GridSpec(3, 5) #, figure=fig)
        gs.update(wspace=0.0, hspace=0.0,bottom=0.2)

        # let's get the initial CID list so we're comparing apples to apples
        allmassfile = f'fitres_combined/Host_AllMass_Combined_SALT2mu.fitres'
        frcid = txtobj(allmassfile,fitresheader=True)
        frcid = getmu.mkcuts(frcid,fitprobmin=0).CID

        labeldict = {'NoHost':'No Host','HostFixed':'Host, M$_0$/M$_1$ Fixed in Training','Host':'Host'}

        for i,prefix in enumerate(['NoHost','HostFixed','Host']):
            # read in the files
            lowmassfile = f'fitres_combined/{prefix}_LowMass_Combined_SALT2mu.fitres'
            highmassfile = f'fitres_combined/{prefix}_HighMass_Combined_SALT2mu.fitres'
            allmassfile = f'fitres_combined/{prefix}_AllMass_Combined_SALT2mu.fitres'
            allmassnoerrfile = f'fitres_combined/{prefix}NoErrs_AllMass_Combined_SALT2mu.fitres'
            
            frlowmass = txtobj(lowmassfile,fitresheader=True)
            frhighmass = txtobj(highmassfile,fitresheader=True)
            frall = txtobj(allmassfile,fitresheader=True)
            frnoerr = txtobj(allmassnoerrfile,fitresheader=True)
            
            frlowmass = getmu.mkcuts(frlowmass,fitprobmin=0)
            frhighmass = getmu.mkcuts(frhighmass,fitprobmin=0)          
            frall = getmu.mkcuts(frall,fitprobmin=0)
            frnoerr = getmu.mkcuts(frnoerr,fitprobmin=0)

            # nuisance parameters
            for fr,frfile in zip([frall,frlowmass,frhighmass,frnoerr],[allmassfile,lowmassfile,highmassfile,allmassnoerrfile]):
                with open(frfile) as fin:
                    for line in fin:
                        if line.startswith('#  alpha0'): salt2alpha = float(line.split()[3])
                        if line.startswith('#  beta0'): salt2beta = float(line.split()[3])
                fr = getmu.getmu(fr,salt2alpha,salt2beta=salt2beta)

                goodcid = np.array([],dtype=int)
                for j,si in enumerate(fr.CID):
                    if si in frcid:
                        goodcid = np.append(goodcid,j)
                for k in fr.__dict__.keys():
                    fr.__dict__[k] = fr.__dict__[k][goodcid]

                if frfile == allmassfile: muresmedian = np.median(frall.mures)
                fr.mures -= muresmedian
                fr.salt2alpha = salt2alpha; fr.salt2beta = salt2beta

            # make the subplot
            ax = fig.add_subplot(gs[i, 0:4])
            axhist = fig.add_subplot(gs[i, 4])
            ax.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
            axhist.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)

            # plot HRs and bins
            zbins = np.logspace(np.log10(0.01),np.log10(1.0),20)
            salt3mubins = binned_statistic(frlowmass.zCMB,frlowmass.mures,bins=zbins,statistic='mean').statistic
            salt3muerrbins = binned_statistic(frlowmass.zCMB,frlowmass.mures,bins=zbins,statistic=errfnc).statistic
            ax.axhline(0,lw=2,color='k')
            ax.errorbar(frlowmass.zCMB,frlowmass.mures,yerr=frlowmass.MUERR,fmt='o',color='b',alpha=0.1)
            ax.errorbar((zbins[1:]+zbins[:-1])/2.,salt3mubins,yerr=salt3muerrbins,fmt='o-',color='b')
            muresbins = np.linspace(-1,1,40)
            axhist.hist(frlowmass.mures,bins=muresbins,orientation='horizontal',color='b',alpha=0.5)

            salt3mubins = binned_statistic(frhighmass.zCMB,frhighmass.mures,bins=zbins,statistic='mean').statistic
            salt3muerrbins = binned_statistic(frhighmass.zCMB,frhighmass.mures,bins=zbins,statistic=errfnc).statistic
            ax.axhline(0,lw=2,color='k')
            ax.errorbar(frhighmass.zCMB,frhighmass.mures,yerr=frhighmass.MUERR,fmt='o',color='r',alpha=0.1)
            ax.errorbar((zbins[1:]+zbins[:-1])/2.,salt3mubins,yerr=salt3muerrbins,fmt='o-',color='r')
            muresbins = np.linspace(-1,1,40)
            axhist.hist(frhighmass.mures,bins=muresbins,orientation='horizontal',color='r',alpha=0.5)

            # summary stats
            #ax.text(0.01,0.99,fr"""log($M_{{\ast}}/M_{{\odot}}$) < 10 RMS = {np.std(frlowmass.mures):.3f}""",
            #        ha='left',va='top',transform=ax.transAxes,color='b')
            #ax.text(0.01,0.9,fr"""log($M_{{\ast}}/M_{{\odot}}$) > 10 RMS = {np.std(frhighmass.mures):.3f}""",
            #        ha='left',va='top',transform=ax.transAxes,color='r')

            # mass step
            md = minimize(lnlikefunc,(0.0,0.0),
                          args=(np.append([0.0]*len(frlowmass.mures),[1.0]*len(frhighmass.mures)),
                                np.append(frlowmass.mures,frhighmass.mures),np.append(frlowmass.MUERR,frhighmass.MUERR),None))
            mass_step = -md.x[1]; mass_steperr = np.sqrt(md.hess_inv[1,1])
            #import pdb; pdb.set_trace()
            chi2 = np.median(frnoerr.FITCHI2/frnoerr.NDOF)
            #chi2 = sigma_clipped_stats(frnoerr.FITCHI2/frnoerr.NDOF)[1]
            ax.text(0.01,0.01,f"Mass Step = ${mass_step:.3f}\pm{mass_steperr:.3f}$ mag",transform=ax.transAxes,ha='left',va='bottom')
            ax.text(0.01,0.1,fr"$\chi^2_{{\nu}}$ = {chi2:.3f}",transform=ax.transAxes,ha='left',va='bottom')
            ax.text(0.01,0.22,f"RMS = {np.std(np.append(frlowmass.MURES,frhighmass.MURES)):.3f} mag",transform=ax.transAxes,ha='left',va='bottom')
            import pdb; pdb.set_trace()
            # matplotlib things
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            ax.xaxis.set_ticks([0.02,0.05,0.1])
            ax.xaxis.set_ticklabels(['0.02','0.05','0.1'])
            ax.set_ylim([-0.5,0.5])
            axhist.set_ylim([-0.5,0.5])
            ax.yaxis.set_ticks([-0.4,-0.2,0.0,0.2,0.4])
            axhist.yaxis.set_ticks([-0.4,-0.2,0.0,0.2,0.4])
            axhist.yaxis.tick_right()

            ax.set_ylabel(f'{labeldict[prefix]}\n$\mu - \mu_{{\Lambda CDM}}$')

        ax.set_xlabel('$z_{CMB}$',fontsize=15)
        plt.savefig('masshubbleresids.png',dpi=200)

    def model_table(self):

        # let's get the initial CID list so we're comparing apples to apples
        allmassfile = f'fitres_combined/Host_AllMass_Combined_SALT2mu.fitres'
        frcid = txtobj(allmassfile,fitresheader=True)
        frcid = getmu.mkcuts(frcid,fitprobmin=0).CID

        for i,prefix in enumerate(['NoHost','HostFixed','Host']):
            # read in the files
            lowmassfile = f'fitres_combined/{prefix}_LowMass_Combined_SALT2mu.fitres'
            highmassfile = f'fitres_combined/{prefix}_HighMass_Combined_SALT2mu.fitres'
            allmassfile = f'fitres_combined/{prefix}_AllMass_Combined_SALT2mu.fitres'
            lowmassnoerrfile = f'fitres_combined/{prefix}NoErrs_LowMass_Combined_SALT2mu.fitres'
            highmassnoerrfile = f'fitres_combined/{prefix}NoErrs_HighMass_Combined_SALT2mu.fitres'
            allmassnoerrfile = f'fitres_combined/{prefix}NoErrs_AllMass_Combined_SALT2mu.fitres'
            
            frlowmass = txtobj(lowmassfile,fitresheader=True)
            frhighmass = txtobj(highmassfile,fitresheader=True)
            frall = txtobj(allmassfile,fitresheader=True)
            frlowmassnoerr = txtobj(lowmassnoerrfile,fitresheader=True)
            frhighmassnoerr = txtobj(highmassnoerrfile,fitresheader=True)
            frnoerr = txtobj(allmassnoerrfile,fitresheader=True)

            frlowmass = getmu.mkcuts(frlowmass,fitprobmin=0,trestwarn=False)
            frhighmass = getmu.mkcuts(frhighmass,fitprobmin=0,trestwarn=False)          
            frall = getmu.mkcuts(frall,fitprobmin=0,trestwarn=False)
            frlowmassnoerr = getmu.mkcuts(frlowmassnoerr,fitprobmin=0,trestwarn=False)
            frhighmassnoerr = getmu.mkcuts(frhighmassnoerr,fitprobmin=0,trestwarn=False)
            frnoerr = getmu.mkcuts(frnoerr,fitprobmin=0,trestwarn=False)

            # nuisance parameters
            for fr,frfile in zip([frall,frlowmass,frhighmass,frnoerr,frlowmassnoerr,frhighmassnoerr],
                                 [allmassfile,lowmassfile,highmassfile,allmassnoerrfile,lowmassnoerrfile,highmassnoerrfile]):
                with open(frfile) as fin:
                    for line in fin:
                        if line.startswith('#  alpha0'): 
                            salt2alpha = float(line.split()[3])
                            salt2alphaerr = float(line.split()[5])
                        if line.startswith('#  beta0'): 
                            salt2beta = float(line.split()[3])
                            salt2betaerr = float(line.split()[5])
                fr = getmu.getmu(fr,salt2alpha=salt2alpha,salt2beta=salt2beta)

                goodcid = np.array([],dtype=int)
                for j,si in enumerate(fr.CID):
                    if si in frcid:
                        goodcid = np.append(goodcid,j)
                for k in fr.__dict__.keys():
                    fr.__dict__[k] = fr.__dict__[k][goodcid]

                if frfile == allmassfile: muresmedian = np.median(frall.mures)
                fr.mures -= muresmedian
                fr.salt2alpha = salt2alpha; fr.salt2beta = salt2beta
                fr.salt2alphaerr = salt2alphaerr; fr.salt2betaerr = salt2betaerr

            # summary stats
            #ax.text(0.01,0.99,fr"""log($M_{{\ast}}/M_{{\odot}}$) < 10 RMS = {np.std(frlowmass.mures):.3f}""",
            #        ha='left',va='top',transform=ax.transAxes,color='b')
            #ax.text(0.01,0.9,fr"""log($M_{{\ast}}/M_{{\odot}}$) > 10 RMS = {np.std(frhighmass.mures):.3f}""",
            #        ha='left',va='top',transform=ax.transAxes,color='r')

            # mass step
            md = minimize(lnlikefunc,(0.0,0.0),
                          args=(np.append([0.0]*len(frlowmass.mures),[1.0]*len(frhighmass.mures)),
                                np.append(frlowmass.mures,frhighmass.mures),np.append(frlowmass.MUERR,frhighmass.MUERR),None))
            mass_step = -md.x[1]; mass_steperr = np.sqrt(md.hess_inv[1,1])

            chi2 = np.median(frnoerr.FITCHI2/frnoerr.NDOF)
            rms_all = sigma_clipped_stats(np.append(frlowmass.MURES,frhighmass.MURES))[2]
            rms_lowmass = sigma_clipped_stats(frlowmass.MURES)[2]
            rms_highmass = sigma_clipped_stats(frhighmass.MURES)[2]

            rms_all = np.std(np.append(frlowmass.MURES,frhighmass.MURES))
            rms_lowmass = np.std(frlowmass.MURES)
            rms_highmass = np.std(frhighmass.MURES)

            #chi2 = sigma_clipped_stats(frnoerr.FITCHI2/frnoerr.NDOF)[1]
            #ax.text(0.01,0.01,f"Mass Step = ${mass_step:.3f}\pm{mass_steperr:.3f}$ mag",transform=ax.transAxes,ha='left',va='bottom')
            #ax.text(0.01,0.1,fr"$\chi^2_{{\nu}}$ = {chi2:.3f}",transform=ax.transAxes,ha='left',va='bottom')
            #ax.text(0.01,0.22,f"RMS = {sigma_clipped_stats(np.append(frlowmass.MURES,frhighmass.MURES))[2]:.3f} mag",transform=ax.transAxes,ha='left',va='bottom')

            modelline_all = f"SALT3.{prefix}&{rms_all:.3f}&{np.median(frnoerr.FITCHI2/frnoerr.NDOF):.2f}&${salt2alpha:.3f}\\pm{salt2alphaerr:.3f}$&${salt2beta:.3f}\\pm{salt2betaerr:.3f}$&${mass_step:.3f}\\pm{mass_steperr:.3f}$\\\\"
            modelline_lowmass = f"$-$ $\\rm log(M_{{\\ast}}/M_{{\\odot}}) < 10$&{rms_lowmass:.3f}&{np.median(frlowmassnoerr.FITCHI2/frlowmassnoerr.NDOF):.2f}&${frlowmass.salt2alpha:.3f}\\pm{frlowmass.salt2alphaerr:.3f}$&${frlowmass.salt2beta:.3f}\\pm{frlowmass.salt2betaerr:.3f}$&\\nodata\\\\"
            modelline_highmass = f"$-$ $\\rm log(M_{{\\ast}}/M_{{\\odot}}) > 10$&{rms_highmass:.3f}&{np.median(frhighmassnoerr.FITCHI2/frhighmassnoerr.NDOF):.2f}&${frhighmass.salt2alpha:.3f}\\pm{frhighmass.salt2alphaerr:.3f}$&${frhighmass.salt2beta:.3f}\\pm{frhighmass.salt2betaerr:.3f}$&\\nodata\\\\"
            print(modelline_all); print(modelline_lowmass); print(modelline_highmass)

            
    def plot_hrs(self):

        plt.rcParams['figure.figsize'] = (12,6)

        # now we need to get hubble residual scatters and host mass steps for everything
        frlowmass = txtobj('fitres/LowMass_Combined_SALT2mu.fitres',fitresheader=True)
        frhighmass = txtobj('fitres/HighMass_Combined_SALT2mu.fitres',fitresheader=True)
        frk21lowmass = txtobj('fitres/lowz_nohost_LowMass_Combined_SALT2mu.fitres',fitresheader=True)
        frk21highmass = txtobj('fitres/lowz_nohost_HighMass_Combined_SALT2mu.fitres',fitresheader=True)

        frlowmass = getmu.mkcuts(frlowmass,fitprobmin=0)#,cmax=0.3)
        frhighmass = getmu.mkcuts(frhighmass,fitprobmin=0)#,cmax=0.3)
        frk21lowmass = getmu.mkcuts(frk21lowmass,fitprobmin=0)#,cmax=0.3)
        frk21highmass = getmu.mkcuts(frk21highmass,fitprobmin=0)#,cmax=0.3)
        with open('fitres/LowMass_Combined_SALT2mu.fitres') as fin:
            for line in fin:
                if line.startswith('#  alpha0'): salt2alpha = float(line.split()[3])
                if line.startswith('#  beta0'): salt2beta = float(line.split()[3])
        print(salt2alpha,salt2beta)
        frlowmass.MURES = getmu.getmu(frlowmass,salt2alpha=salt2alpha,salt2beta=salt2beta).mures
        with open('fitres/HighMass_Combined_SALT2mu.fitres') as fin:
            for line in fin:
                if line.startswith('#  alpha0'): salt2alpha = float(line.split()[3])
                if line.startswith('#  beta0'): salt2beta = float(line.split()[3])
        #print(salt2alpha,salt2beta)
        frhighmass.MURES = getmu.getmu(frhighmass,salt2alpha=salt2alpha,salt2beta=salt2beta).mures
        with open('fitres/K21_Combined_SALT2mu.fitres') as fin:
            for line in fin:
                if line.startswith('#  alpha0'): salt2alpha = float(line.split()[3])
                if line.startswith('#  beta0'): salt2beta = float(line.split()[3])
        print(salt2alpha,salt2beta)
        frk21lowmass.MURES = getmu.getmu(frk21lowmass,salt2alpha=salt2alpha,salt2beta=salt2beta).mures
        #with open('fitres/K21_HighMass_Combined_SALT2mu.fitres') as fin:
        #    for line in fin:
        #        if line.startswith('#  alpha0'): salt2alpha = float(line.split()[3])
        #        if line.startswith('#  beta0'): salt2beta = float(line.split()[3])
        #print(salt2alpha,salt2beta)
        frk21highmass.MURES = getmu.getmu(frk21highmass,salt2alpha=salt2alpha,salt2beta=salt2beta).mures


        magoff = np.median(np.append(frk21lowmass.MURES,frk21highmass.MURES))
        frk21lowmass.MURES -= magoff
        frk21highmass.MURES -= magoff
        frlowmass.MURES -= magoff
        frhighmass.MURES -= magoff

        #import pdb; pdb.set_trace()
        #frlowmass.MURES = frlowmass.MU - cosmo.mu(frlowmass.zHD)
        #frhighmass.MURES = frhighmass.MU - cosmo.mu(frhighmass.zHD)
        #frk21lowmass.MURES = frk21lowmass.MU - cosmo.mu(frk21lowmass.zHD)
        #frk21highmass.MURES = frk21highmass.MU - cosmo.mu(frk21highmass.zHD)

        # matplotlib boredom
        plt.subplots_adjust(wspace=0)
        fig = plt.figure()
        gs = GridSpec(2, 5) #, figure=fig)
        gs.update(wspace=0.0, hspace=0.0,bottom=0.2)
        ax1 = fig.add_subplot(gs[0, 0:4])
        ax1hist = fig.add_subplot(gs[0, 4])
        ax1.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
        ax1hist.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
        ax2 = fig.add_subplot(gs[1, 0:4])
        ax2hist = fig.add_subplot(gs[1, 4])
        ax2.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
        ax2hist.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)

        # top plot - K21
        zbins = np.logspace(np.log10(0.01),np.log10(1.0),20)
        salt3mubins = binned_statistic(frk21lowmass.zCMB,frk21lowmass.MURES,bins=zbins,statistic='mean').statistic
        salt3muerrbins = binned_statistic(frk21lowmass.zCMB,frk21lowmass.MURES,bins=zbins,statistic=errfnc).statistic
        ax1.axhline(0,lw=2,color='k')
        ax1.errorbar(frk21lowmass.zCMB,frk21lowmass.MURES,yerr=frk21lowmass.MUERR,fmt='o',color='b',alpha=0.1)
        ax1.errorbar((zbins[1:]+zbins[:-1])/2.,salt3mubins,yerr=salt3muerrbins,fmt='o-',color='b')
        muresbins = np.linspace(-1,1,40)
        ax1hist.hist(frk21lowmass.MURES,bins=muresbins,orientation='horizontal',color='b',alpha=0.5)

        salt3mubins = binned_statistic(frk21highmass.zCMB,frk21highmass.MURES,bins=zbins,statistic='mean').statistic
        salt3muerrbins = binned_statistic(frk21highmass.zCMB,frk21highmass.MURES,bins=zbins,statistic=errfnc).statistic
        ax1.axhline(0,lw=2,color='k')
        ax1.errorbar(frk21highmass.zCMB,frk21highmass.MURES,yerr=frk21highmass.MUERR,fmt='o',color='r',alpha=0.1)
        ax1.errorbar((zbins[1:]+zbins[:-1])/2.,salt3mubins,yerr=salt3muerrbins,fmt='o-',color='r')
        muresbins = np.linspace(-1,1,40)
        ax1hist.hist(frk21highmass.MURES,bins=muresbins,orientation='horizontal',color='r',alpha=0.5)


        # bottom plot - low and high-mass
        zbins = np.logspace(np.log10(0.01),np.log10(1.0),20)

        salt3mubins = binned_statistic(frlowmass.zCMB,frlowmass.MURES,bins=zbins,statistic='mean').statistic
        salt3muerrbins = binned_statistic(frlowmass.zCMB,frlowmass.MURES,bins=zbins,statistic=errfnc).statistic
        ax2.axhline(0,lw=2,color='k')
        ax2.errorbar(frlowmass.zCMB,frlowmass.MURES,yerr=frlowmass.MUERR,fmt='o',color='b',alpha=0.1)
        ax2.errorbar((zbins[1:]+zbins[:-1])/2.,salt3mubins,yerr=salt3muerrbins,fmt='o-',color='b')
        muresbins = np.linspace(-1,1,40)
        ax2hist.hist(frlowmass.MURES,bins=muresbins,orientation='horizontal',color='b',alpha=0.5)

        salt3mubins = binned_statistic(frhighmass.zCMB,frhighmass.MURES,bins=zbins,statistic='mean').statistic
        salt3muerrbins = binned_statistic(frhighmass.zCMB,frhighmass.MURES,bins=zbins,statistic=errfnc).statistic
        ax2.axhline(0,lw=2,color='k')
        ax2.errorbar(frhighmass.zCMB,frhighmass.MURES,yerr=frhighmass.MUERR,fmt='o',color='r',alpha=0.1)
        ax2.errorbar((zbins[1:]+zbins[:-1])/2.,salt3mubins,yerr=salt3muerrbins,fmt='o-',color='r')
        muresbins = np.linspace(-1,1,40)
        ax2hist.hist(frhighmass.MURES,bins=muresbins,orientation='horizontal',color='r',alpha=0.5)

        for ax in [ax1,ax2]:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            ax.xaxis.set_ticks([0.02,0.05,0.1])
            ax.xaxis.set_ticklabels(['0.02','0.05','0.1'])

        for ax in [ax1,ax1hist,ax2,ax2hist]:
            ax.set_ylim([-0.5,0.5])
        ax1hist.yaxis.tick_right()
        ax2hist.yaxis.tick_right()

        ax1.set_ylabel('$\mu - \mu_{\Lambda CDM}$')
        ax2.set_ylabel('$\mu - \mu_{\Lambda CDM}$')
        ax2.set_xlabel('$z_{CMB}$')

        #ax1.text(0.01,0.99,f"""RMS = {np.std(frk21.MURES):.3f}""",ha='left',va='top',transform=ax1.transAxes)
        ax1.text(0.01,0.99,fr"""log($M_{{\ast}}/M_{{\odot}}$) < 10 RMS = {np.std(frk21lowmass.MURES):.3f}""",
                 ha='left',va='top',transform=ax1.transAxes,color='b')
        ax1.text(0.01,0.9,fr"""log($M_{{\ast}}/M_{{\odot}}$) > 10 RMS = {np.std(frk21highmass.MURES):.3f}""",
                 ha='left',va='top',transform=ax1.transAxes,color='r')

        
        md = minimize(lnlikefunc,(0.0,0.0),
                      args=(np.append([0.0]*len(frk21lowmass.MURES),[1.0]*len(frk21highmass.MURES)),
                            np.append(frk21lowmass.MURES,frk21highmass.MURES),np.append(frk21lowmass.MUERR,frk21highmass.MUERR),None))

        mass_step = -md.x[1]; mass_steperr = np.sqrt(md.hess_inv[1,1])
        #mass_step = np.median(frk21lowmass.MURES)-np.median(frk21highmass.MURES)
        chi2 = np.median(np.append(frk21highmass.FITCHI2/frk21highmass.NDOF,frk21lowmass.FITCHI2/frk21lowmass.NDOF))
        ax1.text(0.01,0.01,f"Mass Step = ${mass_step:.3f}\pm{mass_steperr:.3f}$ mag",transform=ax1.transAxes,ha='left',va='bottom')
        ax1.text(0.01,0.1,fr"$\chi^2_{{\nu}}$ = {chi2:.3f}",transform=ax1.transAxes,ha='left',va='bottom')

        ax2.text(0.01,0.99,fr"""log($M_{{\ast}}/M_{{\odot}}$) < 10 RMS = {np.std(frlowmass.MURES):.3f}""",
                 ha='left',va='top',transform=ax2.transAxes,color='b')
        ax2.text(0.01,0.9,fr"""log($M_{{\ast}}/M_{{\odot}}$) > 10 RMS = {np.std(frhighmass.MURES):.3f}""",
                 ha='left',va='top',transform=ax2.transAxes,color='r')
        md = minimize(lnlikefunc,(0.0,0.01),
            args=(np.append([0.0]*len(frlowmass.MURES),[1.0]*len(frhighmass.MURES)),
            np.append(frlowmass.MURES,frhighmass.MURES),np.append(frlowmass.MUERR,frhighmass.MUERR),None))
        mass_step = -md.x[1]; mass_steperr = np.sqrt(md.hess_inv[1,1])

        #mass_step = np.median(frlowmass.MURES)-np.median(frhighmass.MURES)
        chi2 = np.median(np.append(frhighmass.FITCHI2/frhighmass.NDOF,frlowmass.FITCHI2/frlowmass.NDOF))
        ax2.text(0.01,0.01,f"Mass Step = ${mass_step:.3f}\pm{mass_steperr:.3f}$ mag",transform=ax2.transAxes,ha='left',va='bottom')
        ax2.text(0.01,0.1,fr"$\chi^2_{{\nu}}$ = {chi2:.3f}",transform=ax2.transAxes,ha='left',va='bottom')

        plt.savefig('masshubbleresids.png')
        import pdb; pdb.set_trace()
        return
        
def lnlikefunc(x,p_iae=None,mu_i=None,sigma_i=None,sigma=None,z=None,survey=None):

    #if sigma or sigma == 0.0:
        # fix the dispersion
    #    x[2] = sigma; x[3] = sigma
    
    p_iae[np.where(p_iae == 0)] == 1e-4
    return -np.sum(np.logaddexp(-(mu_i-x[0])**2./(2.0*(sigma_i**2.)) +\
        np.log((1-p_iae)/(np.sqrt(2*np.pi)*np.sqrt(sigma_i**2.))),
        -(mu_i-x[1]-x[0])**2./(2.0*(sigma_i**2.)) +\
        np.log((p_iae)/(np.sqrt(2*np.pi)*np.sqrt(sigma_i**2.)))))

def errfnc(x):
    return(np.std(x)/np.sqrt(len(x)))

def colorscat():

    frlowmass = txtobj('fitres_combined/Host_LowMass_Combined_SALT2mu.fitres',fitresheader=True)
    frhighmass = txtobj('fitres_combined/Host_HighMass_Combined_SALT2mu.fitres',fitresheader=True)
    frk21lowmass = txtobj('fitres_combined/NoHost_LowMass_Combined_SALT2mu.fitres',fitresheader=True)
    frk21highmass = txtobj('fitres_combined/NoHost_HighMass_Combined_SALT2mu.fitres',fitresheader=True)

    frlowmass = getmu.mkcuts(frlowmass,fitprobmin=0)
    frhighmass = getmu.mkcuts(frhighmass,fitprobmin=0)
    frk21lowmass = getmu.mkcuts(frk21lowmass,fitprobmin=0)
    frk21highmass = getmu.mkcuts(frk21highmass,fitprobmin=0)

    from scipy.stats import bootstrap
    def bootstrap_errs(x):
        if len(x) < 2:
            return 0
        result = bootstrap((x,),np.std,n_resamples=100)
        return result.standard_error


    bins = np.linspace(-0.2,0.2,5)
    cbins = binned_statistic(np.append(frlowmass.c,frhighmass.c),np.append(frlowmass.MURES,frhighmass.MURES),bins=bins,statistic='std').statistic
    cbinsk21 = binned_statistic(np.append(frk21lowmass.c,frk21highmass.c),
                                np.append(frk21lowmass.MURES,frk21highmass.MURES),bins=bins,statistic='std').statistic
    cerrbins = binned_statistic(np.append(frlowmass.c,frhighmass.c),np.append(frlowmass.MURES,frhighmass.MURES),bins=bins,statistic=bootstrap_errs).statistic
    cerrbinsk21 = binned_statistic(np.append(frk21lowmass.c,frk21highmass.c),
                                   np.append(frk21lowmass.MURES,frk21highmass.MURES),bins=bins,statistic=bootstrap_errs).statistic

    ax = plt.axes()
    ax.errorbar((bins[1:]+bins[:-1])/2.,cbinsk21,yerr=cerrbinsk21,fmt='o-',label='SALT3.NoHost')
    ax.errorbar((bins[1:]+bins[:-1])/2.,cbins,yerr=cerrbins,fmt='o-',label='SALT3.Host')
    ax.set_ylabel('RMS')
    ax.set_xlabel('$c$')
    ax.legend(loc='lower right',prop={'size':13})
    plt.savefig('colorscat.png')

    import pdb; pdb.set_trace()    
        
if __name__ == "__main__":
    hr = masshubbleresids()
    #hr.main()
    #hr.edit_model()
    #hr.edit_model_list()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hr.main()
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    hr.main()
    #colorscat()
