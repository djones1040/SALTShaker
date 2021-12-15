#!/usr/bin/env python
# D. Jones - 12/15/21

import os
import numpy as np
from saltshaker.util.txtobj import txtobj
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from saltshaker.util import getmu

_nmltmpl = """
  &SNLCINP
     PRIVATE_DATA_PATH = '/Users/David/Dropbox/research/SALTShaker/examples/mass/SALT3TRAIN_K21'

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
        self.edit_model()
        
        # run the LC fitting
        self.snlc_fit()
        
        # SALT2mu to give hubble residuals

        # make some comparison plots

        # measure the mass step, w/ and w/o SALT2mu

    def edit_model(self,modeldir='output'):

        if not os.path.exists('SALT3.HighMass'):
            os.makedirs('SALT3.HighMass')
        if not os.path.exists('SALT3.LowMass'):
            os.makedirs('SALT3.LowMass')

        #os.system(f'cp salt3_template_0.dat.gz ')

        # copy to HighMass dir
        os.system(f'cp {modeldir}/SALT3.INFO SALT3.HighMass/')
        os.system(f'cp {modeldir}/salt3_lc_covariance_01.dat SALT3.HighMass/')
        os.system(f'cp {modeldir}/salt3_color_correction.dat SALT3.HighMass/')
        os.system(f'cp {modeldir}/salt3_lc_variance_0.dat SALT3.HighMass/')
        os.system(f'cp {modeldir}/salt3_template_1.dat SALT3.HighMass/')
        os.system(f'cp {modeldir}/salt3_color_dispersion.dat SALT3.HighMass/')
        os.system(f'cp {modeldir}/salt3_lc_variance_1.dat SALT3.HighMass/')

        # copy to LowMass Dir
        os.system(f'cp {modeldir}/SALT3.INFO SALT3.LowMass/')
        os.system(f'cp {modeldir}/salt3_lc_covariance_01.dat SALT3.LowMass/')
        os.system(f'cp {modeldir}/salt3_color_correction.dat SALT3.LowMass/')
        os.system(f'cp {modeldir}/salt3_lc_variance_0.dat SALT3.LowMass/')
        os.system(f'cp {modeldir}/salt3_template_1.dat SALT3.LowMass/')
        os.system(f'cp {modeldir}/salt3_color_dispersion.dat SALT3.LowMass/')
        os.system(f'cp {modeldir}/salt3_lc_variance_1.dat SALT3.LowMass/')

        # now edit the M0 component for high vs. low-mass SNe
        m0phase,m0wave,m0flux = np.loadtxt(f"{modeldir}/salt3_template_0.dat",unpack=True)
        mhostphase,mhostwave,mhostflux = np.loadtxt(f"{modeldir}/salt3_template_host.dat",unpack=True)

        m0highmassflux = m0flux + mhostflux*0.5 
        m0lowmassflux= m0flux - mhostflux*0.5
        with open('SALT3.HighMass/salt3_template_0.dat','w') as fout:
            for p,w,m in zip(m0phase,m0wave,m0highmassflux):
                print(f'{p:.1f} {w:.2f} {m:8.15e}',file=fout)
        with open('SALT3.LowMass/salt3_template_0.dat','w') as fout:
            for p,w,m in zip(m0phase,m0wave,m0lowmassflux):
                print(f'{p:.1f} {w:.2f} {m:8.15e}',file=fout)

        return

    def snlc_fit(self,clobber=True):
        # let's just fit everything with both SN models and then concat the FITRES files
        # only keeping high vs. low-mass for different ones

        # surveys, filters, kcors
        all_versions = ['SALT3TRAIN_K21_Hamuy1996','SALT3TRAIN_K21_Riess1999','SALT3TRAIN_K21_Jha2006',
                        'SALT3TRAIN_K21_Hicken2009','SALT3TRAIN_K21_CfA4p1','SALT3TRAIN_K21_CfA4p2',
                        'SALT3TRAIN_K21_CSPDR2','SALT3TRAIN_K21_OTHER_LOWZ','SALT3TRAIN_K21_Foundation_DR1']
        all_kcors = ['kcor/kcor_Hamuy1996.fits','kcor/kcor_Riess1999.fits','kcor/kcor_Jha2006.fits','kcor/kcor_Hicken2009.fits',
                     'kcor/kcor_CFA4p1.fits','kcor/kcor_CFA4p2.fits','kcor/kcor_CSPDR2.fits','kcor/kcor_OTHER_LOWZ.fits',
                     'kcor/kcor_Foundation_DR1.fits']
        all_filtlists = ['BVRI','UBVRI','UBVRI','ABIRUVabcdefhjkltuvwxLC','DEFG','PQWT','AtuvxLC','ABIRUVhjkltuvLC','griz']
        
        # do the fitting
        for version,kcor,filtlist,outfile in zip(all_versions,all_kcors,all_filtlists,all_versions):
            if not clobber and os.path.exists(f"{outfile}.FITRES.TEXT"):
                return outfile

            for model in ['HighMass','LowMass','K21']:
                nmltext = _nmltmpl.replace('<data_version>',version).\
                    replace('<kcor>',kcor).\
                    replace('<filtlist>',filtlist).\
                    replace('<outfile>','fitres/'+outfile+'_'+model).\
                    replace('<model>','SALT3.'+model)

                with open('tmp.nml','w') as fout:
                    print(nmltext,file=fout)

                os.system(f'snlc_fit.exe tmp.nml')
                import pdb; pdb.set_trace()
                
        # now catenate everything together but only high- vs. low-mass
        spi = txtobj('SALT3_PARS_INIT_HOSTMASS.LIST')

        fitres_files_highmass = glob.glob('fitres/*HighMass.FITRES.TEXT')
        with open('fitres/HighMass_Combined.FITRES.TEXT','w') as fout:
            for i,ff in enumerate(fitres_files_highmass):
                if 'Combined' in ff and i == 0: raise RuntimeError('bleh!')
                if 'Combined' in ff: continue
                with open(ff) as fin:
                    for line in fin:
                        line = line.replace('\n','')
                        if line.startswith('VARNAMES') and i == 0:
                            print(line,file=fout)
                        elif line.startswith('SN:'):
                            snid = line.split()[1]
                            if snid in spi.SNID:
                                iMass = spi.SNID == snid
                                if spi.xhost[iMass] > 0:
                                    print(line,file=fout)

        fitres_files_lowmass = glob.glob('fitres/*LowMass.FITRES.TEXT')
        with open('fitres/LowMass_Combined.FITRES.TEXT','w') as fout:
            for i,ff in enumerate(fitres_files_highmass):
                if 'Combined' in ff and i == 0: raise RuntimeError('bleh!')
                if 'Combined' in ff: continue
                with open(ff) as fin:
                    for line in fin:
                        line = line.replace('\n','')
                        if line.startswith('VARNAMES') and i == 0:
                            print(line,file=fout)
                        elif line.startswith('SN:'):
                            snid = line.split()[1]
                            if snid in spi.SNID:
                                iMass = spi.SNID == snid
                                if spi.xhost[iMass] < 0:
                                    print(line,file=fout)

        fitres_files_lowmass = glob.glob('fitres/*K21.FITRES.TEXT')
        with open('fitres/K21_Combined.FITRES.TEXT','w') as fout:
            for i,ff in enumerate(fitres_files_highmass):
                if 'Combined' in ff and i == 0: raise RuntimeError('bleh!')
                if 'Combined' in ff: continue
                with open(ff) as fin:
                    for line in fin:
                        line = line.replace('\n','')
                        if line.startswith('VARNAMES') and i == 0:
                            print(line,file=fout)
                        elif line.startswith('SN:'):
                            snid = line.split()[1]
                            if snid in spi.SNID:
                                print(line,file=fout)

                                    
        # now it's SALT2mu time
        os.system('SALT2mu.exe SALT2mu.default file=fitres/HighMass_Combined.FITRES.TEXT prefix=fitres/HighMass_Combined_SALT2mu')
        os.system('SALT2mu.exe SALT2mu.default file=fitres/LowMass_Combined.FITRES.TEXT prefix=fitres/LowMass_Combined_SALT2mu')
        os.system('SALT2mu.exe SALT2mu.default file=fitres/K21_Combined.FITRES.TEXT prefix=fitres/K21_Combined_SALT2mu')

        return

    def plot_hrs(self):

        # now we need to get hubble residual scatters and host mass steps for everything
        frlowmass = txtobj('fitres/LowMass_Combined_SALT2mu',fitresheader=True)
        frlowmass = txtobj('fitres/HighMass_Combined_SALT2mu',fitresheader=True)
        frlowmass = txtobj('fitres/K21_Combined_SALT2mu',fitresheader=True)
        frlowmass = getmu.mkcuts(frlowmass)
        frhighmass = getmu.mkcuts(frhighmass)
        frk21 = getmu.mkcuts(frk21)

        # matplotlib boredom
        plt.subplots_adjust(wspace=0)
        fig = plt.figure()
        gs = GridSpec(2, 5, figure=fig)
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
        salt3mubins = binned_statistic(frk21.zCMB,frk21.MURES,bins=zbins,statistic='mean').statistic
        salt3muerrbins = binned_statistic(frk21.zCMB,frk21.MURES,bins=zbins,statistic=errfnc).statistic
        ax1.axhline(0,lw=2,color='k')
        ax1.errorbar(frk21.zCMB,frk21.MURES,yerr=frk21.MUERR,fmt='o',color='b',alpha=0.1)
        ax1.errorbar((zbins[1:]+zbins[:-1])/2.,salt3mubins,yerr=salt3muerrbins,fmt='o-',color='b')
        ax1.set_xscale('log')
        ax1.xaxis.set_major_formatter(NullFormatter())
        ax1.xaxis.set_minor_formatter(NullFormatter())
        ax1.xaxis.set_ticks([0.01,0.02,0.05,0.1,0.3,0.7])
        ax1.xaxis.set_ticklabels(['0.01','0.02','0.05','0.1','0.3','0.7'])

        muresbins = np.linspace(-1,1,40)
        ax1hist.hist(frk21.MURES,bins=muresbins,orientation='horizontal',color='b',alpha=0.5)

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


        ax2.set_xscale('log')
        ax2.xaxis.set_major_formatter(NullFormatter())
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.xaxis.set_ticks([0.01,0.02,0.05,0.1,0.3,0.7])
        ax2.xaxis.set_ticklabels(['0.01','0.02','0.05','0.1','0.3','0.7'])

        import pdb; pdb.set_trace()
        return
        

def errfnc(x):
    return(np.std(x)/np.sqrt(len(x)))
        
if __name__ == "__main__":
    hr = masshubbleresids()
    hr.main()
