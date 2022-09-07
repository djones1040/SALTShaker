#!/usr/bin/env python
# D. Jones - 6/20/22
# see if the SALT mass step is band-dependent

import os
import glob
import numpy as np
from saltshaker.util.txtobj import txtobj
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from saltshaker.util import getmu
from scipy.stats import binned_statistic
import cosmo
from scipy.optimize import minimize
from astropy.stats import sigma_clipped_stats
import warnings
import copy

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
     SNCCID_LIST   =  <cidlist>
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
     INISTP_PEAKMJD = 0
     INIVAL_PEAKMJD = <inival_peakmjd>
     INISTP_COLOR = 0
     INIVAL_COLOR = <inival_color>
     INISTP_SHAPE = 0
     INIVAL_SHAPE = <inival_shape>

     OPT_XTMW_ERR = 1
     OPT_COVAR_FLUX = 0
     TREST_REJECT  = -15.0, 45.0
     NGRID_PDF     = 0

     FUDGEALL_ITER1_MAXFRAC = 0.02
     FILTLIST_FIT = '<filtlist>'

  &END
"""

_nmltmpl_all = f"""
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

        for filt in 'BVRI':
            self.snlc_fit(filt=filt,outdir=f'fitres_singlefilt/{filt}')

        # measure the mass step, w/ and w/o SALT2mu
        # maximum likelihood calc, alpha/beta from all data
        self.plot_mass_step_allsn()

        #self.plot_mass_step_paper()

    def plot_mass_step_allsn(self):
        # let's make sure nothing crazy is happening with individual HRs
        ax1 = plt.subplot(141)
        ax2 = plt.subplot(142)
        ax3 = plt.subplot(143)
        ax4 = plt.subplot(144)

    def plot_mass_step_paper(self):
        # let's summarize things for the paper

        plt.rcParams['figure.figsize'] = (12,3)
        plt.subplots_adjust(bottom=0.2,wspace=0)
        
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        sedpar = txtobj('hostpars_salt3_lowztraining.txt')


        for filt,label,ax in zip(['V','R','I'],['$V/g$','$R/r$','$I/i$'],[ax1,ax2,ax3]):
            for ix,model in enumerate(['SALT3.NoHost','SALT3.HostFixed','SALT3.Host']):
                with open(f"fitres_combined/{model.split('.')[-1]}_AllMass_Combined_SALT2mu.fitres",'r') as fin:
                    for line in fin:
                        if line.startswith('#  alpha0'): salt2alpha = float(line.split()[3])
                        if line.startswith('#  beta0'): salt2beta = float(line.split()[3])


                frh = txtobj(f'fitres_singlefilt/{filt}/singlefilt_results_{model}_HighMass.FITRES')
                frl = txtobj(f'fitres_singlefilt/{filt}/singlefilt_results_{model}_LowMass.FITRES')

                # combine the low- and high-mass results
                cidfull = np.unique(np.append(frh.CID,frl.CID)) # in case something fails a fitter weirdly
                idxh,idxl = np.array([],dtype=int),np.array([],dtype=int)
                for j,i in enumerate(cidfull):
                    idx = sedpar.snid == i
                    if sedpar.logmass[idx][0] > 10:
                        idxh = np.append(idxh,np.where(frh.CID == i)[0])
                    else:
                        idxl = np.append(idxl,np.where(frl.CID == i)[0])
                #fr = txtobj(f'fitres_combined/NoHost_AllMass_Combined_SALT2mu.fitres',fitresheader=True)
                #import pdb; pdb.set_trace()
                fr = copy.deepcopy(frh)
                for k in frh.__dict__.keys():
                    fr.__dict__[k] = np.append(frl.__dict__[k][idxl],frh.__dict__[k][idxh])

                fr.HOST_LOGMASS = np.zeros(len(fr.CID))
                for j,i in enumerate(fr.CID):
                    idx = sedpar.snid == i
                    fr.HOST_LOGMASS[j] = sedpar.logmass[idx][0]

                fr = getmu.getmu(fr,salt2alpha=salt2alpha,salt2beta=salt2beta)
                fr = getmu.mkcuts(fr,salt2alpha=salt2alpha,salt2beta=salt2beta,fitprobmin=0)
                
                # mass step
                phm = np.zeros(len(fr.CID))
                phm[fr.HOST_LOGMASS > 10] = 1.0
                md = minimize(lnlikefunc,(0.0,0.0),
                              args=(phm,fr.mures,fr.muerr,None))
                mass_step = -md.x[1]; mass_steperr = np.sqrt(md.hess_inv[1,1])

                ax.errorbar(ix,mass_step,yerr=mass_steperr,fmt='o',color='k')

                print(f'filt {filt}, mass_step: {mass_step:.3f} +/- {mass_steperr:.3f}')
                import pdb; pdb.set_trace()
            ax.xaxis.set_ticks([0,1,2])
            ax.xaxis.set_ticklabels(['K21','HostOnly','Host'],rotation=0)
            ax.set_xlim([-1,3])
            ax.set_ylim([0,0.1])

            ax.axhline(0.049,color='k',ls='--')
            ax.fill_between(np.linspace(-1,3,100),[0.049-0.014]*100,[0.049+0.014]*100,color='0.5',alpha=0.5)
            
        ax1.set_ylabel('Mass Step (mag)')
        ax2.yaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])
        ax1.set_title('$V/g$')
        ax2.set_title('$R/r$')
        ax3.set_title('$I/i$')
        plt.savefig('tmp.png')
        import pdb; pdb.set_trace()
                    
    def snlc_fit(self,filt=None,clobber=False,dofit=True,dosalt2mu=True,outdir='fitres'):
        # let's just fit everything with both SN models and then concat the FITRES files
        # only keeping high vs. low-mass for different ones

        # tmp flag
        clobber = True

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
        all_filtlists = ['BVRI','UBVRI','UBVRI','ABIRUVabcdefhjkltuvwxLC','DEFG','PQWT','AtuvxLCw','ABIRUVhjkltuvLC','griz']

        # dictionary to give filter letter corresponding to B, V/g R/r I/i
        filtdict_version = {'SALT3TRAIN_J21_Hamuy1996':{'B':'B','V':'V','R':'R','I':'I'},
                            'SALT3TRAIN_J21_Riess1999':{'B':'B','V':'V','R':'R','I':'I'},
                            'SALT3TRAIN_J21_Jha2006':{'B':'B','V':'V','R':'R','I':'I'},
                            'SALT3TRAIN_J21_Hicken2009':{'B':'hbB','V':'jcV','R':'kdR','I':'leI'},
                            'SALT3TRAIN_J21_CfA4p1':{'B':'D','V':'E','R':'F','I':'G'},
                            'SALT3TRAIN_J21_CfA4p2':{'B':'P','V':'Q','R':'W','I':'T'},
                            'SALT3TRAIN_J21_CSPDR3':{'B':'u','V':'Axvw','R':'L','I':'C'},
                            'SALT3TRAIN_J21_OTHER_LOWZ':{'B':'Bh','V':'Vj','R':'Rk','I':'Il'},
                            'SALT3TRAIN_J21_Foundation_DR1':{'B':None,'V':'g','R':'r','I':'i'}}
        spi = txtobj('SALT3_PARS_INIT_HOSTMASS.LIST')

        # do the fitting
        for model in ['SALT3Models/SALT3.NoHost','SALT3Models/SALT3.HostFixed','SALT3Models/SALT3.Host','SALT3.K21','SALT2.JLA-B14']:
            if dofit:
                prefix = model.split('.')[-1]
                for massmodel in ['HighMass','LowMass']:
                    foutsingle = open(f'fitres_singlefilt/{filt}/singlefilt_results_{model.split("/")[-1]}_{massmodel}.FITRES','w')
                    print('# CID PKMJD PKMJDERR mB mBERR zCMB zHD x0 x0ERR x1 x1ERR c cERR HOST_LOGMASS FITPROB COV_x1_c COV_x1_x0 COV_c_x0',
                          file=foutsingle)

                    for version,kcor,filtlist,outfile in zip(all_versions,all_kcors,all_filtlists,all_versions):

                        # first fit everything
                        nmltext = _nmltmpl_all.replace('<data_version>',version).\
                            replace('<kcor>',kcor).\
                            replace('<filtlist>',filtdict_version[version][filt]).\
                            replace('<outfile>','tmp_full')
                        fr = txtobj('tmp_full.FITRES.TEXT',fitresheader=True)
                        
                        #fr = txtobj(f'fitres_NoHost/{version}_SALT3.NoHost{massmodel}',fitresheader=True)
                        for i,cid in enumerate(fr.CID):
                            if filtdict_version[version][filt] is None: continue
                            nmltext = _nmltmpl.replace('<data_version>',version).\
                                      replace('<kcor>',kcor).\
                                      replace('<filtlist>',filtdict_version[version][filt]).\
                                      replace('<outfile>','tmp').\
                                      replace('<cidlist>',f'"{cid}"').\
                                      replace('<inival_peakmjd>',str(fr.PKMJD[i])).\
                                      replace('<inival_color>',str(fr.c[i])).\
                                      replace('<inival_shape>',str(fr.x1[i]))

                            if '/' in model: nmltext = nmltext.replace('<model>',os.getcwd()+'/'+model+massmodel)
                            else: nmltext = nmltext.replace('<model>',model)
                            with open('tmp2.nml','w') as fout:
                                print(nmltext,file=fout)

                            os.system(f'snlc_fit.exe tmp2.nml')
                            frtmp = txtobj('tmp.FITRES.TEXT',fitresheader=True)
                            if 'mB' not in frtmp.__dict__.keys():
                                continue
                            iCID = np.where(cid == spi.SNID)[0]
                            if not len(iCID): continue
                            if spi.xhost[iCID] > 0: hostmass = 15
                            else: hostmass = 5
                            print(f'{fr.CID[i]} {fr.PKMJD[i]} {fr.PKMJDERR[i]} {frtmp.mB[0]} {frtmp.mBERR[0]} {fr.zCMB[i]} {fr.zHD[i]} {frtmp.x0[0]} {frtmp.x0ERR[0]} {fr.x1[i]} {fr.x1ERR[i]} {fr.c[i]} {fr.cERR[i]} {hostmass} {fr.FITPROB[i]} {fr.COV_x1_c[i]} {fr.COV_x1_x0[i]} {fr.COV_c_x0[i]}',file=foutsingle)
                                
            foutsingle.close()
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

        
if __name__ == "__main__":
    hr = masshubbleresids()
    hr.main()
    #hr.edit_model()
    #hr.edit_model_list()
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    hr.main()
    #colorscat()
