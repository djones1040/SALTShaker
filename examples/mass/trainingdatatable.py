#!/usr/bin/env python
import numpy as np
from saltshaker.util import snana
import glob
import astropy.table as at

def main():
    surveynames = ['Calan-Tololo','CfA1','CfA2','CfA3',
               'SDSS$^{\\rm b}$','SNLS','Misc. low-$z$','PS1 MDS$^{\\rm c}$',
               'DES-spec$^{\\rm d}$','Foundation']
    surveys_lc = ['Hamuy1996_LC','Riess1999_LC','Jha2006_LC','Hicken2009_LC','SDSS','SNLS3_LC',
                  'OTHER_LOWZ_LC','PS1MD','DES','FOUNDATION']
    filters = ['$BVRI$','$UBVRI$','$UBVRI$','$UBVRIri$','$ugriz$',
               '$griz$','$griz$','$UBVRI$','$griz$','$griz$','$griz$']
    citations = ['Hamuy1996','Riess1999','Jha2006','Hicken2009a','Holtzman08','Astier2006',
                 'Jha2007','Scolnic2018','Abbott2019','Foley2018']
    
    snid = np.loadtxt('/Users/David/Dropbox/research/SALT3/examples/SALT3_training_v1.0/SALT3_training_data/SALT3_training.LIST',unpack=True,dtype=str)
    speccount = 0; speccount_jla = 0; totalcount_jla = 0; z_nojla = [];
    z_tot = []; z_jla = []; surveys = []; spectra = []
    for s in snid:
        sn = snana.SuperNova(f"/Users/David/Dropbox/research/SALT3/examples/SALT3_training_v1.0/SALT3_training_data/{s}")
        if len(sn.SPECTRA.keys()): speccount += 1; spectra += [len(sn.SPECTRA.keys())]
        else: spectra += [0]
        
        if sn.SURVEY not in ['FOUNDATION','PS1MD','DES']:
            totalcount_jla += 1
            if len(sn.SPECTRA.keys()): speccount_jla += 1
            z_jla += [float(sn.REDSHIFT_HELIO.split()[0])]
        else:
            z_nojla += [sn.z]
        z_tot += [float(sn.REDSHIFT_HELIO.split()[0])]
        surveys += [sn.SURVEY]
    z_nojla = np.array(z_nojla); z_tot = np.array(z_tot); z_jla = np.array(z_jla); surveys = np.array(surveys)
    spectra = np.array(spectra)
    
    for s,sl,f,c in zip(surveynames,surveys_lc,filters,citations):
        nsn = len(surveys[surveys == sl])
        zmin = np.min(z_tot[surveys == sl])
        zmed = np.median(z_tot[surveys == sl])
        zmax = np.max(z_tot[surveys == sl])
        nspec = np.sum(spectra[surveys == sl])
        
        outline = f"{s}&{nsn:.0f}&{nspec:.0f}&{zmin:.3f}&{zmed:.3f}&{zmax:.3f}&{f}&\\citet{{%s}}\\\\"%c
        print(outline)
        
def main_newdata():
    pksnid,pkmjd = np.loadtxt('/Users/David/Dropbox/research/SALT3/examples/SALT3_training_v1.0/SALT3_PKMJD_INIT.LIST.bkp',unpack=True,dtype=str)
    pkmjd = pkmjd.astype(float)

    surveynames = ['Calan-Tololo','CfA1','CfA2','CfA3',
                   'SDSS$^{\\rm b}$','SNLS','Misc. low-$z$','CfA4','CSP','Foundation','PS1 MDS$^{\\rm c}$',
                   'DES-spec$^{\\rm d}$']
    surveys_lc = ['Hamuy1996_LC','Riess1999_LC','Jha2006_LC','Hicken2009_LC','SDSS','SNLS3_LC',
                  'OTHER_LOWZ_LC','CFA4p1','PS1_LOWZ_COMBINED(CSP)','FOUNDATION','PS1MD','DES']
    filters = ['$BVRI$','$UBVRI$','$UBVRI$','$UBVRIri$','$ugriz$',
               '$griz$','$griz$','$UBVRI$','$BVri$','$BVgri$','$griz$','$griz$','$griz$']
    citations = ['Hamuy1996','Riess1999','Jha2006','Hicken2009a','Holtzman08','Astier2006',
                 'Jha2007','Hicken2012','Krisciunas2017','Foley2018','Scolnic2018','Abbott2019']
    snidpassescuts = np.loadtxt('/Users/David/Dropbox/research/SALT3/examples/SALT3_training_v1.0/output_test/salt3train_snparams.txt',unpack=True,usecols=[0],dtype=str)
    oldsnid = np.loadtxt('/Users/David/Dropbox/research/SALT3/examples/JLA_training_orig/JLA_training_origlc/JLA_training_origlc.LIST',unpack=True,dtype=str)    
    oldsnidlist = []
    for o in oldsnid:
        with open(f"/Users/David/Dropbox/research/SALT3/examples/JLA_training_orig/JLA_training_origlc/{o}") as fin:
            for line in fin:
                if line.startswith('SNID:'):
                    oldsnidlist += [line.split()[1].replace('\n','')]
                    break
    oldsnidlist = np.array(oldsnidlist)

    snid = np.loadtxt('/Users/David/Dropbox/research/SALT3/examples/SALT3_training_v1.0/SALT3_training_data/SALT3_training.LIST',unpack=True,dtype=str)
    speccount = 0; speccount_jla = 0; totalcount_jla = 0; z_nojla = [];
    z_tot = []; z_jla = []; surveys_old = []; surveys_new = []; spectra_new = []; spectra_old = []
    for s in snid:
        snnew = snana.SuperNova(f"/Users/David/Dropbox/research/SALT3/examples/SALT3_training_v1.0/SALT3_training_data/{s}")
        if str(snnew.SNID) not in snidpassescuts:
            continue
        if str(snnew.SNID) in oldsnidlist:
            
            try: snold = snana.SuperNova(f"/Users/David/Dropbox/research/SALT3/examples/JLA_training_orig/JLA_training_origlc/{oldsnid[oldsnidlist == str(snnew.SNID)][0]}")
            except: import pdb; pdb.set_trace()
            surveys_old += [snold.SURVEY]
            if len(snold.SPECTRA.keys()):
                nspec = 0
                for k in snold.SPECTRA.keys():
                    phase = (snold.SPECTRA[k]['SPECTRUM_MJD'] - pkmjd[pksnid == str(snold.SNID)])[0]/(1+float(snold.REDSHIFT_HELIO.split()[0]))
                    if phase > -20 and phase < 47: # and snold.SPECTRA[k]['SPECTRUM_MJD'] > snold.MJD.min():
                        nspec += 1
                speccount += 1; spectra_old += [nspec]

            else: spectra_old += [0]
        if len(snnew.SPECTRA.keys()):
            nspec = 0
            for k in snnew.SPECTRA.keys():
                phase = (snnew.SPECTRA[k]['SPECTRUM_MJD'] - pkmjd[pksnid == str(snnew.SNID)])[0]/(1+float(snnew.REDSHIFT_HELIO.split()[0]))
                if phase > -20 and phase < 47: # and snnew.SPECTRA[k]['SPECTRUM_MJD'] > snnew.MJD.min():
                    nspec += 1
            speccount += 1; spectra_new += [nspec]
        else:
            spectra_new += [0]
        
        if snnew.SURVEY not in ['CFA4p1','CSP','FOUNDATION','PS1MD','DES']:
            totalcount_jla += 1
            if len(snnew.SPECTRA.keys()): speccount_jla += 1
            z_jla += [float(snnew.REDSHIFT_HELIO.split()[0])]
        else:
            z_nojla += [snnew.z]
        z_tot += [float(snnew.REDSHIFT_HELIO.split()[0])]
        surveys_new += [snnew.SURVEY.replace('CFA4p2','CFA4p1')]
    z_nojla = np.array(z_nojla); z_tot = np.array(z_tot); z_jla = np.array(z_jla); surveys_old = np.array(surveys_old); surveys_new = np.array(surveys_new)
    spectra_new = np.array(spectra_new); spectra_old = np.array(spectra_old)
    #import pdb; pdb.set_trace()
    for s,sl,f,c in zip(surveynames,surveys_lc,filters,citations):
        nsn = len(surveys_new[surveys_new == sl])
        zmin = np.min(z_tot[surveys_new == sl])
        zmed = np.median(z_tot[surveys_new == sl])
        zmax = np.max(z_tot[surveys_new == sl])
        nspec_old = np.sum(spectra_old[surveys_old == sl])
        nspec_new = np.sum(spectra_new[surveys_new == sl])
        outline = f"{s}&{nsn:.0f}&{nspec_old:.0f}&{nspec_new:.0f}&{zmin:.3f}&{zmed:.3f}&{zmax:.3f}&{f}&\\citet{{%s}}\\\\"%c
        print(outline)

        if s == 'Misc. low-$z$':
            iSurveysOld = (surveys_new == 'Hamuy1996_LC') | (surveys_new == 'Riess1999_LC') | (surveys_new == 'Jha2006_LC') | \
                          (surveys_new == 'Hicken2009_LC') | (surveys_new == 'SDSS') | (surveys_new == 'SNLS3_LC') | \
                          (surveys_new == 'OTHER_LOWZ_LC')
            nsn = len(surveys_new[iSurveysOld])
            zmin = np.min(z_tot[iSurveysOld])
            zmed = np.median(z_tot[iSurveysOld])
            zmax = np.max(z_tot[iSurveysOld])
            nspec_old = np.sum(spectra_old)
            nspec_new = np.sum(spectra_new[iSurveysOld])
            outline = f"{{\\bf SALT2.4 Total}}&{nsn:.0f}&{nspec_old:.0f}&{nspec_new:.0f}&{zmin:.3f}&{zmed:.3f}&{zmax:.3f}&\\nodata&\\nodata\\\\"
            print(outline)
            print('\\hline')
        if sl == 'DES':
            iSurveysOld = (surveys_new == 'CFA4p1') | (surveys_new == 'PS1_LOWZ_COMBINED(CSP)') | (surveys_new == 'FOUNDATION') | (surveys_new == 'PS1MD') | (surveys_new == 'DES')
            nsn = len(surveys_new[iSurveysOld])
            zmin = np.min(z_tot[iSurveysOld])
            zmed = np.median(z_tot[iSurveysOld])
            zmax = np.max(z_tot[iSurveysOld])
            nspec_old = 0 #np.sum(spectra_old[iSurveysOld])
            nspec_new = np.sum(spectra_new[iSurveysOld])
            outline = f"{{\\bf New Data Total}}&{nsn:.0f}&{nspec_old:.0f}&{nspec_new:.0f}&{zmin:.3f}&{zmed:.3f}&{zmax:.3f}&\\nodata&\\nodata\\\\"
            print(outline)
            print('\\hline\\\\[-1.5ex]')
            nsn = len(surveys_new)
            zmin = np.min(z_tot)
            zmed = np.median(z_tot)
            zmax = np.max(z_tot)
            nspec_old = np.sum(spectra_old)
            nspec_new = np.sum(spectra_new)
            outline = f"{{\\bf SALT3 Total}}&{nsn:.0f}&{nspec_old:.0f}&{nspec_new:.0f}&{zmin:.3f}&{zmed:.3f}&{zmax:.3f}&\\nodata&\\nodata\\\\"
            print(outline)

def main_hostmass():
    
    surveynames = ['Calan-Tololo','CfA1','CfA2','CfA3',
                   'SDSS','SNLS','Misc. low-$z$','CfA4','CSP','Foundation','PS1 MDS',
                   'DES-spec']
    surveys_lc = ['SALT3TRAIN_PanPlus_Hamuy1996','SALT3TRAIN_PanPlus_Riess1999','SALT3TRAIN_PanPlus_Jha2006',
                  'SALT3TRAIN_PanPlus_Hicken2009','SALT3TRAIN_PanPlus_SDSS','SALT3TRAIN_PanPlus_SNLS3',
                  'SALT3TRAIN_PanPlus_OTHER_LOWZ','SALT3TRAIN_PanPlus_CfA4p1','SALT3TRAIN_PanPlus_CSPDR3',
                  'SALT3TRAIN_PanPlus_Foundation_DR1','SALT3TRAIN_PanPlus_PS1MD','SALT3TRAIN_PanPlus_DESSN3YR']
    filters = ['$BVRI$','$UBVRI$','$UBVRI$','$UBVRIri$','$ugriz$',
               '$griz$','$griz$','$UBVRI$','$BVri$','$BVgri$','$griz$','$griz$','$griz$']
    citations = ['Hamuy1996','Riess1999','Jha2006','Hicken2009a','Holtzman08','Astier2006',
                 'Jha2007','Hicken2012','Krisciunas2017','Foley2018','Scolnic2018','Abbott2019']
    snidpassescuts = np.loadtxt('/Users/David/Dropbox/research/SALTShaker/examples/mass/output/salt3train_snparams.txt',unpack=True,usecols=[0],dtype=str)
    params = at.Table.read('SALT3_PARS_INIT_HOSTMASS.LIST',format='ascii')

    n_highmass_tot,n_lowmass_tot,n_highmass_spec_tot,n_lowmass_spec_tot = 0,0,0,0
    for snn,f,s,c in zip(surveynames,filters,surveys_lc,citations):
        snfiles = glob.glob(f"/Users/David/Dropbox/research/SALTShaker/examples/mass/SALT3TRAIN_PanPlus/{s}/*.gz")
        if not len(snfiles): continue
        n_highmass,n_lowmass,n_highmass_spec,n_lowmass_spec = 0,0,0,0
        for snf in snfiles:
            sn = snana.SuperNova(snf)
            if sn.SNID not in params['SNID']:
                continue
            if sn.SNID not in snidpassescuts:
                continue
            elif params['xhost'][params['SNID'] == sn.SNID] > 0:
                n_highmass += 1
                n_highmass_spec += len(sn.SPECTRA.keys())
            else:
                n_lowmass += 1
                n_lowmass_spec += len(sn.SPECTRA.keys())
        print(f"{snn}&{n_lowmass}&{n_lowmass_spec}&&{n_highmass}&{n_highmass_spec}&{f}&\\citet{{%s}}\\\\"%c)
        n_highmass_tot += n_highmass
        n_lowmass_tot += n_lowmass
        n_highmass_spec_tot += n_highmass_spec
        n_lowmass_spec_tot += n_lowmass_spec
    print("\\hline\\\\*[-1.5ex]")
    print(f"Total&{n_lowmass_tot}&{n_lowmass_spec_tot}&&{n_highmass_tot}&{n_highmass_spec_tot}&\\nodata&\\nodata\\\\")
        
if __name__ == "__main__":
    #main()
    #main_newdata()
    main_hostmass()
