#!/usr/bin/env python
# D. Jones - 1/16/20

from saltshaker.util import snana
import numpy as np
import os
import glob
from txtobj import txtobj

filtdict = {'u_CSP':'    a',
            'g_CSP':'    b',
            'r_CSP':'    c',
            'i_CSP':'    d',
            'B_CSP':'    e',
            'V_CSP_3014':'         f',
            'V_CSP_3009':'         g',
            'V_CSP':'    h',
            'Y_RC':'    i',
            'J_RC1':'    j',
            'J_RC2':'    k',
            'H_RC':'    l',
            'Y_WIRC':'     m',
            'J_WIRC':'     n',
            'H_WIRC':'     o',
            'K_WIRC':'     p',
            'u_prime':'      q',
            'r_prime':'      r',
            'i_prime':'      s',
            'J':'t',
            'H':'u',
            'K':'v',
            'V':'w',
            'I':'x',
            'U':'y',
            'R':'z',
            'B':'A',
            'Y_AND':'    B',
            'J_AND':'    C',
            'H_AND':'    D',
            'K_AND':'    E',
            'J_HST':'    F',
            'H_HST':'    G',
            'Y_P':'  Y',
            'J_P':'  J',
            'H_P':'  H',
            'K_P':'  K'}

from astropy.coordinates import SkyCoord
import astropy.units as u

def vnew(ra, dec, z):
    c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame = 'icrs')
    c_icrs = c_icrs.galactic
    b = c_icrs.b.degree
    l = c_icrs.l.degree

    b = np.radians(b)
    l = np.radians(l)
    l_0 = np.radians(264.14)
    b_0 = np.radians(48.26)


    v = (float(z)*3*10**5 + 371 * (np.sin(b) * np.sin(b_0) + np.cos(b) * np.cos(b_0) * np.cos(l-l_0)))/(3*10**5)

    return v

def main():


    # find BayeSN files
    snfiles_bayesn = glob.glob('NIR_training_data_Gautham/*')
    snidlist,snfileslist,sncomblist = np.array([]),np.array([]),np.array([])
    for snf in snfiles_bayesn:
        try: sn = snana.SuperNova(snf)
        except: continue

        # combine files with same SNID
        if sn.SNID in sncomblist:
            os.system(f"cp NIR_training_data/{sn.SNID}_combined.txt NIR_training_data/{sn.SNID}_combined_bkp.txt")
            snfile1 = f"NIR_training_data/{sn.SNID}_combined_bkp.txt"
            with open(f"NIR_training_data/{sn.SNID}_combined.txt",'w') as fout,\
                 open(snfile1,'r') as fin, open(snf) as fin2:
                for line in fin:
                    if line.startswith("OBS:"):
                        #filt = line.split()[2]
                        #line = line.replace(f" {filt} ",f" {filtdict[filt]} ")
                        print(line.replace('\n',''),file=fout)
                    elif line.startswith('FILTERS:'):
                        print(f'FILTERS: abcdefghijklmnopqrstuvwxyzABCDEFGYJHK',file=fout)
                    elif not line.startswith("END:"):
                        print(line.replace('\n',''),file=fout)
                for line in fin2:
                    if line.startswith("OBS:"):
                        filt = line.split()[2]
                        line = line.replace(f" {filt} ",f" {filtdict[filt]} ")
                        print(line.replace('\n',''),file=fout)
                print("END:",file=fout)
            os.system(f"rm NIR_training_data/{sn.SNID}_combined_bkp.txt")
            
        elif sn.SNID in snidlist:
            snfile1 = snfileslist[sn.SNID == snidlist][0]
            with open(f"NIR_training_data/{sn.SNID}_combined.txt",'w') as fout,\
                 open(snfile1,'r') as fin, open(snf) as fin2:
                for line in fin:
                    if line.startswith("OBS:"):
                        filt = line.split()[2]
                        line = line.replace(f" {filt} ",f" {filtdict[filt]} ")
                        print(line.replace('\n',''),file=fout)
                    elif line.startswith('FILTERS:'):
                        print(f'FILTERS: abcdefghijklmnopqrstuvwxyzABCDEFGYJHK',file=fout)
                    elif not line.startswith("END:"):
                        print(line.replace('\n',''),file=fout)
                for line in fin2:
                    if line.startswith("OBS:"):
                        filt = line.split()[2]
                        line = line.replace(f" {filt} ",f" {filtdict[filt]} ")
                        print(line.replace('\n',''),file=fout)
                print("END:",file=fout)
            os.system(f"rm NIR_training_data/{snfile1.split('/')[-1]}")
            sncomblist = np.append(sncomblist,sn.SNID)
        else:
            with open(snf,'r') as fin, open(f"NIR_training_data/{snf.split('/')[-1]}",'w') as fout:
                for line in fin:
                    if line.startswith("OBS:"):
                        filt = line.split()[2]
                        line = line.replace(f" {filt} ",f" {filtdict[filt]} ")
                        print(line.replace('\n',''),file=fout)
                    elif line.startswith('FILTERS:'):
                        print(f'FILTERS: abcdefghijklmnopqrstuvwxyzABCDEFGYJHK',file=fout)
                    else:
                        print(line.replace('\n',''),file=fout)


        snidlist = np.append(snidlist,sn.SNID)
        snfileslist = np.append(snfileslist,snf)

    # copy over spectra
    snfiles = glob.glob('../SALT3_training_v1.0/SALT3_training_data/*')
    for snf in snfiles:
        try: sn = snana.SuperNova(snf)
        except: continue


        if str(sn.SNID) in snidlist or sn.SNID in snidlist or \
           (not isinstance(sn.SNID,int) and not isinstance(sn.SNID,float) and sn.SNID.replace('sn','') in snidlist):
            if len(sn.SPECTRA.keys()):
                origfile = snfileslist[(sn.SNID == snidlist) | (sn.SNID.replace('sn','') == snidlist)][0]

                with open(origfile,'r') as fin1, open(snf,'r') as fin2, open(f"NIR_training_data/{sn.SNID}_spectra.txt",'w') as fout:
                    for line in fin1:
                        if line.startswith("OBS:"):
                            filt = line.split()[2]
                            line = line.replace(f" {filt} ",f" {filtdict[filt]} ")
                            print(line.replace('\n',''),file=fout)
                        elif line.startswith('FILTERS:'):
                            print(f'FILTERS: abcdefghijklmnopqrstuvwxyzABCDEFGYJHK',file=fout)
                        elif not line.startswith('END:'):
                            print(line.replace('\n',''),file=fout)

                    start = False
                    for line in fin2:
                        if line.startswith('NSPECTRA:'): start = True
                        if start:
                            print(line.replace('\n',''),file=fout)
                os.system(f"rm {origfile.replace('NIR_training_data_Gautham','NIR_training_data')}")
                os.system(f"rm NIR_training_data/{sn.SNID}_combined.txt")
                
    # copy over SALT3 files that aren't included yet
    snfiles = glob.glob('../SALT3_training_v1.0/SALT3_training_data/*')
    for snf in snfiles:
        try: sn = snana.SuperNova(snf)
        except: continue
        if 'spec' not in snf and 'Foundation' in snf and os.path.exists(snf.replace('Foundation_DR1','Foundation_DR1_spec')): continue
        
        try:
            if sn.SNID not in snidlist and sn.SNID.replace('sn','') not in snidlist:
                os.system(f'cp {snf} NIR_training_data/')
        except:
            if sn.SNID not in snidlist:
                os.system(f'cp {snf} NIR_training_data/')

        #if sn.SNID in snidlist or sn.SNID.replace('sn','') in snidlist and len(sn.SPECTRA.keys()):
        #    newfile = snfileslist
                
        #import pdb; pdb.set_trace()

def initlists(dofit=False,cpdata=False):

    surveys = ['Hamuy1996_LC','Riess1999_LC','Jha2006_LC',
               'Hicken2009_LC','OTHER_LOWZ_LC','SDSS',
               'SNLS3_LC','FOUNDATION','PS1MD',
               'DES','CFA4p1','CFA4p2','LOWZ']
    if cpdata:
        # make the data directories
        for s in surveys:
            os.system(f'mkdir lcfit_data/{s}')

        with open('NIR_training_data/NIR_training_data.LIST') as fin:
            for line in fin:
                line = line.replace('\n','')
                sn = snana.SuperNova(f'NIR_training_data/{line}')
                os.system(f'cp NIR_training_data/{line} lcfit_data/{sn.SURVEY}/')
    if dofit:
        # make the nml files using the config as a basis
        filtlist = ['UBVRIfhjkl','UBVRIfhjkl',
                    'UBVRIfhjkl','UBVRIfhjklabcde',
                    'UBVRIfhjkl','tvywxABCfhjklugriz',
                    'griz','griz','griz','griz','BVri','BVri',
                    'abcdefghqrswxyzA']
        kcorlist = ['kcor_Other_LOWZ.fits','kcor_Other_LOWZ.fits',
                    'kcor_Other_LOWZ.fits','kcor_Hicken09.fits',
                    'kcor_Other_LOWZ.fits','kcor_SDSS_Swope_Keplercam.fits',
                    'kcor_SNLS.fits','kcor_PS1_none.fits',
                    'kcor_PS1_offsets.fits','kcor_DECam.fits',
                    'kcor_CFA41.fits','kcor_CFA42.fits',
                    'kcor_lowz_BayeSN.fits']

        for s,f,k in zip(surveys,filtlist,kcorlist):
            nmltext = f"""
      &SNLCINP

         PRIVATE_DATA_PATH = 'lcfit_data/'
         VERSION_PHOTOMETRY = '{s}'
         KCOR_FILE		   = 'kcor/{k}'

         NFIT_ITERATION = 3
         INTERP_OPT		= 1

         SNTABLE_LIST = 'FITRES(text:key)'
         TEXTFILE_PREFIX  = 'lcfit_data/{s}'

         LDMP_SNFAIL = T
         USE_MWCOR = F

         H0_REF	  = 70.0
         OLAM_REF =	 0.70
         OMAT_REF =	 0.30
         W0_REF	  = -1.00

         SNCID_LIST	   =  0
         CUTWIN_CID	   =  0, 20000000
         SNCCID_LIST   =  ''
         SNCCID_IGNORE =  

         cutwin_redshift   = 0.001, 2.0
         cutwin_Nepoch	  =	 1

         RV_MWCOLORLAW = 3.1
         OPT_MWCOLORLAW = 99
         OPT_MWEBV = 3

         MAGOBS_SHIFT_PRIMARY = ' '
         EPCUT_SNRMIN = ''
         ABORT_ON_NOEPOCHS = F

      &END



      &FITINP

         FITMODEL_NAME	= 'SALT2.JLA-B14'

         PRIOR_MJDSIG		 = 5.0
         PRIOR_LUMIPAR_RANGE = -5.0, 5.0
         PRIOR_LUMIPAR_SIGMA = 0.1

         OPT_COVAR = 1
         OPT_XTMW_ERR = 1

         TREST_REJECT  = -15.0, 45.0
         NGRID_PDF	   = 0

         FUDGEALL_ITER1_MAXFRAC = 0.02
         FILTLIST_FIT = '{f}'

      &END

    """
            with open(f'nml/{s}.nml','w') as fout:
                print(nmltext,file=fout)

        # run all the fitting
        for nmlfile in glob.glob('nml/*nml'):
            os.system(f'snlc_fit.exe {nmlfile}')
    
    # grab the params from the output FITRES file
    s3pars = txtobj('../SALT3_training_v1.0/SALT3_PARS_INIT.LIST')
    pksnid,pkmjd = np.loadtxt('../SALT3_training_v1.0/SALT3_PKMJD_INIT.LIST',unpack=True,dtype=str)
    for i in range(len(pksnid)):
        if pksnid[i].startswith('sn19') or pksnid[i].startswith('sn20'):
            pksnid[i] = pksnid[i][2:]
    for i in range(len(s3pars.SNID)):
        if s3pars.SNID[i].startswith('sn19') or s3pars.SNID[i].startswith('sn20'):
            s3pars.SNID[i] = s3pars.SNID[i][2:]
    
    frfiles = glob.glob('lcfit_data/*FITRES.TEXT')
    
    with open('SALT3_NIR_PKMJD_INIT.LIST','w') as fout1, open('SALT3_NIR_PARS_INIT.LIST','w') as fout2:
        print('# SNID zHelio x0 x1 c FITPROB',file=fout2)

        for pks,pkm in zip(pksnid,pkmjd):
            print(f'{pks} {pkm}',file=fout1)
                
        for i,s in enumerate(s3pars.SNID):
            print(f'{s3pars.SNID[i]} {s3pars.zHelio[i]} {s3pars.x0[i]} {s3pars.x1[i]} {s3pars.c[i]} {s3pars.FITPROB[i]}',
                  file=fout2)
                
                
        for frf in frfiles:
            fr = txtobj(frf,fitresheader=True)
            for j,i in enumerate(fr.CID):
                #if '05eq' in fr.CID[j]: import pdb; pdb.set_trace()
                if str(i) not in pksnid and 'sn'+str(i) not in pksnid and not str(i).startswith('sn'):
                    print(f'{i} {fr.PKMJD[j]:.3f}',file=fout1)
                    print(f'{i} {fr.zHEL[j]:.6f} {fr.x0[j]:.8f} {fr.x1[j]:.3f} {fr.c[j]:.3f} {fr.FITPROB[j]}',file=fout2)
                elif str(i).startswith('sn') and str(i)[2:] not in pksnid:
                    print(f'{i} {fr.PKMJD[j]:.3f}',file=fout1)
                    print(f'{i} {fr.zHEL[j]:.6f} {fr.x0[j]:.8f} {fr.x1[j]:.3f} {fr.c[j]:.3f} {fr.FITPROB[j]}',file=fout2)                    
                    
def change_vpec():

    files = np.loadtxt('/Users/David/Dropbox/research/SALT3/examples/SALT3_NIR/lcfit_data/LOWZ/LOWZ.LIST',
                       unpack=True,dtype=str)
    files = [f"/Users/David/Dropbox/research/SALT3/examples/SALT3_NIR/lcfit_data/LOWZ/{f}" for f in files]
    for f in files:
        sn = snana.SuperNova(f)
        #import pdb; pdb.set_trace()
        zfinal = vnew(float(sn.RA.split()[0]),float(sn.DECL.split()[0]),float(sn.REDSHIFT_HELIO.split()[0]))

        os.system(f'cp {f} {f}_bkp.txt')
        with open(f'{f}_bkp.txt','r') as fin, open(f,'w') as fout:
            for line in fin:
                line = line.replace('\n','')
                if line.startswith('REDSHIFT_FINAL:'):
                    print(f'REDSHIFT_FINAL: {zfinal:.6f}',file=fout)
                else:
                    print(line,file=fout)
        os.system(f'rm {f}_bkp.txt')

def add_peakmjd():
    files = np.loadtxt('NIR_training_data/NIR_training_data.LIST',unpack=True,dtype=str)
    snid,peakmjd = np.loadtxt('SALT3_NIR_PKMJD_INIT.LIST',unpack=True,dtype=str)
    peakmjd = peakmjd.astype(float)
    
    for f in files:
        f = f"NIR_training_data/{f}"
        sn = snana.SuperNova(f)
        if sn.SNID not in snid: continue
        pkmjd_forsnid = peakmjd[sn.SNID == snid][0]
        
        with open(f) as fin,open(f+'_peakmjd.dat','w') as fout:
            for line in fin:
                if line.startswith('MWEBV:'):
                    print(line.replace('\n',''),file=fout)
                    print(f'PEAKMJD: {pkmjd_forsnid:.2f}',file=fout)
                else:
                    print(line.replace('\n',''),file=fout)
    return
    
if __name__ == "__main__":
    #main()
    #initlists()
    #change_vpec()
    add_peakmjd()
