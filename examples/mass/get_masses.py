#!/usr/bin/env python
# D. Jones - 5/13/21

# make sure we have good host masses for everything
# 1. get host coords for missing SNe in training (low-z for now) - done
# 2. check all host coords visually (low-z for now)
# 3. get aperture photometry, full low-z sample (fix WISE)
# 4. check host SExtractor ellipse params
# 5. record which ones have been visually inspected
# 6. run the LePHARE fits
# 7. look at the LePHARE output
# 8. 

import os
import numpy as np
import glob
import snana
from txtobj import txtobj
from coordutils import GetSexigesimalString

def get_host_coords():

    hm = txtobj('/Users/David/Dropbox/research/H0/hosts/GPC1v3_hosts.txt')
    
    # get all the low-z SNe, figure out which ones don't have masses yet
    files = glob.glob('../SALT3TRAIN_K21_PUBLIC/SALT3TRAIN_K21/*/*')
    for f in files:
        if f.endswith('~') or f.endswith('LIST') or f.endswith('README'): continue
        #print(f)
        sn = snana.SuperNova(f)
        if float(sn.REDSHIFT_HELIO.split()[0]) > 0.1: continue
        # first just figure out which ones need RA/dec
        if isinstance(sn.SNID,float): sn.SNID = str(int(sn.SNID))
        if isinstance(sn.SNID,int): sn.SNID = str(int(sn.SNID))
        
        if sn.SNID not in hm.ID and sn.SNID.upper() not in hm.ID:
            if 'DEC' not in sn.__dict__.keys():
                sn.DEC = sn.DECL
            
            try:
                sn.DEC = float(sn.DEC)
            except:
                sn.DEC = float(sn.DEC.split()[0])

            try:
                sn.RA = float(sn.RA)
            except:
                sn.RA = float(sn.RA.split()[0])

            ra,dec = GetSexigesimalString(sn.RA,sn.DEC)
            print(f'YSEmasterlist.py -n {sn.SNID} {ra} {dec} --type SNIa -z {sn.REDSHIFT_HELIO.split()[0]}')
            print(f'getHostPhot.py -p -e {sn.SNID}')

def check_lowz_hostpos():

    hm = txtobj('/Users/David/Dropbox/research/H0/hosts/GPC1v3_hosts.txt')
    
    # get all the low-z SNe, figure out which ones don't have masses yet
    files = glob.glob('../SALT3TRAIN_K21_PUBLIC/SALT3TRAIN_K21/*/*')
    snidlist = []
    for f in files:
        if f.endswith('~') or f.endswith('LIST') or f.endswith('README'): continue

        sn = snana.SuperNova(f)
        if float(sn.REDSHIFT_HELIO.split()[0]) > 0.1: continue
        # first just figure out which ones need RA/dec
        if isinstance(sn.SNID,float): sn.SNID = str(int(sn.SNID))
        if isinstance(sn.SNID,int): sn.SNID = str(int(sn.SNID))
        
        if sn.SNID not in hm.ID and sn.SNID.upper() in hm.ID:
            snidlist += [sn.SNID.upper()]
        else:
            snidlist += [sn.SNID]
            
    print(f"getHostPhot.py --plothostpos --hypheninname -e {','.join(snidlist)}")

def getAperPhot():

    hm = txtobj('/Users/David/Dropbox/research/H0/hosts/GPC1v3_hosts.txt')
    
    # get all the low-z SNe, figure out which ones don't have masses yet
    files = glob.glob('../SALT3TRAIN_K21_PUBLIC/SALT3TRAIN_K21/*/*')
    snidlist = []
    for f in files:
        if f.endswith('~') or f.endswith('LIST') or f.endswith('README'): continue

        sn = snana.SuperNova(f)
        if float(sn.REDSHIFT_HELIO.split()[0]) > 0.1: continue
        # first just figure out which ones need RA/dec
        if isinstance(sn.SNID,float): sn.SNID = str(int(sn.SNID))
        if isinstance(sn.SNID,int): sn.SNID = str(int(sn.SNID))
        
        if sn.SNID not in hm.ID and sn.SNID.upper() in hm.ID:
            snidlist += [sn.SNID.upper()]
            print(f"getHostPhot.py  --hypheninname -e {sn.SNID.upper()} -a --surveys GALEX,SDSS,PS1,2MASS --clobber")
        else:
            print(f"getHostPhot.py  --hypheninname -e {sn.SNID} -a --surveys GALEX,SDSS,PS1,2MASS --clobber")
            
def add_mass_to_lc():

    data = at.Table.read('hostpars_salt3_lowztraining.txt',format='ascii')
    files = glob.glob('SALT3TRAIN_K21/*/*')

    for f in files:
        if not f.endswith('~') and \
           not f.endswith('LIST') and \
           not f.endswith('README'):
            sn = snana.SuperNova(f)
            iD = data['SNID'] == sn.SNID
            if not len(data[iD]):
                raise RuntimeError(f'no mass for SN {sn.SNID}')
            
            if not os.path.exists(os.path.dirname(f.replace('SALT3TRAIN_K21','SALT3TRAIN_J21'))):
                os.makedirs(f.replace('SALT3TRAIN_K21','SALT3TRAIN_J21'))

            with open(f) as fin, open(f.replace('SALT3TRAIN_K21','SALT3TRAIN_J21'),'w') as fout:
                lines = fin.readlines()

                has_mass,has_mass_err = False,False
                for line in lines:
                    if line.startswith('HOST_LOGMASS'): has_mass = True
                    if line.startswith('HOST_LOGMASS_ERR'): has_mass_err = True
                    
                for line in lines:
                    line = line.replace('\n','')
                    if has_mass and not has_mass_err and line.startswith('HOST_LOGMASS:'):
                        print(f'HOST_LOGMASS: {logmass:.3f}',file=fout)
                        print(f'HOST_LOGMASS_ERR: {logmass_err:.3f}',file=fout)                        
                    if not line.startswith('HOST_LOGMASS:'):
                        print(line,file=fout)

def make_init_lists():

    ip = txtobj('SALT3TRAIN_K21/SALT3_PARS_INIT.LIST')
    hm = txtobj('hostpars_salt3_lowztraining.txt')
    with open('SALT3_PARS_INIT_HOSTMASS.LIST','w') as fout:
        print('# SNID zHelio x0 x1 c logmass xhost FITPROB',file=fout)
        for i in range(len(ip.SNID)):
            iMass = np.where((ip.SNID[i] == hm.snid) | (ip.SNID[i].upper() == hm.snid))[0]
            if len(iMass):
                logmass = hm.logmass[iMass][0]
                logmass_err_low = hm.logmass[iMass][0]-hm.logmass_low[iMass][0]
                logmass_err_high = hm.logmass_high[iMass][0]-hm.logmass[iMass][0]
                if logmass > 10: xhost = 0.5
                elif logmass <= 10: xhost = -0.5
                print(f"{ip.SNID[i]} {ip.zHelio[i]:.6f} {ip.x0[i]:.8f} {ip.x1[i]:.4f} {ip.c[i]:.6f} {logmass:.3f} {xhost:.1f} {ip.FITPROB[i]:.6f}",file=fout)
                    
if __name__ == "__main__":
    #main()
    #check_lowz_hostpos()
    #getAperPhot()
    #add_mass_to_lc()
    make_init_lists()
