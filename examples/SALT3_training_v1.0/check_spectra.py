#!/usr/bin/env python

from saltshaker.util import snana
import numpy as np
import glob
from numpy import array,log10,unique,where,absolute
import os
import extinction

_telluric = [7550,7700]
_halpha = [6563-20,6563+20]

def append_spec_to_file(specfile,initfile,outfile,mjd,mwav,redshift=None):

    w,f,df = np.loadtxt(specfile,unpack=True)
    
    with open(outfile,'w') as fout,open(initfile) as fin:

        specid = 0
        has_spectra = False
        for line in fin:
            if line.startswith('NSPECTRA:'):
                nspec = int(line.split()[1])
                print(f'NSPECTRA: {nspec+1}',file=fout)
                has_spectra = True
            else:
                print(line.replace('\n',''),file=fout)

            if line.startswith('SPECTRUM_ID:'): specid = int(line.split()[1])
                
        if not has_spectra:
            print('# =============================================',file=fout)
            print('NSPECTRA: 1',file=fout)
            print('\nNVAR_SPEC: 5',file=fout)
            print('VARNAMES_SPEC: LAMMIN LAMMAX  FLAM  FLAMERR DQ\n',file=fout)
            


        specid += 1
        sn_spectrum=snana.SuperNovaSpectrum(specfile)

        # we have to undo the MW extinction corrections
        a_filt = extinction.fitzpatrick99(sn_spectrum.WAVE,mwav)
        sn_spectrum.FLUX *= 10**(-0.4*a_filt)
        sn_spectrum.FLUXERR *= 10**(-0.4*a_filt)
        sn_spectrum.WAVE *= (1+redshift)
        
        print('\nSPECTRUM_ID: %i'%specid,file=fout)
        print('SPECTRUM_MJD: %9.2f'%mjd,file=fout)
        print('SPECTRUM_NLAM: %9.0f'%len(sn_spectrum.WAVE),file=fout)
        resolution=sn_spectrum.WAVE[1]-sn_spectrum.WAVE[0]
        for wl,fl,flerr,dq in zip(sn_spectrum.WAVE,sn_spectrum.FLUX,sn_spectrum.FLUXERR,sn_spectrum.VALID):
            wl_l=wl-resolution/2.0
            wl_u=wl+resolution/2.0
            if wl > _telluric[0] and wl < _telluric[1]: dq = 0
            if redshift is not None and wl/(1+redshift) > _halpha[0] and wl/(1+redshift) < _halpha[1]: dq = 0
            if fl == fl and flerr == flerr: print('SPEC: %9.2f %9.2f %9.5e %9.5e %i'%(wl_l,wl_u,fl,flerr,dq),file=fout)
        print('SPECTRUM_END:\n',file=fout)  


        return( None )
        

def main():

    # need to mask telluric regions and occasionally Halpha
    # conservatively remove the last ~200 angstroms from every spectrum?
    badspec = ['sn2001ba-20010430-ui-kaepora.flm','sn2002de-20020614.19-fast-kaepora.flm',
               'sn2003fa-20030731.21-fast-kaepora.flm','sn2005eu-20051006.584-deimos-kaepora.flm',
               'sn2005eu-20051007.38-fast-kaepora.flm','sn2005eu-20051009.38-fast-kaepora.flm',
               'sn2005eu-20051011.350-ui-corrected-kaepora.flm',
               'sn2006ac-20060304.54-fast-kaepora.flm',
               'sn2006d-20060222.513-ui-kaepora.flm',
               'sn2006lf-20061220.150-ui-corrected-kaepora.flm',
               'sn2006lf-20061221.21-fast-kaepora.flm',
               'sn2008bf-20080509-ui-kaepora.flm','sn2006nz-20061124-ntt-kaepora.flm','sn2006nz-20061213-ntt-kaepora.flm']

    
    snidlist,pkmjdlist = np.loadtxt('SALT3_PKMJD_INIT.LIST',unpack=True,dtype=str)
    pkmjdlist = pkmjdlist.astype(float)
    
    snfiles = np.loadtxt('SALT3_training_data_withCSP/SALT3_training.LIST',dtype=str)
    for s in snfiles:
        snfile = f"SALT3_training_data/{s}"
        sn = snana.SuperNova(snfile)
        try:
            pkmjd = pkmjdlist[sn.SNID == snidlist][0]
        except:
            # looks like these are just SDSS and PS1, which is fine
            continue
        
        specfiles = np.concatenate(
            (glob.glob(f'/Users/David/Downloads/david_spectra_list*/*{sn.SNID}-*flm'),
             glob.glob(f"/Users/David/Downloads/david_spectra_list*/*{sn.SNID.replace('sn19','')}-*flm"),
             glob.glob(f"/Users/David/Downloads/david_spectra_list*/*{sn.SNID.replace('sn19','19')}-*flm"),
             glob.glob(f"/Users/David/Downloads/david_spectra_list*/*{sn.SNID.replace('sn20','')}-*flm"),
             glob.glob(f"/Users/David/Downloads/david_spectra_list*/*{sn.SNID.replace('sn20','20')}-*flm")))
        specfiles = np.unique(specfiles)
        
        # try a couple options if we can't find any
        if not len(specfiles):
        #    print(f'no specfiles for {sn.SNID}')
            continue
        for specf in specfiles:

            if specf.split('/')[-1] in badspec: continue
            
            w,f,df = np.loadtxt(specf,unpack=True)
            if not len(w[w > 3500]): continue
            
            # do we have this spectrum already?
            with open(specf) as fin:
                for line in fin:
                    if line.startswith('# MJD'):
                        kmjd = float(line.split()[2])
                    elif line.startswith('# AV_MW'):
                        kav = float(line.split()[2])
                    else:
                        continue

            if (kmjd-pkmjd)/(1+float(sn.REDSHIFT_HELIO.split()[0])) < -20 or (kmjd-pkmjd)/(1+float(sn.REDSHIFT_HELIO.split()[0])) > 50:
                continue
                    
            have_spec = False
            for k in sn.SPECTRA.keys():
                mjd = sn.SPECTRA[k]['SPECTRUM_MJD']
                if np.abs(mjd-kmjd) < 0.5:
                    have_spec = True
            if not have_spec:
                # now write the new spectrum to file
                append_spec_to_file(specf,snfile,f'SALT3_training_data/{sn.SNID}_newspec.dat',
                                    kmjd,kav,redshift=float(sn.REDSHIFT_HELIO.split()[0]))

                snfile = f'SALT3_training_data/{sn.SNID}_newspec.dat'
                os.system(f"cp {snfile} {snfile+'.bkp'}")
                snfile = f'SALT3_training_data/{sn.SNID}_newspec.dat.bkp'
                
                sn = snana.SuperNova(f'SALT3_training_data/{sn.SNID}_newspec.dat')
                print(f'{sn.SNID} {specf}')
                #import pdb; pdb.set_trace()
    return

def edit_listfile():

    snfiles = np.loadtxt('SALT3_training_data/SALT3_training.LIST',dtype=str)

    outfiles = []
    for s in snfiles:
        snfile = f"SALT3_training_data/{s}"
        sn = snana.SuperNova(snfile)

        newfile = glob.glob(f"SALT3_training_data/{sn.SNID}_newspec.dat")
        if len(newfile):
            outfiles += [newfile[0].split('/')[-1]]
        else:
            outfiles += [s]

    with open('SALT3_training_data/SALT3_training_test.LIST','w') as fout:
        for o in outfiles:
            print(o,file=fout)

def view_spec():

    from matplotlib.backends.backend_pdf import PdfPages
    import pylab as plt
    
    specfiles = np.loadtxt('speclist.txt',unpack=True,dtype=str)
    
    with PdfPages(f'kaepora_spec.pdf') as pdf:
        for s in specfiles:

            wave,flux,df = np.loadtxt(s,unpack=True)
            f = plt.figure()
            plt.plot(wave,flux)
            plt.title(s.split('/')[-1])
        
            pdf.savefig(f)

def imprecise_mjds():

    snid,pkmjd = np.loadtxt('SALT3_PKMJD_INIT.LIST.bkp',unpack=True,dtype=str)
    pkmjd = pkmjd.astype(float)
    
    snfiles = np.loadtxt('SALT3_training_data/SALT3_training.LIST',dtype=str)
    for snf in snfiles:
        if 'newspec' not in snf: continue
        sn = snana.SuperNova(f"SALT3_training_data/{snf}")
        for k in sn.SPECTRA.keys():
            if sn.SPECTRA[k]['SPECTRUM_MJD'] == int(sn.SPECTRA[k]['SPECTRUM_MJD']):
                phase = (sn.SPECTRA[k]['SPECTRUM_MJD'] - pkmjd[snid == sn.SNID])[0]/(1+float(sn.REDSHIFT_HELIO.split()[0]))
                print(sn.SNID,k,phase)
                
if __name__ == "__main__":
    #main()
    #edit_listfile()
    #view_spec()
    imprecise_mjds()
