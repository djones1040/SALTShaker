# combine photometry and spectra from
# multiple photometric systems
from saltshaker.util import snana
import numpy as np

def combine_lcs(snfile1,snfile2,outfile):

    with open(outfile,'w') as fout:

        # get the SuperNova objects to make
        # life easier
        sn1 = snana.SuperNova(snfile1)
        sn2 = snana.SuperNova(snfile2)
        
        # first, do the photometry
        for write in [False,True]:
            nobs = 0
            with open(snfile1) as fin:
                for line in fin:
                    if line.startswith('END'): break

                    if write:
                        if line.startswith('NOBS:'):
                            print(f'NOBS: {nobs:.0f}',file=fout)
                        else:
                            print(line.replace('\n',''),file=fout)
                    elif line.startswith('OBS:'):
                        nobs += 1
                    
            with open(snfile2) as fin:
                for line in fin:
                    if line.startswith('OBS:'):
                        if not write:
                            nobs += 1
                            continue
                        # check for duplicate observations
                        MJD = float(line.split()[1])
                        filt = line.split()[2]
                        iMatch = np.where(
                            (sn1.FLT == filt) &
                            (np.abs(sn1.MJD-MJD) < 0.01))[0]
                        if not len(iMatch):
                            print(line.replace('\n',''),file=fout)
                        else:
                            print('warning : duplicate obs detected, be careful!')
                
        print('END_PHOTOMETRY:',file=fout)

        # are there spectra?
        if len(sn1.SPECTRA.keys()):
            with open(snfile1) as fin:
                start = False
                for line in fin:
                    if line.startswith('NSPECTRA'): start = True
                    if start:
                        print(line.replace('\n',''),file=fout)
        # assume no spectra in file #2 because I'm lazy rn
        
        print('END:',file=fout)

    return
