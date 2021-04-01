#!/usr/bin/env python
# D. Jones - 7/31/20

import glob
import numpy as np
from saltshaker.util import snana
import pylab as plt
plt.ion()

def main():
    
    lcfiles = glob.glob('SALT3_training_data/Foundation_DR1_*txt')
    specfiles = glob.glob('../../../SALT3_runpipe/Pantheon_Foundation_DES/Pantheon_Found_DES/found*.txt')
    for l in lcfiles:
        if 'spec' in l: continue
        print(l)
        sn = snana.SuperNova(l)
        specfilematches = glob.glob(f'../../../SALT3_runpipe/Pantheon_Foundation_DES/Pantheon_Found_DES/foun*{sn.SNID}*txt')
        if not len(specfilematches):
            print('no match found!')
            import pdb; pdb.set_trace()
            continue
        else:
            specfile = specfilematches[0]
            sns = snana.SuperNova(specfile)
            for k in sns.SPECTRA.keys():
                plt.clf()
                plt.plot((sns.SPECTRA[k]['LAMMIN']+sns.SPECTRA[k]['LAMMAX'])/2.,sns.SPECTRA[k]['FLAM'])
                YorN = input('spectra ok?')
                if YorN == 'y':
                    print(f"writing spectra for SN {sn.SNID} to file {l.replace('Foundation_DR1','Foundation_DR1_spec')}")

                    with open(l.replace('Foundation_DR1','Foundation_DR1_spec'),'w') as fout, open(l) as fin, open(specfile) as finspec:
                        for line in fin:
                            if not line.startswith('END:'):
                                print(line.replace('\n',''),file=fout)
                            else:
                                print('END_PHOTOMETRY:',file=fout)
                                print('',file=fout)
                                print('# =============================================',file=fout)
                                
                                
                        start = False
                        for line in finspec:
                            if line.startswith('NSPECTRA'):
                                start = True
                            if start:
                                print(line.replace('\n',''),file=fout)

def mainfromlist():

    import os
    
    lcfiles = glob.glob('SALT3_training_data/Foundation_DR1_*txt')
    specfiles = glob.glob('../../../SALT3_runpipe/Pantheon_Foundation_DES/Pantheon_Found_DES/found*.txt')
    specfilesbad,maskingregions = np.loadtxt('foundspecmasking.list',dtype=str,unpack=True)
    for l in lcfiles:
        if 'spec' in l: continue
        print(l)
        if not os.path.exists(l.replace('Foundation_DR1','Foundation_DR1_spec')):
            os.system(f"cp {l} {l.replace('Foundation_DR1','Foundation_DR1_spec')}")

        if l not in specfilesbad: continue
        
        mask = maskingregions[l == specfilesbad][0].split(',')
        
        sn = snana.SuperNova(l)
        specfilematches = glob.glob(f'../../../SALT3_runpipe/Pantheon_Foundation_DES/Pantheon_Found_DES/foun*{sn.SNID}*txt')

        specfile = specfilematches[0]
        sns = snana.SuperNova(specfile)
        for k in sns.SPECTRA.keys():
            print(f"writing spectra for SN {sn.SNID} to file {l.replace('Foundation_DR1','Foundation_DR1_spec')}")

            with open(l.replace('Foundation_DR1','Foundation_DR1_spec'),'w') as fout, open(l) as fin, open(specfile) as finspec:
                for line in fin:
                    if not line.startswith('END:'):
                        print(line.replace('\n',''),file=fout)
                    else:
                        print('END_PHOTOMETRY:',file=fout)
                        print('',file=fout)
                        print('# =============================================',file=fout)
                                
                                
                start = False
                for line in finspec:
                    if line.startswith('NSPECTRA'):
                        start = True
                    if start and line.startswith('SPEC:'):
                        lineparts = line.split()
                        lammin = float(lineparts[1])
                        if len(mask) == 2:
                            if lammin > int(mask[0]) and lammin < int(mask[1]):
                                lineparts[-1] = '0'
                        elif len(mask) == 4:
                            if lammin > int(mask[0]) and lammin < int(mask[1]):
                                lineparts[-1] = '0'
                            elif lammin > int(mask[2]) and lammin < int(mask[3]):
                                lineparts[-1] = '0'
                        if lammin > 7550 and lammin < 7700:
                            lineparts[-1] = '0'
                        print(' '.join(lineparts),file=fout)
                    elif start:
                        print(line.replace('\n',''),file=fout)

                                
if __name__ == "__main__":
    #main()
    mainfromlist()
