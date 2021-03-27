#!/usr/bin/env python
# D. Jones - 3/25/21
# make new LIST files for the
# validation and training samples
# having new spectra and CSP sample

import numpy as np
import snana

def main():
    origfiles = np.loadtxt('SALT3_training_data/SALT3_training.LIST',unpack=True,usecols=[0],dtype=str)
    trainingfiles = np.loadtxt('SALT3_training_data/SALT3_training_traininghalf.LIST',unpack=True,usecols=[0],dtype=str)
    validfiles = np.loadtxt('SALT3_training_data/SALT3_training_validhalf.LIST',unpack=True,usecols=[0],dtype=str)

    with open('SALT3_training_data/SALT3_training_traininghalf_new.LIST','w') as fout_t,\
         open('SALT3_training_data/SALT3_training_validhalf_new.LIST','w') as fout_v:

        tlist = []
        for t in trainingfiles:
            snid = str(snana.SuperNova(f"../../../SALT3_bkp/examples/SALT3_training_v1.0/SALT3_training_data/{t}").SNID)
            for o in origfiles:
                #if snid in o:
                osnid = None
                with open(f"SALT3_training_data/{o}") as fin:
                    for line in fin:
                        if line.startswith('SNID:'):
                            osnid = line.split()[1].replace('\n','')
                            break
                if snid == osnid:
                    if o not in tlist:
                        print(o,file=fout_t)
                    tlist += [o]
                    break

        vlist = []
        for v in validfiles:
            snid = str(snana.SuperNova(f"../../../SALT3_bkp/examples/SALT3_training_v1.0/SALT3_training_data/{v}").SNID)
            for o in origfiles:
                osnid = None
                with open(f"SALT3_training_data/{o}") as fin:
                    for line in fin:
                        if line.startswith('SNID:'):
                            osnid = line.split()[1].replace('\n','')
                            break
                if snid == osnid:
                    if o not in vlist:
                        print(o,file=fout_v)
                    vlist += [o]
                    break

def main2():
    
    origfiles = np.loadtxt('SALT3_training_data/SALT3_training.LIST',unpack=True,usecols=[0],dtype=str)
    with open('SALT3_training_data/SALT3_training_traininghalf_new.LIST','w') as fout_t,\
         open('SALT3_training_data/SALT3_training_validhalf_new.LIST','w') as fout_v:
        for i,o in enumerate(origfiles):
            if i % 2:
                print(o,file=fout_t)
            else:
                print(o,file=fout_v)
        

        
if __name__ == "__main__":
    #main()
    main2()
