#!/usr/bin/env python
# D. Jones - 10/27/20

import numpy as np
from saltshaker.util import snana

def main():
    surveys_orig = ['Hamuy1996_LC','Riess1999_LC','Jha2006_LC','Hicken2009_LC','SDSS','SNLS3_LC',
                    'OTHER_LOWZ_LC','CFA4p1','CFA4p2']
    surveys_lc = ['Hamuy1996_LC','Riess1999_LC','Jha2006_LC','Hicken2009_LC','SDSS','SNLS3_LC',
                  'OTHER_LOWZ_LC','PS1MD','DES','FOUNDATION','CFA4p1','CFA4p2']

    snfiles = np.loadtxt('/Users/David/Dropbox/research/SALT3/examples/SALT3_training_v1.0/SALT3_training_data/SALT3_training.LIST',unpack=True,dtype=str)

    with open('SALT3_training_data/SALT3_training_traininghalf.LIST','w') as fouttrain,\
         open('SALT3_training_data/SALT3_training_validhalf.LIST','w') as foutvalid:
        for i,s in enumerate(snfiles):
            print(f"SALT3_training_data/{s}")
            sn = snana.SuperNova(f"SALT3_training_data/{s}")
            if sn.SURVEY in surveys_orig:
                print(s,file=fouttrain)
                print(s,file=foutvalid)
            else:
                if i % 2: print(s,file=fouttrain)
                else: print(s,file=foutvalid)

if __name__ == "__main__":
    main()

