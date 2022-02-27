#!/usr/bin/env python
# D. Jones - 2/27/22
# create five training sets, each omitting 20% of the data
# we can co-opt a more complicated procedure by looking at salt3train_snparams.out, choosing 20% of the SNe, and deleting them from the input pkmjd list
#
# then have to train 10x models, 5 w/ host comp, 5 w/o host comp
#
# then we have to rebuild the distance fitting code to get apples-to-apples comparisons going

import numpy as np


def main():
    snids = np.loadtxt(
        'SALT3.host/salt3train_snparams.txt',
        unpack=True,usecols=[0],dtype=str)
    snids = np.unique(snids)
    
    list1,list2,list3,list4,list5 = [],[],[],[],[]
    for s in snids:
        randval = np.random.uniform(0,1)
        if randval <= 0.2:
            list1 += [s]
        elif randval > 0.2 and randval <= 0.4:
            list2 += [s]
        elif randval > 0.4 and randval <= 0.6:
            list3 += [s]
        elif randval > 0.6 and randval <= 0.8:
            list4 += [s]
        elif randval > 0.8:
            list5 += [s]

    for i,l in enumerate([list1,list2,list3,list4,list5]):
        with open('SALT3_PKMJD_INIT_PP.LIST') as fin, \
             open(f'SALT3_PKMJD_INIT_PP_{i+1}.LIST','w') as fout:

            for line in fin:
                snid = line.split()[0]
                if snid not in l and snid in snids:
                    print(line.replace('\n',''),file=fout)
    
    return

if __name__ == "__main__":
    main()
