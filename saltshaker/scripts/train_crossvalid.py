#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import os
from os import path
import subprocess
import argparse

from saltshaker.training.TrainSALT import RunTraining

parser = argparse.ArgumentParser(description='Run a series of SALTshaker trainings with heldout data ')
parser.add_argument('configfile',type=str, 
                    help='trainsalt configfile')
parser.add_argument('trainfraction',type=float,help='Fraction of data to use in training')
parser.add_argument('nummodels',type=int,help='Number of models to train for crossvalidation')
parser.add_argument('--outputdir',type=str, default=''
                    help='Where to put outputted models')

args = parser.parse_args()


# In[ ]:



with open(args.configfile,'r') as file: 
    for line in file:
        if 'snlists' in line:
            snlists= line.split('=')[-1]
            snlists=snlists.split(',')
            snlists=[x.strip() for x in snlists]
            break
    else:
        raise ValueError("couldn't find snlists")
        
snfiles=[]

for snlist in snlists:
    with open(snlist,'r') as file:
        #print(file.read())
        dirname=path.dirname(snlist)
        snfiles+=[(path.join(dirname,x.strip()) if '/' not in x else x ) for x in file]
    


snfiles=np.array(snfiles)
trainfraction=.9
trainsize=int(len(snfiles)*args.trainfraction)


# In[25]:


for iteration in range(nummodels):
    outputdir=path.join(args.outputdir,f'crossvalidation_{iteration}')
    os.makedirs(outputdir,exist_ok=True)
    trainsample=np.random.choice(snfiles,trainsize,replace=False)
    validsample=snfiles[~np.isin(snfiles,trainsample)]
    listfile=path.join(outputdir,'trainsample.LIST')
    with open(listfile,'w') as file:
        file.write('\n'.join(trainsample))
    with open(path.join(outputdir,'validsample.LIST'),'w') as file:
        file.write('\n'.join(validsample))
    RunTraining().main( configfile, [configfile,'--outputdir',outputdir,'--snlists',listfile])
    #subprocess.call(['trainsalt',configfile,'--outputdir',outputdir,'--snlists',listfile,'--errors_from_hessianapprox','True'])


# In[33]:





# In[35]:





# In[36]:





# In[ ]:




