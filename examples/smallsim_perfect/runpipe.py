#!/usr/bin/env python
# D. Jones, M. Dai - 9/16/19

from salt3.pipeline.pipeline import *

def runpipe():
    pipe = SALT3pipe(finput='sampleinput.txt')
    pipe.build(data=False,mode='customize',onlyrun=['sim','train','lcfit','getmu'])
    pipe.configure()
    pipe.glue(['sim','train'])
    pipe.glue(['train','lcfit'],on='model')
    pipe.glue(['sim','lcfit'],on='phot')
    pipe.glue(['lcfit','getmu'])
    pipe.run(onlyrun=['sim','train','lcfit','getmu'])

if __name__ == "__main__":
	runpipe()
