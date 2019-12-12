#!/usr/bin/env python
# D. Jones, M. Dai - 9/16/19

from salt3.pipeline.pipeline import *

def runpipe():
    pipe = SALT3pipe(finput='pipeline_training.txt')
    pipe.build(data=False,mode='customize',onlyrun=['sim','train','lcfit','getmu'])
    pipe.configure()
    pipe.glue(['sim','train'])
#    pipe.glue(['train','lcfit'],on='model')
#    pipe.glue(['sim','lcfit'],on='phot')
#    pipe.glue(['lcfit','getmu'])
#    import pdb; pdb.set_trace()
    pipe.run(onlyrun=['sim','train'])

if __name__ == "__main__":
	runpipe()
