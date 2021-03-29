#!/usr/bin/env python
# D. Jones, M. Dai - 9/16/19

from saltshaker.pipeline.pipeline import *

def runpipe():
    pipe = SALT3pipe(finput='pipeline_training.txt')
    pipe.build(data=False,mode='customize',onlyrun=['train','lcfit'])
    pipe.configure()

    pipe.run(onlyrun=['lcfit'])

if __name__ == "__main__":
	runpipe()
