#!/usr/bin/env python
# D. Jones, M. Dai - 9/16/19

from salt3.pipeline.pipeline import *
import sys

def runpipe():
    pipe = SALT3pipe(finput=sys.argv[1])
    pipe.build(data=False,mode='customize',onlyrun=['lcfit'])
    pipe.configure()

    pipe.run(onlyrun=['lcfit'])

if __name__ == "__main__":
	runpipe()
