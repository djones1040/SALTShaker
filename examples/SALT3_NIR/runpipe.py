#!/usr/bin/env python
# D. Jones, M. Dai - 9/16/19

import sys, argparse, configparser, os


usagestring = """Lightcurve fitting
"""

parser = argparse.ArgumentParser(usage=usagestring, conflict_handler="resolve")
parser.add_argument('config', default='pipeline_training.txt', type=str,
                    help='configuration file')
options = parser.parse_args()
from saltshaker.pipeline.pipeline import *

def runpipe():
    pipe = SALT3pipe(finput=options.config)
    pipe.build(data=False,mode='customize',onlyrun=['lcfit'])
    pipe.configure()

    pipe.run(onlyrun=['lcfit'])

if __name__ == "__main__":
	runpipe()
