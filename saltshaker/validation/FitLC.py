#!/usr/bin/env python
# D. Jones - 5/20/21
"""Utility to fit SNANA light curve
files with sncosmo, given a kcor
and a model directory"""

import numpy as np
import argparse
import sncosmo

class LC:
    def __init__(self):
        pass

    def add_args(self, parser=None, usage=None, config=None):
        if parser == None:
            parser = argparse.ArgumentParser(usage=usage, conflict_handler="resolve")

        # The basics
        parser.add_argument('-v', '--verbose', action="count", dest="verbose",
                            default=0,help='verbosity level')
        parser.add_argument('lcfile', type=str, default=None,
                            help='light curve file')
        parser.add_argument('kcorfile', type=str, default=None,
                            help='kcor file')
        parser.add_argument('-m','--modeldir', type=str, default='$SNDATA_ROOT/models/SALT3/SALT3.K21',
                            help='SALT3 model directory')
        
        return parser

if __name__ == "__main__":
    flc = LC()

    parser = flc.add_args(usage=usagestring)
    args = parser.parse_args()
    flc.options = args
    flc.main()
