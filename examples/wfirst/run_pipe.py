import sys
sys.path.append('/project2/rkessler/SURVEYS/WFIRST/ROOT/SALT3/saltshaker')
import pandas as pd
from importlib import reload
import pipeline.pipeline
#reload(pipeline.pipeline)
from pipeline.pipeline import *
import plotting.plots as plots
import util.adjfitres as adjfitres

def run_wfirst_pipeline():

	pipe = SALT3pipe(finput='run_wfirst.txt')
	pipe.build(data=False,mode='customize',onlyrun=['sim','lcfit','getmu'])
	#pipe.build(data=False,mode='customize',onlyrun=['biascorsim','biascorlcfit'])
	pipe.configure()
	
	pipe.glue(['sim','lcfit'],on='phot')

	pipe.glue(['lcfit','getmu'])
	pipe.run(onlyrun=['sim','lcfit','getmu'])
	
	
	#pipe.run(onlyrun=['biascorsim','biascorlcfit'])
	

if __name__=='__main__':
	run_wfirst_pipeline()
