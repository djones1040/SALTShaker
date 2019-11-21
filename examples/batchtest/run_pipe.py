import sys
sys.path.append('/project2/rkessler/SURVEYS/WFIRST/ROOT/SALT3/salt3')
import pandas as pd
from importlib import reload
import pipeline.pipeline
#reload(pipeline.pipeline)
from pipeline.pipeline import *

def test_pipeline():
    pipe = SALT3pipe(finput='sampleinput.txt')
    pipe.build(data=False,mode='customize',onlyrun=['sim','train','lcfit','getmu'])
    pipe.configure()
    #pipe.glue(['byosed','sim'])
    pipe.glue(['sim','lcfit'],on='phot')
    #pipe.glue(['train','lcfit'],on='model')
    pipe.glue(['lcfit','getmu'])
    #pipe.run(onlyrun=['sim','train','lcfit','getmu'])
    #pipe.glue(['getmu','cosmofit'])
    pipe.run(onlyrun=['sim','train','lcfit','getmu'])

if __name__=='__main__':
    test_pipeline()
