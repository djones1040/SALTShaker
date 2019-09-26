import sys
sys.path.append('/home/midai/salt3_local/SALT3/salt3')
import pandas as pd
import pipeline.pipeline
from pipeline.pipeline import *

pipe = SALT3pipe(finput='sampleinput.txt')
pipe.build(data=False,mode='customize',onlyrun=['byosed','sim','train'])
pipe.configure()
pipe.glue(['sim','train'])
pipe.run()
pipe.build(data=False,mode='customize',onlyrun=['lcfit','getmu','cosmofit'])
pipe.configure()
pipe.glue(['train','lcfit'],on='model')
pipe.glue(['sim','lcfit'],on='phot')
pipe.glue(['lcfit','getmu'])
pipe.glue(['getmu','cosmofit'])
pipe.run()
