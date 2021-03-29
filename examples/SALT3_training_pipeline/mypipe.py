from saltshaker.pipeline.pipeline import SALT3pipe
import numpy as np

def MyPipe(finput,**kwargs):
    pipe = SALT3pipe(finput=finput)
    pipe.build(data=False,mode='customize',onlyrun=['sim','train','lcfit','biascorsim','biascorlcfit','getmu','cosmofit'])
    skip_config = np.any(['skip_config' in key for key in kwargs.keys()])
    if not skip_config:
        pipe.configure()
        pipe.glue(['sim','train'])
        pipe.glue(['train','lcfit'],on='model')
        pipe.glue(['sim','lcfit'],on='phot')
        pipe.glue(['lcfit','getmu'])
        pipe.glue(['train','biascorlcfit'],on='model')
        pipe.glue(['biascorsim','biascorlcfit'])
        pipe.glue(['biascorlcfit','getmu'])
        pipe.glue(['getmu','cosmofit'])

    return pipe

