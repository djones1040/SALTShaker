from saltshaker.pipeline.pipeline import SALT3pipe

def MyPipe(finput,**kwargs):
    pipe = SALT3pipe(finput=finput)
    pipe.build(data=False,mode='customize',onlyrun=['sim','train','lcfit','biascorsim','biascorlcfit','getmu','cosmofit'])
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

