from saltshaker.pipeline.pipeline import SALT3pipe

def MyPipe(finput,**kwargs):
    pipe = SALT3pipe(finput=finput)
    pipe.build(data=False,mode='customize',
			   onlyrun=['train','lcfit'])
    pipe.configure()
    pipe.glue(['train','lcfit'],on='model')

    return pipe

