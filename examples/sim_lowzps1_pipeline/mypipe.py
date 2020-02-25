from pipeline import SALT3pipe

def MyPipe(finput,**kwargs):
    pipe = SALT3pipe(finput=finput)
    pipe.build(data=False,mode='customize',onlyrun=['sim','train'])
    pipe.configure()
    pipe.glue(['sim','train'])

    return pipe
