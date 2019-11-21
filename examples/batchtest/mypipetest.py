def MyPipe(finput,**kwargs):
    from pipeline import SALT3pipe 
    # write your own pipeline here        
    pipe = SALT3pipe(finput)
    pipe.build(data=False,mode='customize',onlyrun=['byosed','sim','train','lcfit'])
    pipe.configure()
    pipe.glue(['sim','train'])
    pipe.glue(['sim','lcfit'])
    return pipe
