def MyPipe(finput,**kwargs):
    from pipeline import SALT3pipe 
    # write your own pipeline here        
#    pipe = SALT3pipe(finput)
#    pipe.build(data=False,mode='customize',onlyrun=['byosed','sim','train','lcfit'])
#    pipe.configure()
#    pipe.glue(['sim','train'])
#    pipe.glue(['sim','lcfit'])
#    return pipe

    pipe = SALT3pipe(finput=finput)
    pipe.build(data=False,mode='customize',onlyrun=['sim','train','lcfit','getmu'])
    pipe.configure()
    #pipe.glue(['byosed','sim'])
    pipe.glue(['sim','lcfit'],on='phot')
    pipe.glue(['sim','train'])
    pipe.glue(['train','lcfit'],on='model')
    pipe.glue(['lcfit','getmu'])
    #pipe.run(onlyrun=['sim','train','lcfit','getmu'])
    #pipe.glue(['getmu','cosmofit'])
    #pipe.run(onlyrun=['sim','train','lcfit','getmu'])
    return pipe

