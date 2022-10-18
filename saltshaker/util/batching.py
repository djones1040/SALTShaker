from saltshaker.training import datamodels
import jax
from jax import numpy as jnp
from jax.experimental import sparse
import numpy as np

from scipy import optimize, stats

import warnings


def optimizepaddingsizes(numbatches,datasizes):
    def parstobins(pars):
        pars=np.abs(pars)
        pars=np.concatenate([[0],pars,[1.0001]])
        return np.cumsum(pars/pars.sum())

    def loss(pars):#    binlocs=[14,60]
        bins=parstobins(pars)* max(datasizes) 
        spacerequired=stats.binned_statistic(datasizes, datasizes,statistic='count', bins=bins).statistic* np.floor(bins[1:])
        return spacerequired.sum()
    ndim=numbatches-1

    vertices=np.random.default_rng(seed=13241245642435).random(size=(ndim+1,ndim))
    
    result=optimize.minimize(loss,[0]*ndim,  method='Nelder-Mead', options= {

        'initial_simplex':vertices})
    pars=result.x
    bins=np.floor(( parstobins(pars)[1:])*max(datasizes)).astype(int)
    bins[-1]=max(datasizes)
    return bins



def batchdatabysize(data):

    batcheddata={ }

    for x in data:
        key=len(x)
        if key in batcheddata:
            
            batcheddata[key] += [x]
        else:
            batcheddata[key] =[x]
            
    def repackforvmap(data):
        __ismapped__=data[0].__ismapped__
        unpacked=[x.unpack() for x in data]
        
        for j,varname in enumerate( data[0].__slots__):

            vals=np.array([(unpacked[i][j]) for i in range(len(unpacked))])
            if isinstance(vals[0],sparse.BCOO) : 
                yield sparse.bcoo_concatenate([x.reshape((1,*x.shape)).update_layout(n_batch=1) for x in vals] ,dimension=0)
            else: 
                if not (varname in __ismapped__ ):
                    assert(np.all(vals[0]==vals), "Unmapped quantity different between different objects")
                    yield vals[0]
                else:
                    yield  np.stack(vals,axis=0) 
    
    return [list(repackforvmap(x)) for x in batcheddata.values()]



def walkargumenttree(x,targetsize,ncalls=0):
    if isinstance(x,dict):
        return {key:walkargumenttree(val,targetsize,ncalls+1) for key,val in x.items()} 
    elif hasattr(x,'shape') :
        if len(x.shape)>0 and x.shape[0]==targetsize:
            return 0
        else: return None
    elif hasattr(x,'__len__'):
        if len(x)==targetsize and ncalls>0:
            return 0
        else :
            return type(x)(walkargumenttree(y,targetsize,ncalls+1) for y in x )
    else : return None

def batchedmodelfunctions(function,batcheddata, dtype,flatten=False):
    if issubclass(dtype,datamodels.modeledtrainingdata) :
        pass
    elif dtype in ['light-curves','spectra']:
        if dtype=='light-curves':
            dtype=datamodels.modeledtraininglightcurve
        else:
            dtype=datamodels.modeledtrainingspectrum
    else:
        raise ValueError(f'Invalid datatype {dtype}')

    batchedindexed= [dict([(x,y) for x,y in zip(dtype.__slots__,batch) if x in 
                 dtype.__indexattributes__ ]) for batch in batcheddata]
    
    def vectorized(pars,*batchedargs,**batchedkwargs):
        result=[]
        if len(batchedargs)==0:
            batchedargs=[[]]*len(batcheddata)
        else:
            batchedargs= list(zip(*[x if hasattr(x,'__len__') else [x]*len(batcheddata) for x in batchedargs]))
        
        if (batchedkwargs)=={}:
            batchedkwargs=[{}]*len(batcheddata)
        else:
            batchedkwargs= [{y:x[i] if hasattr(x,'__len__') else x for y,x in batchedkwargs.items()}
                            for i in range(len(batcheddata))]
        for batch,indexdict,args,kwargs in zip(batcheddata,batchedindexed,batchedargs,batchedkwargs):

            def funpacked (lc,pars,kwargs,*args):
                lc= dtype.repack(lc)
                pars=datamodels.SALTparameters.tree_unflatten((),pars)

                return function(lc,pars,*args,**kwargs)
#             import pdb;pdb.set_trace()
            newpars=datamodels.SALTparameters(indexdict, pars)
            parsvmapped=newpars.tree_flatten()[0]
            newargs=kwargs,*args

            #Determine which axes of the arguments correspond to the lightcurve data
            targetsize=indexdict['ix0'].size
            #The batched data has prenoted which arguments are to be mapped over; otherwise need to attempt to determine it programatically
            try:
                inaxes= [(0 if (x in dtype.__ismapped__) else None) for x in dtype.__slots__],newpars.mappingaxes,*walkargumenttree(newargs,targetsize)
                mapped=jax.vmap(  funpacked,in_axes= 
                    inaxes
                            )(
                    batch,list(parsvmapped),*newargs
                )
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
            if flatten: mapped=mapped.flatten()
            result+=[mapped ]
        if flatten:
            return jnp.concatenate(result)
        else: 
            return result
    return vectorized
