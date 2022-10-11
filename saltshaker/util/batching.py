from saltshaker.training import datamodels
import jax
from jax import numpy as jnp
from jax.experimental import sparse
import numpy as np

from scipy import optimize, stats

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
        unpacked=[x.unpack() for x in data]
        for j in range(len(unpacked[0])):
            vals=np.array([(unpacked[i][j]) for i in range(len(unpacked))])
            if isinstance(vals[0],sparse.BCOO) : 
                yield sparse.bcoo_concatenate([x.reshape((1,*x.shape)).update_layout(n_batch=1) for x in vals] ,dimension=0)
            else: 
                if np.all(vals[0]==vals):
                    yield vals[0]
                else:
                    yield  np.stack(vals,axis=0) 
    
    return [list(repackforvmap(x)) for x in batcheddata.values()]



def walkargumenttree(x,targetsize):
    if isinstance(x,dict):
        return {key:walkargumenttree(val,targetsize) for key,val in x.items()} 
    elif hasattr(x,'shape') :
        if len(x.shape)>0 and x.shape[0]==targetsize:
            return 0
        else: return None
    elif hasattr(x,'__len__'):
        if len(x)==targetsize:
            return 0
        else :
            return type(x)(walkargumenttree(y,targetsize) for y in x )
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
        if len(batchedkwargs)=={}:
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
            newargs=batch,list(parsvmapped),kwargs,*args
            
            #Determine which axes of the arguments correspond to the lightcurve data
            targetsize=indexdict['ix0'].size
            
            inaxes=walkargumenttree(newargs,targetsize)
            mapped=jax.vmap(  funpacked,in_axes= 
                inaxes
                        )(
                *newargs
            )
            if flatten: mapped=mapped.flatten()
            result+=[mapped ]
        if flatten:
            return jnp.concatenate(result)
        else: 
            return result
    return vectorized
