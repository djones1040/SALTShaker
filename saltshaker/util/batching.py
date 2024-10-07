from saltshaker.training import datamodels
import jax
from jax import numpy as jnp
from jax.experimental import sparse
import numpy as np
import pickle

from scipy import optimize, stats

import warnings


def optimizepaddingsizes(numbatches,datasizes):
    """Given a set of the sizes of batches of data, and a number of batches to divide them into, each of which is zero-padded, determine the zero-paddings that will minimize the additional space required to store the results
    """
    #Transform a set of n-1 unconstrained parameters into n+1 bin edges, starting at 0 and monotonically increasing to 1
    def parstobins(pars,largest):
        #Probably a better transform to use here!
        pars=np.abs(pars)
        pars=np.concatenate([[0],pars,[1.]])
        bins=np.cumsum(pars/pars.sum())
        bins*=largest
        bins[-1]+=.1
        return bins
    
    #Define the loss function to be used here: defined here as just the space required to store the result
    def loss(pars):#    binlocs=[14,60]
        bins=parstobins(pars,max(datasizes))
        spacerequired=stats.binned_statistic(datasizes, datasizes,statistic='count', bins=bins).statistic* np.floor(bins[1:])
        return spacerequired.sum()
    ndim=numbatches-1

    #Integer programming doesn't seem to have a particularly easy implementation in python
    #I use Nelder-Mead method since it doesn't use gradients, and it gets reasonable results
    #Starting vertices are chosen based on a fixed rng seed to eliminate reproducibility error
    vertices=np.random.default_rng(seed=13241245642435).random(size=(ndim+1,ndim))
    result=optimize.minimize(loss,[0]*ndim,  method='Nelder-Mead', options= {

        'initial_simplex':vertices})
    pars=result.x
    bins=parstobins(pars,max(datasizes))
    padsizes= stats.binned_statistic(datasizes, datasizes,statistic= lambda x: x.max() if x.size>0 else 0, bins=bins).statistic
    padsizes[np.isnan(padsizes)]=0
    padsizes=padsizes.astype(int)
    finalspacecost=(stats.binned_statistic(datasizes, datasizes,statistic='count', bins=bins).statistic* padsizes).sum()
    padsizes=padsizes[np.nonzero(padsizes)]
    return padsizes,sum(datasizes)/finalspacecost



def batchdatabysize(data,outdir,prefix=''):
    """ Given a set of  data, divide them into batches each of a fixed size, for easy use with jax's batching methods. Quantities that are marked as 'mapped' by their objects will be turned into arrays, otherwise the values are tested for equality and given as unmapped objects  """
    batcheddata={ }

    for x in data:
        #determine length of each piece of data
        key=len(x)
        #assign data to an appropriate entry in dictionary based on length
        if key in batcheddata:
            
            batcheddata[key] += [x]
        else:
            batcheddata[key] =[x]
    
    #Given a batch of data, unpacks it and stacks it along the first axis for use with jax's vmap method   
    def repackforvmap(data):
        __ismapped__=data[0].__ismapped__
        #Given n data with m attributes, this yields an n x m list of lists
        unpacked=[x.unpack() for x in data]
        
        #Want to convert that into an m-element list of n-element arrays or single values
        for j,varname in enumerate( data[0].__slots__):

            vals=([(unpacked[i][j]) for i in range(len(unpacked))])
            #If it's a sparse array, concatenate along new "batched" axis for use with vmap
            if isinstance(vals[0],sparse.BCOO) : 
                yield sparse.bcoo_concatenate([x.reshape((1,*x.shape)).update_layout(n_batch=1) for x in vals] ,dimension=0)
            else: 
                if not (varname in __ismapped__ ):
                    #If an attribute is not to be mapped over, the single value is set, and it is verified that it is the same for all elements
                    assert(np.all(vals[0]==vals), "Unmapped quantity different between different objects")
                    yield vals[0]
                else:
                    yield  np.stack(vals,axis=0) 
    #Returns a list of batches of data suitable for use with the batchedmodelfunctions function
    for i,x in enumerate(batcheddata.values()):
        jax.clear_caches()
        with open(f'{outdir}/caching_{prefix}_{i}.pkl','wb') as fout:
            pickle.dump({'data':list(repackforvmap(x))},fout)

    #return [list(repackforvmap(x)) for x in batcheddata.values()]



def walkargumenttree(x,targetsize,ncalls=0):
    """ Walks argument trees checking for whether the first axis of each leaf matches the size of the mapped axes; if so return 0, otherwise None"""
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

def batchedmodelfunctions(function,batcheddata, dtype,flatten=False,sum=False):
    """Constructor function to map a function that takes a modeledtrainingdata object as first arg and a SALTparameters object as second arg over batched, zero-padded data"""
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
        if sum: result=0
        else: result=[]
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
#             try:
            inaxes= [(0 if (x in dtype.__ismapped__) else None) for x in dtype.__slots__],newpars.mappingaxes,*walkargumenttree(newargs,targetsize)
            mapped=jax.vmap(  funpacked,in_axes= 
                inaxes
                        )(
                batch,list(parsvmapped),*newargs
            )
#             except Exception as e:
#                 print(e)
#                 import pdb;pdb.set_trace()
            if flatten: mapped=mapped.flatten()
            if sum: 
                result+=mapped.sum()
            else:
                result+=[mapped ]
        if flatten:
            return jnp.concatenate(result)
        else: 
            return result
    return vectorized
