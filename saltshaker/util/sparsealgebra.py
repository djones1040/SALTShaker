import numpy as np
from jax import numpy as jnp

class SparseMatrix:
    def __init__(self,*args):
        if len(args)==1:
            mat=args[0]
            self.shape=mat.shape
            self.rowidxs,self.colidxs=np.nonzero(mat)
            self.values=mat[np.nonzero(mat)]
            self.rowidxscompressed=[(idx,np.where(idx==self.rowidxs)[0]) for idx in np.arange(self.shape[0])]
        elif len(args)==4:
            self.shape,self.rowidxs,self.colidxs,self.values=args
            self.rowidxscompressed=[(idx,np.where(idx==self.rowidxs)[0]) for idx in np.arange(self.shape[0])]
        elif len(args)==5:
            self.shape,self.rowidxs,self.colidxs,self.values,self.rowidxscompressed=args
        else:
            raise ValueError()            
        

    def multidot(self,prevector,postvector):
        assert((postvector.shape,)==self.shape[1])
        assert((prevector.shape,)==self.shape[0])
        return (self.values*prevector[self.colidxs]*postvector[self.rowidxs]).sum()

    def dot(self,vector,returnsparse=False):
        assert(vector.shape==(self.shape[1],))
        
        multvals=self.values*vector[self.colidxs]
        if returnsparse:
            idxs=jnp.array([rowidx for rowidx,idxs in (self.rowidxscompressed) if len(idxs)>0])
            vals=jnp.array([multvals[idxs].sum() for rowidx,idxs in (self.rowidxscompressed)if len(idxs)>0])
            return SparseVector(self.shape[0],idxs,vals)
        else:
            return jnp.array([multvals[idxs].sum() for rowidx,idxs in (self.rowidxscompressed)])
         
    def elementwisemultiply(self,vector,axis):
        assert((axis==0) or (axis==1))
        assert((self.shape[axis],)==vector.shape)
        idxs= self.rowidxs if axis==0 else self.colidxs
        return SparseMatrix(self.shape, self.rowidxs,self.colidxs,self.values*vector[idxs],self.rowidxscompressed)

        
        
        
class SparseVector:
    
    def __init__(self,*args):
        if len(args)==1:
            vec=args[0]
            self.shape=vec.shape
            self.idxs=np.nonzero(vec)
            self.vals=vec[np.nonzero(vec)]
        elif len(args)==3:
            self.shape,self.idxs,self.vals=args
        else:
            raise ValueError()
    
    def dot(self,othervec):
        assert(self.shape==othervec.shape)
        return (othervec[self.idxs]*self.vals).sum()
