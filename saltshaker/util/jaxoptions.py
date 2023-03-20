import jax
from jax.experimental import sparse
from scipy import sparse as scisparse 
import sys
from jax import numpy as jnp
from tqdm.notebook import trange
from jax.nn import one_hot
def in_ipynb():
    try:
        cfg = get_ipython().config 
        return True

    except NameError:
        return False

if in_ipynb():
    from tqdm.notebook import tqdm,trange
else:
    from tqdm import tqdm,trange

usetqdm= sys.stdout.isatty() or in_ipynb()


def sparsejac(fun,difffunc,argnums, forward ):
    if len(argnums)>1: raise NotImplementedError('Sparse jacobians differentiated w.r.t. multiple arguments have not been implemented')
    diffargidx=argnums[0]
    
    def sparsejacconstruct(*args,**kwargs):
    
        numout=jax.eval_shape( (lambda x: fun(*args[:diffargidx],x,*args[diffargidx+1:],**kwargs)),args[diffargidx]).shape
        shapein=args[diffargidx].shape
        
        if len(numout)>1 or len(shapein)>1: raise NotImplementedError('Jacobians of non-vectors is not implemented')
        jshape=(numout[0],shapein[0])
        localdiff=difffunc
        
        matrixproduct = jax.jit( jax.vmap(lambda vec: difffunc(vec,*args,**kwargs)) )
        
        diffchunksize=100 
        
        
        rangefunc= trange if usetqdm else range
        if forward:
            return scisparse.hstack([scisparse.csc_matrix(
                    matrixproduct( one_hot(jnp.arange(i*diffchunksize, (i+1)*diffchunksize ), jshape[1] ))
                ).T for i in (rangefunc(jshape[1] //diffchunksize +1 ))])[:, :jshape[1]]
        else:
            return sparse.bcoo_concatenate([sparse.BCOO.fromdense(
                    localdiff(jnp.zeros(jshape[0]).at[i].set(1) ,*args,**kwargs)
                ).reshape(1,jshape[1])
                                     for i in (rangefunc(jshape[0]))],dimension=0)
    return sparsejacconstruct

def wrapjvpmultipleargs(fun,argnums):
    if len(argnums)>1: raise NotImplementedError('Wrapped jacobian-vector products differentiated w.r.t. multiple arguments have not been implemented')
    diffargidx=argnums[0]
    return lambda vec, *args,**kwargs: jax.jvp(
        lambda x: fun(*args[:diffargidx],x,*args[diffargidx+1:],**kwargs)

                     ,[args[diffargidx]], [vec])[1]

def wrapvjpmultipleargs(fun,argnums):
    if len(argnums)>1: raise NotImplementedError('Wrapped vector-jacobian products differentiated w.r.t. multiple arguments have not been implemented')
    diffargidx=argnums[0]
    return lambda vec, *args,**kwargs: jax.vjp(
        lambda x: fun(*args[:diffargidx],x,*args[diffargidx+1:],**kwargs)

                     ,args[diffargidx])[1](vec)[0]
    

def jaxoptions(fun,static_argnums=None,static_argnames=None,diff_argnum=0, jitdefault=False):
    blank=lambda x,*args,**kwargs: x
    bools=[True,False]
    gradoptions=[ 
None,'jacfwd','jacrev','jvp','vjp',
        'sparsejacfwd','sparsejacrev',
        'grad','valueandgrad'
    ]
    
    permutations={}
    for jitted in bools:
        compilefunc=(jax.jit if jitted else blank)
        for difftype in gradoptions:
            compileargs=[static_argnums,static_argnames,]
            match difftype:
                case 'valueandgrad':
                    gradfunc= jax.value_and_grad(fun,argnums=diff_argnum)
                case 'grad':
                    gradfunc= jax.grad(fun,argnums=diff_argnum)
                case 'jacfwd':
                    gradfunc= jax.jacfwd(fun,argnums=[diff_argnum] )
                case 'jacrev':
                    gradfunc= jax.jacrev(fun,argnums=[diff_argnum] )
                case 'jvp':
                    gradfunc=wrapjvpmultipleargs(fun,argnums=[diff_argnum] )
                    if not (static_argnums is None): compileargs[0]=[x+1 for x in static_argnums]
                case 'vjp':
                    gradfunc=wrapvjpmultipleargs(fun,argnums=[diff_argnum] )
                    if not (static_argnums is None): compileargs[0]=[x+1 for x in static_argnums]
                case 'sparsejacfwd':
                    permutations[(jitted,difftype)]= sparsejac( 
                        permutations[(jitted,None)],
                        permutations[(jitted, 'jvp')],[diff_argnum],True)
                    continue
                case 'sparsejacrev':
                    permutations[(jitted,difftype)]= sparsejac( 
                        permutations[(jitted,None)],
                        permutations[(jitted, 'vjp')],[diff_argnum],False)
                    continue
                case None:
                    gradfunc=fun
            permutations[(jitted,difftype)]= compilefunc(gradfunc,
                    static_argnums= compileargs[0],static_argnames=compileargs[1])
            
                
    def wrapped(*args,jit=jitdefault,diff=None,**kwargs):
        if diff in ['jvp','vjp']:
            difffunction=permutations[(jit,diff)]
            return lambda vec: difffunction(vec,*args,**kwargs)
        return permutations[(jit,diff)](*args,**kwargs)
    wrapped.__doc__=fun.__doc__
    return wrapped

def sparsejaxoptions(fun,static_argnums=None,static_argnames=None,jac_argnums=[0]):
    blank=lambda x,*args,**kwargs: x
    bools=[True,False]
    permutations={(jitted,jacced,forward): (jax.jit if jitted else blank)(
            ((jax.jacfwd if forward else jax.jacrev) if jacced else blank)(sparse.sparsify(fun),argnums=jac_argnums)
        ,static_argnums= static_argnums,static_argnames=static_argnames ) for jitted in bools for jacced in bools for forward in bools}
    
    def wrapped(*args,jit=True,jac=False,forward=False,**kwargs):
        return permutations[(jit,jac,forward)](*args,**kwargs)
    wrapped.__doc__=fun.__doc__
    return wrapped
