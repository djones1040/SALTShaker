import jax
from jax.experimental import sparse
def jaxoptions(fun,static_argnums=None,static_argnames=None,jac_argnums=[0]):
    blank=lambda x,*args,**kwargs: x
    bools=[True,False]
    permutations={(jitted,jacced,forward): (jax.jit if jitted else blank)(
            ((jax.jacfwd if forward else jax.jacrev) if jacced else blank)((fun),argnums=jac_argnums)
        ,static_argnums= static_argnums,static_argnames=static_argnames ) for jitted in bools for jacced in bools for forward in bools}
    
    def wrapped(*args,jit=True,jac=False,forward=False,**kwargs):
        return permutations[(jit,jac,forward)](*args,**kwargs)
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
