import random
import jax, jax.numpy as jnp
from functools import partial
from tensorly.decomposition import parafac

from beartype import beartype
from beartype.typing import Callable, Tuple
from jaxtyping import jaxtyped, Float, Array, ArrayLike

cpd = parafac

def unfold_kolda(tensor: ArrayLike, mode: int) -> Array:
    '''Tensor unfolding as defined in "Tensor decompositions and applications." from Kolda and Bader.'''
    return jnp.reshape(jnp.moveaxis(tensor, mode, 0), shape=(tensor.shape[mode], -1), order='F')

def get_random_key() -> Array:
    random_int = random.randint(0, 1000)
    return jax.random.key(random_int)

def search_rank(tensor: ArrayLike, max_rank: int = 100, **args):
    for rank in range(1, max_rank+1):
        _, errors = cpd(tensor, rank, return_errors=True, **args)
        last_error = errors[-1]
        if last_error < 0.01: return rank
    return -1

@jaxtyped(typechecker=beartype)
def collect_information(
    function: Callable, 
    N: int, 
    m: int, 
    range = (0, 1),
    key: Array = get_random_key(),
) -> Tuple[
    Float[Array, 'N m'], 
    Float[Array, 'N n'], 
    Float[Array, 'n m N'],
]:
    assert(callable(function))

    lo, hi = range

    jacobian = jax.jacobian(function)
    X = jax.random.uniform(key, shape=(N, m), minval=lo, maxval=hi)
    Y = jnp.stack([function(x) for x in X], axis=0)
    J = jnp.stack([jacobian(x) for x in X], axis=2)
    return X, Y, J

def make_polynomial(coefs: Float[Array, 'd']) -> Callable:
    return partial(jnp.polyval, jnp.flip(coefs))

def make_polynomials(coefs: Float[Array, 'n d']) -> Callable:
    polynomials = [make_polynomial(c) for c in coefs]
    def _(x): return jnp.array([f(xi) for f, xi in zip(polynomials, x)])
    return _
