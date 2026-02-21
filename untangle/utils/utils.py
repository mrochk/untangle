import random
from jax import Array
import jax, jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Iterable, Callable
from tensorly.decomposition import parafac

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

def collect_information(function: Callable, N: int, m: int, key: Array = get_random_key()):
    assert(callable(function))

    jacobian = jax.jacobian(function)
    X = jax.random.uniform(key, shape=(N, m))
    Y = jnp.stack([function(x) for x in X], axis=0)
    J = jnp.stack([jacobian(x) for x in X], axis=2)
    return X, Y, J

def inference(W: Array, V: Array, g: Iterable[Callable]) -> Callable:
    assert all([callable(gi) for gi in g])

    g_inf = lambda x: jnp.array([gi(xi) for gi, xi in zip(g, x)])
    return lambda x: W @ g_inf(V.T @ x)
