import random
import jax, jax.numpy as jnp
from functools import partial

from beartype import beartype
from beartype.typing import Callable, Tuple
from jaxtyping import jaxtyped, Float, Array

def make_log(verbose: int): return lambda *args: print(*args) if verbose > 0 else None

def get_random_key() -> Array:
    random_int = random.randint(0, 1000)
    return jax.random.key(random_int)

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
    '''Collect outputs and stacked jacobian of function with respect to X.'''

    assert(callable(function))

    lo, hi = range
    X = jax.random.uniform(key, shape=(N, m), minval=lo, maxval=hi)

    Y = jax.vmap(function)(X)
    J = jax.vmap(jax.jacobian(function))(X).transpose((1, 2, 0))

    return X, Y, J

def make_polynomial(coefs: Float[Array, 'd']) -> Callable:
    return partial(jnp.polyval, jnp.flip(coefs))

def make_polynomials(coefs: Float[Array, 'n d']) -> Callable:
    polynomials = [make_polynomial(c) for c in coefs]
    def _(x): return jnp.array([f(xi) for f, xi in zip(polynomials, x)])
    return _

def reconstruct_tensor(factors, weights):
    W, V, H = factors

    N, m, n = H.shape[0], V.shape[0], W.shape[0]
    tensor = jnp.zeros(shape=(n, m, N))

    rank = W.shape[1]
    for r in range(rank):
        rank1 = W[:, r][:, None, None] * V[:, r][None, :, None] * H[:, r][None, None, :]
        tensor += weights[r] * rank1

    return tensor

def relative_error(tensor, factors, weights = None):
    if weights is None: weights = jnp.ones(factors[0].shape[1])
    return jnp.linalg.norm(tensor - reconstruct_tensor(factors, weights)) / jnp.linalg.norm(tensor)
