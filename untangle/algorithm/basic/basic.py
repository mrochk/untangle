'''Implementation of the paper https://arxiv.org/pdf/1410.4060.'''

import jax, jax.numpy as jnp
from scipy.linalg import block_diag

from beartype import beartype
from beartype.typing import Tuple, Callable
from jaxtyping import jaxtyped, Array, Float, ArrayLike

from untangle.decomposition import cpd
from untangle.utils import make_polynomials

@jaxtyped(typechecker=beartype)
def inference(
    W: Float[ArrayLike, 'n r'], 
    V: Float[ArrayLike, 'm r'], 
    coefs: Float[ArrayLike, 'r d'],
) -> Callable:
    return lambda x: W @ make_polynomials(coefs)(V.T @ x)

@jaxtyped(typechecker=beartype)
def decoupling_basic(
    X: Float[Array, 'N m'], 
    Y: Float[Array, 'N n'], 
    J: Float[Array, 'n m N'], 
    rank: int,
    degree: int = 3,
    verbose: bool = False,
    **tensorly_cpd_kwargs,
) -> Tuple[
    Float[Array, 'n r'], 
    Float[Array, 'm r'], 
    Float[Array, 'r d'],
]:
    '''Basic decoupling algorithm as described in https://arxiv.org/pdf/1410.4060.

    Args description (below) assumes the initial function takes m inputs and returns n outputs.

    Args:
        X: Operating points, of shape (N, m).
        Y: Function outputs for X, of shape (N, n).
        J: Stacked jacobian of shape (n, m, N).
        rank: Rank of the CPD.
        degree: Degree of internal polynomials. Defaults to 3.
        verbose: Verbose output yes/no. Defaults to False.

    All additional arguments are passed to tensorly CP decomposition function.

    Returns W, V and g.
    '''    
    
    log = lambda *values: print(*values) if verbose else None

    log('Computing CP decomposition of J...')
    (weights, (W, V, H)), _ = cpd(tensor=J, rank=rank, **tensorly_cpd_kwargs)
    W = jnp.array(W); V = jnp.array(V)

    log('Recovering internal univariates coefficients...')
    coefs = find_internals_coefficients(X, Y, W, V, degree)

    return W, V, coefs

def find_internals_coefficients(
    X: Float[Array, 'N m'],
    Y: Float[Array, 'N n'],
    W: Float[Array, 'n r'],
    V: Float[Array, 'm r'],
    degree: int,
) -> Float[Array, 'r d']:

    N = X.shape[0]
    rank = W.shape[1]

    def vandermonde_diag(X, d: int):
        def vandermonde_vector(v, d): 
            return jnp.array([v**e for e in range(0, d+1)])

        return block_diag(*(vandermonde_vector(x, d) for x in X))

    W_diag = block_diag(*(W for _ in range(N)))

    Z = jnp.array([V.T @ x for x in X])

    Xk = jnp.concatenate([vandermonde_diag(z, degree) for z in Z], axis=0)

    coefs = jnp.linalg.lstsq(W_diag @ Xk, jnp.concatenate(Y))[0].reshape(rank, -1)
    assert tuple(coefs.shape) == (rank, degree+1)

    return coefs
