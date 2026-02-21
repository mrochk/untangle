'''Implementation of the paper https://arxiv.org/pdf/1410.4060.'''

from jax import Array
import jax.numpy as jnp
from typing import Tuple
from jax.typing import ArrayLike
from scipy.linalg import block_diag
from numpy.polynomial import Polynomial

from untangle.utils import cpd

def decoupling_basic(
    X: ArrayLike, 
    Y: ArrayLike, 
    J: ArrayLike, 
    rank: int,
    degree: int = 3,
    verbose: bool = False,
    **args,
) -> Tuple[Array, Array, Tuple[Polynomial, ...]]:
    '''Basic decoupling algorithm as described in https://arxiv.org/pdf/1410.4060.

    Args description (below) assumes the initial function takes m inputs and returns n outputs.

    Args:
        X (ArrayLike): Operating points, of shape (N, m).
        Y (ArrayLike): Function outputs for X, of shape (N, n).
        J (ArrayLike): Stacked jacobian of shape (n, m, N).
        rank (int): Rank of the CPD.
        degree (int, optional): Degree of internal polynomials. Defaults to 3.
        verbose (bool, optional): Verbose output yes/no. Defaults to False.

    All additional arguments are passed to tensorly CP decomposition function.

    Returns W, V and g.
    '''    
    
    log = lambda *values: print(*values) if verbose else None

    log('Computing CP decomposition of J...')
    _, (W, V, H) = cpd(tensor=J, rank=rank, verbose=int(verbose), **args)

    log('Recovering internal univariates...')
    g = recover_internals(X, Y, W, V, degree)

    return W, V, g

def recover_internals(X: ArrayLike, Y: ArrayLike, W: ArrayLike, V: ArrayLike, degree: int):
    N = X.shape[0]
    rank = W.shape[1]

    def vandermonde_diag(X, d: int):
        def vandermonde_vector(v, d): 
            return jnp.array([v**e for e in range(0, d+1)])

        return block_diag(*(vandermonde_vector(x, d) for x in X))

    W_diag = block_diag(*(W for _ in range(N)))

    Z = jnp.array([V.T @ x for x in X])

    Xk = jnp.concatenate([vandermonde_diag(z, degree) for z in Z], axis=0)

    coefs, _, _, _ = jnp.linalg.lstsq(W_diag @ Xk, jnp.concatenate(Y))
    coefs = coefs.reshape((rank, -1))

    return tuple([Polynomial(c) for c in coefs])
