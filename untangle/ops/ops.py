'''Tensor operations that are not by default in jax.numpy.'''

import jax.numpy as jnp
from jaxtyping import jaxtyped, Float, Array, ArrayLike
from beartype import beartype

def unfold_kolda(tensor: ArrayLike, mode: int) -> ArrayLike:
    '''Tensor unfolding as defined in "Tensor decompositions and applications" from Kolda and Bader.'''
    return jnp.reshape(jnp.moveaxis(tensor, mode, 0), shape=(tensor.shape[mode], -1), order='F')

@jaxtyped(typechecker=beartype)
def khatri_rao(A: Float[Array, 'm k'], B: Float[Array, 'n k']) -> Float[Array, 'mn k']:
    m, k = A.shape
    n, _ = B.shape
    return (A[:, None, :] * B[None, :, :]).reshape(m*n, k)
