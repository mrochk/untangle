'''
Implementation of the method described in Hollander's PhD thesis, section 8.2.3.
'''

import jax, jax.numpy as jnp
from scipy.linalg import block_diag

from beartype import beartype
from beartype.typing import Tuple, Callable
from jaxtyping import jaxtyped, Array, Float, ArrayLike

from untangle.decomposition import cpd
from untangle.utils import make_polynomials

def decoupling_basic_constrained(
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
    pass