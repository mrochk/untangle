from jaxtyping import jaxtyped, Float, Array
from beartype import beartype

@jaxtyped(typechecker=beartype)
def khatri_rao(A: Float[Array, 'm k'], B: Float[Array, 'n k']):
    m, k = A.shape
    n, _ = B.shape
    return (A[:, None, :] * B[None, :, :]).reshape(m*n, k)
