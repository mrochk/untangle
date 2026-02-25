import jax, jax.numpy as jnp

from untangle.decomposition import cpd
from untangle.utils import reconstruct_tensor

tensor = jax.random.normal(jax.random.key(42), (3, 4, 10))

factors, weights = cpd(tensor, 10, max_iters=1000, verbose=1)

res = reconstruct_tensor(factors, weights)

print(tensor.shape, res.shape)

print(tensor[0])
print(res[0])
