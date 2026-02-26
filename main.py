import jax, jax.numpy as jnp

from untangle.algorithm import decoupling_basic
from untangle.utils import collect_information

n = 3; m = 3; N = 10; rank = 4 # we know that this is a rank four

def f(u):
    u1, u2, u3 = u
    return jnp.array([
        -4 * u1**2 + 8 * u1 * u3 + 6 * u1 - 3 * u3**2 - 8 * u3 - 6,
        2 * u1**2 - 4 * u1 * u3 - 3 * u1 + u2**3 + 6 * u2**2 * u3 + 12 * u2 * u3**2 - u2 + 8 * u3**3 + 2 * u3**2 + u3 + 3,
        -2 * u1**2 + 4 * u1 * u3 + 4 * u1 - 2 * u3**2 - 3 * u3 - u2 - 8,
    ])

X, Y, J = collect_information(f, N, n)

decoupling, _ = decoupling_basic(X, Y, J, rank, verbose=1)

x = jnp.array([0.5, 0.5, 0.5])

print(f(x))
print(decoupling(x))
