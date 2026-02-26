import jax, jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline

from untangle.algorithm import CMTF_SSD
from untangle.utils import collect_information

def fit_internals(Z, R):
    internals = []
    for rank in range(R.shape[1]):
        rr = R[:, rank]
        zr = Z[:, rank]
        idx = jnp.argsort(zr)
        g = make_smoothing_spline(zr[idx], rr[idx])
        internals.append(g)

    def g(x):
        return jnp.array([gi(xi) for gi, xi in zip(internals, x)])

    return g

n = 3; m = 3; N = 100; rank = 4 # we know that this is a rank four

def f(u):
    u1, u2, u3 = u
    return jnp.array([
        -4 * u1**2 + 8 * u1 * u3 + 6 * u1 - 3 * u3**2 - 8 * u3 - 6,
        2 * u1**2 - 4 * u1 * u3 - 3 * u1 + u2**3 + 6 * u2**2 * u3 + 12 * u2 * u3**2 - u2 + 8 * u3**3 + 2 * u3**2 + u3 + 3,
        -2 * u1**2 + 4 * u1 * u3 + 4 * u1 - 2 * u3**2 - 3 * u3 - u2 - 8,
    ])

X, Y, J = collect_information(f, N, n)

W, V, H, R = CMTF_SSD(J, Y, X, rank, verbose=1)

Z = X @ V

for r in range(rank):
    zr = Z[:, r]
    idx = jnp.argsort(zr)
    plt.scatter(zr[idx], R[:, r][idx])

plt.savefig('test.png')

internals = fit_internals(Z, R)

def inference(x):
    return W @ internals(V.T @ x)

print(inference(jnp.array([0.5, 0.5, 0.5])))
print(f(jnp.array([0.5, 0.5, 0.5])))