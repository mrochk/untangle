import jax, jax.numpy as jnp
from scipy.interpolate import make_smoothing_spline

from jaxtyping import jaxtyped, Array, Float
from beartype import beartype

from untangle.utils import get_random_key, relative_error, make_log
from untangle.ops import unfold_kolda, khatri_rao

def init(J: Float[Array, 'n m N'], rank: int):
    n, m, N = J.shape
    W = jax.random.normal(get_random_key(), shape=(n, rank))
    V = jax.random.normal(get_random_key(), shape=(m, rank))
    H = jax.random.normal(get_random_key(), shape=(N, rank))
    R = jax.random.normal(get_random_key(), shape=(N, rank))
    return W, V, H, R

def CMTF_lstsq(X1, Y1, X2, Y2, lam: float):
    X = jnp.concatenate([X1, (lam * X2)], axis=0)
    Y = jnp.concatenate([Y1.T, (lam * Y2)], axis=0)
    return jnp.linalg.lstsq(X, Y)[0].T

def normalize_columns_V(W: Float[Array, 'n r'], V: Float[Array, 'm r']):
    r = W.shape[1]

    for i in range(r):
        colV = V[:, i]
        colW = W[:, i]
        norm = jnp.linalg.norm(colV) + 1e-12
        V = V.at[:, i].set(colV / norm)
        W = W.at[:, i].set(colW * norm)

    return W, V

@jaxtyped(typechecker=beartype)
def decoupling_CMTF_SSD(
    J: Float[Array, 'n m N'],
    Y: Float[Array, 'N n'],
    X: Float[Array, 'N m'],
    rank: int,
    lam: float = 0.1,
    max_iters: int = 100,
    tol: float = 1e-6,
    verbose: int = 0,
):
    log = make_log(verbose)

    W, V, H, R = init(J, rank)

    norm = jnp.linalg.norm(J)

    errors = []

    min_error = float('inf')

    for iteration in range(max_iters):
        W = CMTF_lstsq(X1=khatri_rao(H, V), Y1=unfold_kolda(J, 0), X2=R, Y2=Y, lam=lam)
        V = jnp.linalg.lstsq(khatri_rao(H, W), unfold_kolda(J, 1).T)[0].T
        W, V = normalize_columns_V(W, V)

        H = jnp.linalg.lstsq(khatri_rao(V, W), unfold_kolda(J, 2).T)[0].T
        R = jnp.linalg.lstsq(W, Y.T)[0].T

        H, R = projection(H, R, X @ V)

        factors = W, V, H
        error = relative_error(J, factors)

        if error < min_error:
            min_error = error
            best = (W.copy(), V.copy(), H.copy(), R.copy())

        if iteration > 0:
            diff = abs(error - errors[-1])
            log(f'iteration {iteration+1}: error = {error:.4f}, diff = {diff:.8f}')
            #if diff < tol * norm: log(f'stopping at iteration {iteration+1}'); break

        else: log(f'iter {iteration+1}: error = {error:.4f}')
        errors.append(error)

    log(f'returning best with error = {min_error:.4f}')
    return best
        
def projection(
    H: Array,
    R: Array,
    Z: Array,
):
    """Compute the smoothing spline projection for G and R."""

    assert H.shape[0] == Z.shape[0]
    assert tuple(H.shape) == tuple(R.shape)

    rank = H.shape[1]
    for j in range(rank):
        xj = Z[:, j].copy()

        idx = jnp.argsort(xj)
        inv = jnp.argsort(idx)

        xj_s = xj[idx]
        Gj_s = H[:, j].copy()[idx]
        Rj_s = R[:, j].copy()[idx]

        #if use_gj_prime:
            #gj_prime = make_smoothing_spline(xj_s, Gj_s)
            #H = H.at[:, j].set(gj_prime(X[:, j]))
            #R = R.at[:, j].set(gj_prime.antiderivative()(X[:, j]))
            ## must account for bias
        #else:
            #gj = make_smoothing_spline(xj_s, Rj_s)
            #H = H.at[:, j].set(gj.derivative()(X[:, j]))
            #R = R.at[:, j].set(gj(X[:, j]))

        try:
            gj = make_smoothing_spline(xj_s, Rj_s)

        except ValueError as e:
            print(e)
            print(xj_s)

        H = H.at[:, j].set(gj.derivative()(Z[:, j]))
        R = R.at[:, j].set(gj(Z[:, j]))

    return H, R
