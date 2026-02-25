import unittest
import jax, jax.numpy as jnp

from untangle.algorithm import decoupling_basic
from untangle.algorithm.basic import inference
from untangle.decomposition import search_rank
from untangle.utils import get_random_key, collect_information

class TestDecouplingBasic(unittest.TestCase):
    def setUp(self): print()

    def test_simple_function(self):
        '''Function given in example (4) of the paper.'''

        n = 3; m = 3; N = 10; rank = 4 # we know that this is a rank four

        def f(u):
            u1, u2, u3 = u
            return jnp.array([
                -4 * u1**2 + 8 * u1 * u3 + 6 * u1 - 3 * u3**2 - 8 * u3 - 6,
                2 * u1**2 - 4 * u1 * u3 - 3 * u1 + u2**3 + 6 * u2**2 * u3 + 12 * u2 * u3**2 - u2 + 8 * u3**3 + 2 * u3**2 + u3 + 3,
                -2 * u1**2 + 4 * u1 * u3 + 4 * u1 - 2 * u3**2 - 3 * u3 - u2 - 8,
            ])

        X, Y, J = collect_information(f, N, n)
        self.assertTrue(tuple(X.shape) == (N, m))
        self.assertTrue(tuple(Y.shape) == (N, n))
        self.assertTrue(tuple(J.shape) == (n, m, N))

        W, V, g = decoupling_basic(X, Y, J, rank=rank, degree=3, linesearch=True, n_iter_max=1000, verbose=1, normalize_factors=True)
        decoupling = inference(W, V, g)

        x = jax.random.uniform(get_random_key(), shape=m)
        truth, decoupled = f(x), decoupling(x)

        error = jnp.linalg.norm(truth - decoupled) / jnp.linalg.norm(truth)
        self.assertLess(error, 0.1)

    def test_simple_function2(self):
        n = 5; m = 3; N = 20

        def f(u):
            u1, u2, u3 = u
            return jnp.array([
                -4 * u1**2 + 8 * u1 * u3 + 6 * u1 - 3 * u3**2 - 8 * u3 - 6,
                2 * u1**2 - 4 * u1 * u3 - 3 * u1 + u2**3 + 6 * u2**2 * u3 + 12 * u2 * u3**2 - u2 + 8 * u3**3 + 2 * u3**2 + u3 + 3,
                -2 * u1**2 + 4 * u1 * u3 + 4 * u1 - 2 * u3**2 - 3 * u3 - u2 - 8,
                6 * u1 * u2**2 - 2 * u3**2 - 10,
                u3**3 - 3*u2,
            ])

        X, Y, J = collect_information(f, N, m)
        self.assertTrue(tuple(X.shape) == (N, m))
        self.assertTrue(tuple(Y.shape) == (N, n))
        self.assertTrue(tuple(J.shape) == (n, m, N))

        rank, _ = search_rank(J, linesearch=True, tol=0.01)

        W, V, g = decoupling_basic(X, Y, J, rank=rank, degree=3, linesearch=True, verbose=1)
        decoupling = inference(W, V, g)

        x = jax.random.uniform(get_random_key(), shape=m)
        truth = f(x)
        decoupled = decoupling(x)

        error = jnp.linalg.norm(truth - decoupled) / jnp.linalg.norm(truth)
        self.assertLess(error,  0.1)
