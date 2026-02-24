import unittest
import jax, jax.numpy as jnp

from untangle.algorithm import decoupling_basic
from untangle.algorithm.basic import inference
from untangle.utils import get_random_key, collect_information
from untangle.decomposition import cpd, search_rank

class TestDecouplingBasic(unittest.TestCase):

    def test_simple_function(self):
        '''Function given in example (4) of the paper.'''

        n = 3; m = 3; N = 10; rank = 4 # we know that this is a rank four

        f1 = lambda u1, u2, u3: -4 * u1**2 + 8 * u1 * u3 + 6 * u1 - 3 * u3**2 - 8 * u3 - 6
        f2 = lambda u1, u2, u3: 2 * u1**2 - 4 * u1 * u3 - 3 * u1 + u2**3 + 6 * u2**2 * u3 + 12 * u2 * u3**2 - u2 + 8 * u3**3 + 2 * u3**2 + u3 + 3
        f3 = lambda u1, u2, u3: -2 * u1**2 + 4 * u1 * u3 + 4 * u1 - 2 * u3**2 - 3 * u3 - u2 - 8
        f  = lambda u: jnp.array([f1(*u), f2(*u), f3(*u)])

        X, Y, J = collect_information(f, N, n)
        self.assertTrue(tuple(X.shape) == (N, m))
        self.assertTrue(tuple(Y.shape) == (N, n))
        self.assertTrue(tuple(J.shape) == (n, m, N))

        errors = []

        for i in range(5):
            W, V, g = decoupling_basic(X, Y, J, rank=rank, degree=3, linesearch=True, n_iter_max=1000)
            decoupling = inference(W, V, g)

            x = jax.random.uniform(get_random_key(), shape=m)
            truth = f(x)
            decoupled = decoupling(x)

            error = jnp.linalg.norm(truth - decoupled) / jnp.linalg.norm(truth)
            errors.append(error)

        self.assertTrue(any([e < 0.1 for e in errors]))

    def test_simple_function2(self):
        n = 5; m = 3; N = 20

        f1 = lambda u1, u2, u3: -4 * u1**2 + 8 * u1 * u3 + 6 * u1 - 3 * u3**2 - 8 * u3 - 6
        f2 = lambda u1, u2, u3: 2 * u1**2 - 4 * u1 * u3 - 3 * u1 + u2**3 + 6 * u2**2 * u3 + 12 * u2 * u3**2 - u2 + 8 * u3**3 + 2 * u3**2 + u3 + 3
        f3 = lambda u1, u2, u3: -2 * u1**2 + 4 * u1 * u3 + 4 * u1 - 2 * u3**2 - 3 * u3 - u2 - 8
        f4 = lambda u1, u2, u3: 6 * u1 * u2**2 - 2 * u3**2 - 10
        f5 = lambda u1, u2, u3: u3**3 - 3*u2

        def f(u):
            u1, u2, u3 = u
            return jnp.array([f1(u1, u2, u3), f2(u1, u2, u3), f3(u1, u2, u3), f4(u1, u2, u3), f5(u1, u2, u3)])

        X, Y, J = collect_information(f, N, m)
        self.assertTrue(tuple(X.shape) == (N, m))
        self.assertTrue(tuple(Y.shape) == (N, n))
        self.assertTrue(tuple(J.shape) == (n, m, N))

        errors = []
        for _ in range(3):

            rank = search_rank(J, linesearch=True)
            self.assertGreater(rank, 0)

            W, V, g = decoupling_basic(X, Y, J, rank=rank, degree=3, linesearch=True)
            decoupling = inference(W, V, g)

            x = jax.random.uniform(get_random_key(), shape=m)
            truth = f(x)
            decoupled = decoupling(x)

            error = jnp.linalg.norm(truth - decoupled) / jnp.linalg.norm(truth)
            errors.append(error)

        self.assertTrue(any([e <= 0.1 for e in errors]))
