import jax
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

from jaxtyping import jaxtyped, Float, Array
from beartype.typing import Optional, Tuple
from beartype import beartype

from untangle.utils import get_random_key, make_log
from untangle.decomposition import cpd, cpd_constrained

@jaxtyped(typechecker=beartype)
def run_many_cpd(
    tensor: Float[Array, 'n m N'], 
    rank: int, 
    n: int = mp.cpu_count(), 
    random_state: Optional[Array] = None,
    verbose: int = 0,
) -> Tuple[Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']], Float[Array, 'r']]:

    '''Runs many CP decompositions in parallel and returns the one with the lowest error.'''

    log = make_log(verbose)

    keys = jax.random.split(get_random_key() if random_state is None else random_state, num=n)

    def run_once(i):
        factors, weights, errors = cpd(tensor, rank, verbose=verbose > 1, random_state=keys[i])
        log(f'run {i}: {errors[-1]:.5f} ({len(errors)} iters)')
        return factors, weights, errors[-1]

    results = []

    with ThreadPoolExecutor() as ex:
        futures = [ex.submit(run_once, i) for i in range(n)]

        with tqdm(total=n, desc='Computing CPD') as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    factors, weights, best_error = min(results, key=lambda x: x[2])
    log(f'best error = {best_error:.5f}')

    return factors, weights

@jaxtyped(typechecker=beartype)
def run_many_cpd_constrained(
    tensor: Float[Array, 'n m N'], 
    X: Float[Array, 'N m'],
    rank: int, 
    degree: int,
    n: int = mp.cpu_count(), 
    random_state: Optional[Array] = None,
    verbose: int = 0,
) -> Tuple:

    '''Runs many CP decompositions in parallel and returns the one with the lowest error.'''

    log = make_log(verbose)

    keys = jax.random.split(get_random_key() if random_state is None else random_state, num=n)

    def run_once(i):
        factors, dcoefs, errors = cpd_constrained(
            tensor, X, rank, degree, verbose=(verbose > 1), random_state=keys[i],
        )
        log(f'run {i}: {errors[-1]:.5f} ({len(errors)} iters)')
        return factors, dcoefs, errors[-1]

    results = []

    with ThreadPoolExecutor() as ex:
        futures = [ex.submit(run_once, i) for i in range(n)]

        with tqdm(total=n, desc='Computing CPD with Polynomial Constraint') as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    factors, dcoefs, best_error = min(results, key=lambda x: x[2])
    log(f'best error = {best_error:.5f}')

    return factors, dcoefs
