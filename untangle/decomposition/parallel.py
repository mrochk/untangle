import jax
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

from untangle.utils import get_random_key
from .cpd import cpd

def run_many_cpd(tensor, rank, n: int = mp.cpu_count(), verbose: int = 0):
    '''Runs many CP decompositions in parallel and return the one with the lowest error.'''

    def run_once(i):
        factors, weights, errors = cpd(tensor, rank, verbose=verbose > 1, random_state=get_random_key())
        if verbose > 0: print(f'run {i}: {errors[-1]:.5f} ({len(errors)} iters)', flush=True)
        return factors, weights, errors[-1]

    with ThreadPoolExecutor() as ex: results = list(ex.map(run_once, range(n)))

    factors, weights, best_error = min(results, key=lambda x: x[2])
    if verbose > 0: print(f'best error = {best_error:.5f}', flush=True)

    return factors, weights 
