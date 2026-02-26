from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

from untangle.utils import get_random_key, make_log
from .cpd import cpd

def run_many_cpd(tensor, rank, n: int = mp.cpu_count(), verbose: int = 0):
    '''Runs many CP decompositions in parallel and returns the one with the lowest error.'''

    log = make_log(verbose)

    def run_once(i):
        factors, weights, errors = cpd(tensor, rank, verbose=verbose > 1, random_state=get_random_key())
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
