import jax.numpy as jnp

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from untangle.decomposition import cpd
from untangle.utils import make_log

@jaxtyped(typechecker=beartype)
def search_rank(
    tensor: Float[Array, 'n m N'],
    max_rank: int = 20,
    eps: float = 0.01,
    early_stop: bool = True,
    plot: bool = False,
    verbose: bool = False,
    **cpd_kwargs,
) -> int:

    log = make_log(int(verbose))
    cpd_errors = []
    returned_rank = None

    log('Searching CPD rank:')
    for rank in range(1, max_rank + 1):

        log(f'\n- computing CPD for rank {rank}...')
        error = cpd(tensor, rank, **cpd_kwargs)[2][-1]
        cpd_errors.append(error)
        log(f'- error = {error:.5f}...')

        if error <= eps:
            if returned_rank is None: returned_rank = rank
            if early_stop: log(f'- error < eps ({eps}) for rank {rank}, breaking loop'); break

    if plot:
        plt.figure(figsize=(7, 4))
        plt.plot(jnp.arange(1, max_rank + 1 if not early_stop else returned_rank + 1), cpd_errors, color='black')
        plt.title('CPD Rank Search')
        plt.ylabel('CPD Reconstruction Error')
        plt.xlabel('Rank')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.axvline(x=returned_rank, color='red')
        plt.savefig('rank_errors.png')
        plt.close()
        log('plot saved at <rank_errors.png>')

    return returned_rank
