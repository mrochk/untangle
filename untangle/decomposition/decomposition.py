import tensorly as tl
import jax, jax.numpy as jnp
from jaxtyping import jaxtyped, Array, Float
from beartype import beartype
from tensorly.decomposition._cp import (
    error_calc, 
    cp_normalize, 
    initialize_cp,
    unfolding_dot_khatri_rao, 
)

def error_calc_(tensor, norm_tensor, weights, factors, mttkrp):
    return error_calc(tensor, norm_tensor, weights, factors, None, None, mttkrp)

def search_rank(tensor: Float[Array, 'n m N'], max_rank: int = 100, **args):
    for rank in range(1, max_rank+1):
        _, errors = cpd(tensor, rank, **args)
        last_error = errors[-1]
        if last_error < 0.01: return rank
    return 0

@jaxtyped(typechecker=beartype)
def cpd(
    tensor: Float[Array, 'n m N'],
    rank: int,
    n_iter_max: int = 100,
    tol: float = 1e-8,
    verbose: bool = False,
    random_state = None,
    normalize_factors: bool = False,
    linesearch: bool = False,
):
    '''
    From Tensorly: https://github.com/tensorly/tensorly/blob/main/tensorly/decomposition/_cp.py.
    Simplified version, works only for 3d tensors.
    '''

    log = lambda *args: print(*args) if verbose else None

    if linesearch:
        acc_pow = 2.0 # Extrapolate to the iteration^(1/acc_pow) ahead
        acc_fail = 0  # How many times acceleration have failed
        max_fail = 4  # Increase acc_pow with one after max_fail failure

    weights, factors = initialize_cp(
        tensor, rank, 
        init='random', 
        random_state=random_state, 
        normalize_factors=normalize_factors,
    )

    errors = []
    norm_tensor = tl.norm(tensor, 2)

    for iteration in range(n_iter_max):

        if linesearch and iteration % 2 == 0:
            factors_last = [jnp.copy(f) for f in factors]
            weights_last = jnp.copy(weights)

        for mode in range(3):

            pseudo_inverse = jnp.ones((rank, rank), **tl.context(tensor))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse = pseudo_inverse * tl.dot(
                        tl.conj(tl.transpose(factor)), factor
                    )

            pseudo_inverse = (
                tl.reshape(weights, (-1, 1))
                * pseudo_inverse
                * tl.reshape(weights, (1, -1))
            )

            mttkrp = unfolding_dot_khatri_rao(tensor, (weights, factors), mode)

            factor = tl.transpose(
                tl.solve(tl.conj(tl.transpose(pseudo_inverse)), tl.transpose(mttkrp))
            )
            factors[mode] = factor

        # Will we be performing a line search iteration
        if linesearch and iteration % 2 == 0 and iteration > 5: line_iter = True
        else: line_iter = False

        # Calculate the current unnormalized error if we need it
        if not line_iter:
            unnorml_rec_error, tensor, norm_tensor = error_calc_(
                tensor, norm_tensor, weights, factors, mttkrp
            )

        # Start line search if requested.
        if line_iter:
            jump = iteration ** (1.0 / acc_pow)

            new_weights = weights_last + (weights - weights_last) * jump
            new_factors = [
                factors_last[ii] + (factors[ii] - factors_last[ii]) * jump
                for ii in range(tl.ndim(tensor))
            ]

            new_rec_error, new_tensor, new_norm_tensor = error_calc_(
                tensor, norm_tensor, new_weights, new_factors, None,
            )

            if (new_rec_error / new_norm_tensor) < errors[-1]:
                factors, weights = new_factors, new_weights
                tensor, norm_tensor = new_tensor, new_norm_tensor
                unnorml_rec_error = new_rec_error
                acc_fail = 0

                log(f'Accepted line search jump of {jump}.')
            else:
                unnorml_rec_error, tensor, norm_tensor = error_calc_(
                    tensor, norm_tensor, weights, factors, mttkrp
                )
                acc_fail += 1

                log(f'Line search failed for jump of {jump}.')

                if acc_fail == max_fail:
                    acc_pow += 1.0
                    acc_fail = 0

                    log('Reducing acceleration.')

        if not line_iter:
            rec_error = unnorml_rec_error / norm_tensor
            errors.append(rec_error)

        if iteration >= 1:
            rec_error_decrease = errors[-2] - errors[-1]

            log(f'iteration {iteration}, reconstruction error: {rec_error}, decrease = {rec_error_decrease}, unnormalized = {unnorml_rec_error}')

            stop_flag = tl.abs(rec_error_decrease) < tol

            if stop_flag:
                log(f'CPD converged after {iteration} iterations.')
                break

        else: log(f'reconstruction error={errors[-1]}')

        if normalize_factors:
            weights, factors = cp_normalize((weights, factors))

    W, V, H = factors
    W = jnp.array(W)
    V = jnp.array(V)
    H = jnp.array(H)

    return (weights, (W, V, H)), errors

@jaxtyped(typechecker=beartype)
def cpd_polynomial_constraint(
    tensor: Float[Array, 'n m N'],
    rank: int,
    n_iter_max: int = 100,
    tol: float = 1e-8,
    verbose: bool = False,
    random_state = None,
    normalize_factors: bool = False,
    linesearch: bool = False,
):
    pass