from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def optimize_cem(
    cost_function: Callable[[ArrayLike], jnp.ndarray],
    init_mean: ArrayLike,
    init_var: ArrayLike,
    key: jnp.ndarray,
    n_iter: int,
    n_population: int,
    n_elite: int,
    lower_bound: ArrayLike,
    upper_bound: ArrayLike,
    epsilon: float = 0.001,
    alpha: float = 0.25,
    return_history: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Cross Entropy Method (CEM) optimizer.

    Parameters
    ----------
    cost_function
        A function for computing costs over a batch of candidate solutions.
    init_mean
        The mean of the initial candidate distribution.
    init_var
        The variance of the initial candidate distribution.
    key
        Random number generator key.
    n_iter
        The maximum number of iterations to perform during optimization.
    n_population
        The number of candidate solutions to be sampled at every iteration.
    n_elite
        The number of top solutions that will be used to obtain the distribution
        at the next iteration.
    lower_bound
        An array of lower bounds.
    upper_bound
        An array of upper bounds.
    epsilon, optional
        A minimum variance. If the maximum variance drops below epsilon,
        optimization is stopped.
    alpha, optional
        Controls how much of the previous mean and variance is used for the
        next iteration:
        next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly
        for variance.
    return_history, optional
        Return search history.

    Returns
    -------
    sol
        Solution.

    path, optional
        History of intermediate solutions, distribution means.

    samples, optional
        History of all samples.
    """
    ub = jnp.asarray(upper_bound)
    lb = jnp.asarray(lower_bound)

    if n_elite > n_population:
        raise ValueError(
            f"Number of elites {n_elite=} must be at most the population "
            f"size {n_population=}."
        )

    mean = jnp.asarray(init_mean)
    var = jnp.asarray(init_var)

    if return_history:
        path = []
        sample_history = []

    for t in range(n_iter):
        if jnp.max(var) <= epsilon:
            break

        key, step_key = jax.random.split(key, 2)
        mean, var, samples = step_cem(
            cost_function,
            mean,
            var,
            step_key,
            n_elite,
            n_population,
            lb,
            ub,
            alpha,
        )
        if return_history:
            path.append(jnp.copy(mean))
            sample_history.append(samples)

    if return_history:
        return mean, jnp.vstack(path), jnp.vstack(sample_history)
    else:
        return mean


def step_cem(
    cost_function: Callable[[ArrayLike], jnp.ndarray],
    mean: jnp.ndarray,
    var: jnp.ndarray,
    step_key: jnp.ndarray,
    n_elite: int,
    n_population: int,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    alpha: float,
):
    lb_dist = mean - lb
    ub_dist = ub - mean
    constrained_var = jnp.minimum(
        jnp.minimum((0.5 * lb_dist) ** 2, (0.5 * ub_dist) ** 2),
        var,
    )

    samples = (
        jax.random.truncated_normal(
            step_key, -2.0, 2.0, shape=(n_population, mean.shape[0])
        )
        * jnp.sqrt(constrained_var)[jnp.newaxis]
        + mean[jnp.newaxis]
    )
    costs = cost_function(samples)
    ranking = jnp.argsort(costs)
    elites = samples[ranking][:n_elite]
    mean = alpha * mean + (1.0 - alpha) * jnp.mean(elites, axis=0)
    var = alpha * var + (1.0 - alpha) * jnp.var(elites, axis=0)
    return mean, var, samples
