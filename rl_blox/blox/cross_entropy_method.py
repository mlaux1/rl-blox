from typing import Callable

import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def optimize_cem(
    fitness_function: Callable[[ArrayLike], jnp.ndarray],
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
    """Cross Entropy Method (CEM) optimizer (maximizer).

    Parameters
    ----------
    fitness_function
        A function for computing fitnesses of a batch of candidate solutions.
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
    mean = jnp.asarray(init_mean)
    var = jnp.asarray(init_var)
    ub = jnp.asarray(upper_bound)
    lb = jnp.asarray(lower_bound)

    if n_elite > n_population:
        raise ValueError(
            f"Number of elites {n_elite=} must be at most the population "
            f"size {n_population=}."
        )

    if return_history:
        path = []
        sample_history = []

    for t in range(n_iter):
        if jnp.max(var) <= epsilon:
            break

        key, step_key = jax.random.split(key, 2)
        samples = cem_sample(mean, var, step_key, n_population, lb, ub)
        f = fitness_function(samples)
        mean, var = cem_update(samples, f, mean, var, n_elite, alpha)

        if return_history:
            path.append(jnp.copy(mean))
            sample_history.append(samples)

    if return_history:
        return mean, jnp.vstack(path), jnp.vstack(sample_history)
    else:
        return mean


def cem_sample(
    mean: jnp.ndarray,
    var: jnp.ndarray,
    step_key: jnp.ndarray,
    n_population: int,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
) -> jnp.ndarray:
    r"""Sample from search distribution.

    Parameters
    ----------
    mean : array, shape (n_parameters,)
        Mean of search distribution.

    var : array, shape (n_parameters,)
        Variance per dimension of search distribution.

    step_key : array
        Random key for sampling.

    n_population : int
        Number of samples to draw.

    lb : array, (n_parameters,)
        Lower bound for sampling.

    ub : array, (n_parameters,)
        Upper bound for sampling.

    Returns
    -------
    samples : array, shape (n_population, n_parameters)
        Samples :math:`lb \leq x_i \leq ub` from search distribution.
    """
    chex.assert_equal_shape((mean, var))
    chex.assert_equal_shape((mean, lb))
    chex.assert_equal_shape((mean, ub))

    lb_dist = mean - lb
    ub_dist = ub - mean
    constrained_var = jnp.minimum(
        jnp.minimum((0.5 * lb_dist) ** 2, (0.5 * ub_dist) ** 2),
        var,
    )
    samples = (
        jax.random.truncated_normal(
            step_key, -2.0, 2.0, shape=(n_population,) + mean.shape
        )
        * jnp.sqrt(constrained_var)[jnp.newaxis]
        + mean[jnp.newaxis]
    )
    return samples


def cem_update(
    samples: jnp.ndarray,
    fitness: jnp.ndarray,
    mean: jnp.ndarray,
    var: jnp.ndarray,
    n_elite: int,
    alpha: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Update search distribution with cross entropy method (CEM).

    Based on the top k samples :math:`x_1, \ldots, x_k`, the search
    distribution is updated according to

    .. math::

        \mu_{t+1} &= \alpha \mu_t + (1-\alpha) \bar{x}\\
        \sigma^2_{t+1} &= \alpha \sigma_t^2
        + (1-\alpha) \frac{1}{k}\sum_i (x_i - \bar{x})^2

    with :math:`\bar{x} = \frac{1}{k}\sum_i x_i`.

    Parameters
    ----------
    samples : array, shape (n_population, n_parameters)
        Samples from current search distribution.

    fitness : array, shape (n_population,)
        Fitness values obtained for samples. Larger values are better.

    mean : array, shape (n_parameters,)
        Mean of current search distribution.

    var : array, shape (n_parameters,)
        Variance per dimension of current search distribution.

    n_elite : int
        Number of samples used for the update.

    alpha : float
        Weight of the old distribution in the update. (1 - alpha) is similar
        to a learning rate.

    Returns
    -------
    mean : array, shape (n_parameters,)
        Mean of new search distribution.

    var : array, shape (n_parameters,)
        Variance per dimension of new search distribution.
    """
    _, top_k = jax.lax.top_k(fitness, n_elite)
    elites = jnp.take(samples, top_k, axis=0)
    mean = alpha * mean + (1.0 - alpha) * jnp.mean(elites, axis=0)
    var = alpha * var + (1.0 - alpha) * jnp.var(elites, axis=0)
    return mean, var
