# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import dataclasses
import math
import warnings
from collections import namedtuple

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct
from jax.typing import ArrayLike
from scipy.spatial.distance import pdist
from tqdm.rich import trange

from ..logging.logger import LoggerBase


@struct.dataclass
class CMAESConfig:
    """Configuration of CMA-ES."""

    active: bool
    """Active CMA-ES with negative weighted covariance matrix update."""
    bounds: jnp.ndarray | None
    """Upper and lower bounds for each parameter."""
    maximize: bool
    """Maximize return or minimize cost?"""
    min_variance: float
    """Minimum variance before restart."""
    min_fitness_dist: float
    """Minimum distance between fitness values before restart."""
    max_condition: float
    """Maximum condition of covariance matrix."""
    n_samples_per_update: int
    """Number of samples from the search distribution (population size)."""
    n_params: int
    """Dimensionality of the search space."""
    mu: int
    """Number of samples to update the search distribution."""
    weights: jnp.ndarray
    """Sample weights for mean recombination."""
    mueff: float
    """Effective sample size for update."""
    cc: float
    """Time constant for cumulation of the covariance."""
    cs: float
    """Time constant for cumulation for sigma control."""
    c1: float
    """Learning rate for rank-one update."""
    cmu: float
    """Learning rate for rank-mu update."""
    damps: float
    """Damping for sigma."""
    ps_update_weight: float
    """Update weight for evolution path of sigma."""
    hsig_threshold: float
    """Is used to update the covariance evolution path."""
    eigen_update_freq: int
    """Update frequency for inverse square root of covariance matrix."""
    alpha_old: float
    """Constant for rank-mu update in aCMA-ES."""
    neg_cmu: float
    """Learning rate for active rank-mu update."""

    @classmethod
    def create(
        cls,
        active: bool,
        bounds: jnp.ndarray | None,
        maximize: bool,
        min_variance: float,
        min_fitness_dist: float,
        max_condition: float,
        n_params: int,
        n_samples_per_update: int,
    ):
        if bounds is not None:
            bounds = jnp.asarray(bounds)

        if n_samples_per_update is None:
            n_samples_per_update = 4 + int(3 * math.log(n_params))

        mu = n_samples_per_update / 2.0
        weights = math.log(mu + 0.5) - jnp.log1p(jnp.arange(int(mu)))
        mu = int(mu)
        weights = weights / jnp.sum(weights)
        mueff = 1.0 / float(jnp.sum(weights**2))

        cc = (4 + mueff / n_params) / (n_params + 4 + 2 * mueff / n_params)
        cs = (mueff + 2) / (n_params + mueff + 5)
        c1 = 2 / ((n_params + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * mueff - 2 + 1.0 / mueff) / (
            (n_params + 2) ** 2 + mueff
        )
        damps = (
            1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n_params + 1)) - 1) + cs
        )

        ps_update_weight = math.sqrt(cs * (2 - cs) * mueff)
        hsig_threshold = 2 + 4.0 / (n_params + 1)
        eigen_update_freq = int(
            n_samples_per_update / ((c1 + cmu) * n_params * 10)
        )

        alpha_old = 0.5
        neg_cmu = (
            (1.0 - cmu) * 0.25 * mueff / ((n_params + 2) ** 1.5 + 2.0 * mueff)
        )

        return cls(
            active=active,
            bounds=bounds,
            maximize=maximize,
            min_variance=min_variance,
            min_fitness_dist=min_fitness_dist,
            max_condition=max_condition,
            n_samples_per_update=n_samples_per_update,
            n_params=n_params,
            mu=mu,
            weights=weights,
            mueff=mueff,
            cc=cc,
            cs=cs,
            c1=c1,
            cmu=cmu,
            damps=damps,
            ps_update_weight=ps_update_weight,
            hsig_threshold=hsig_threshold,
            eigen_update_freq=eigen_update_freq,
            alpha_old=alpha_old,
            neg_cmu=neg_cmu,
        )


@dataclasses.dataclass(frozen=False)
class CMAESState:
    """State of CMA-ES."""

    key: jnp.ndarray
    """Key for PRNG."""
    it: int
    """Current iteration."""
    eigen_decomp_updated: int
    """Iteration when inverse square root of covariance matrix was updated."""
    mean: jnp.ndarray
    """Mean of search distribution."""
    last_mean: jnp.ndarray
    """Last mean of search distribution."""
    var: float
    """Variance of search distribution."""
    cov: jnp.ndarray
    """Covariance of search distribution."""
    invsqrtC: jnp.ndarray
    """Inverse square root of covariance."""
    best_fitness: float
    """Best fitness obtained so far."""
    best_fitness_it: int
    """Iteration of best obtained fitness."""
    best_params: jnp.ndarray
    """Best parameters obtained so far."""
    pc: jnp.ndarray
    """Evolution path for covariance."""
    ps: jnp.ndarray
    """Evolution path for sigma"""

    @classmethod
    def create(
        cls,
        key: jnp.ndarray | None,
        initial_params: jnp.ndarray,
        variance: float,
        covariance: jnp.ndarray | None,
    ):
        if key is None:
            key = jax.random.key(0)

        mean = jnp.asarray(initial_params).copy()

        if covariance is None:
            cov = jnp.eye(len(mean))
        else:
            cov = jnp.asarray(covariance).copy()
            if cov.ndim == 1:
                cov = jnp.diag(cov)

        pc = jnp.zeros(len(mean))
        ps = jnp.zeros(len(mean))

        return cls(
            key=key,
            it=0,
            eigen_decomp_updated=0,
            mean=mean,
            last_mean=jnp.copy(mean),
            var=variance,
            cov=cov,
            invsqrtC=inv_sqrt(cov)[0],
            best_fitness=jnp.inf,
            best_fitness_it=0,
            best_params=jnp.copy(mean),
            pc=pc,
            ps=ps,
        )


@dataclasses.dataclass(frozen=False)
class Population:
    """Sampled population."""

    samples: jnp.ndarray
    fitness: list[float]

    @classmethod
    def create(cls, samples):
        return cls(samples=samples, fitness=[np.inf] * len(samples))


def inv_sqrt(cov: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute inverse square root of a covariance matrix."""
    cov = jnp.triu(cov) + jnp.triu(cov, 1).T
    D, B = jnp.linalg.eigh(cov)
    # avoid numerical problems
    D = jnp.maximum(D, jnp.finfo(float).eps)
    D = jnp.sqrt(D)
    return B.dot(jnp.diag(1.0 / D)).dot(B.T), B, D


def sample_population(config: CMAESConfig, state: CMAESState) -> jnp.ndarray:
    """Sample population from search distribution.

    Parameters
    ----------
    config : CMAESConfig
        CMA-ES configuration.

    state : CMAESState
        State of CMA-ES.

    Returns
    -------
    samples : array, shape (n_samples_per_update, n_params)
        Samples from search distribution of CMA-ES.
    """
    state.key, sampling_key = jax.random.split(state.key, 2)
    samples = jax.random.multivariate_normal(
        sampling_key,
        state.mean,
        state.var * state.cov,
        (config.n_samples_per_update,),
    )
    if config.bounds is not None:
        samples = jnp.clip(samples, config.bounds[:, 0], config.bounds[:, 1])
    return samples


def get_next_parameters(
    config: CMAESConfig, state: CMAESState, population: Population
) -> jnp.ndarray:
    """Get next individual/parameter vector for evaluation.

    Parameters
    ----------
    config : CMAESConfig
        CMA-ES configuration.

    state : CMAESState
        State of CMA-ES.

    population : Population
        Current population.

    Returns
    -------
    params : array, shape (n_params,)
        Parameter vector
    """
    k = state.it % config.n_samples_per_update
    return population.samples[k]


def set_evaluation_feedback(
    config: CMAESConfig,
    state: CMAESState,
    population: Population,
    feedback: ArrayLike,
):
    """Set feedbacks for the parameter vector.

    Parameters
    ----------
    config : CMAESConfig
        CMA-ES configuration.

    state : CMAESState
        State of CMA-ES.

    population : Population
        Current population.

    feedback : array-like
        Feedbacks for each step or for the episode, depends on the problem.
    """
    k = state.it % config.n_samples_per_update
    fitness_k = float(jnp.sum(feedback))
    if config.maximize:
        fitness_k = -fitness_k

    population.fitness[k] = fitness_k

    if fitness_k <= state.best_fitness:
        state.best_fitness = fitness_k
        state.best_fitness_it = state.it
        state.best_params = population.samples[k]

    state.it += 1


def update_search_distribution(
    config: CMAESConfig, state: CMAESState, population: Population
):
    """Update search distribution of CMA-ES.

    Parameters
    ----------
    config : CMAESConfig
        CMA-ES configuration.

    state : CMAESState
        State of CMA-ES.

    population : Population
        Current population.
    """
    samples = population.samples
    fitness = jnp.asarray(population.fitness)

    # 1) Update sample distribution mean

    state.last_mean = state.mean
    ranking = jnp.argsort(fitness, axis=0)
    update_samples = samples[ranking[: config.mu]]
    state.mean = jnp.sum(
        config.weights[:, jnp.newaxis] * update_samples, axis=0
    )

    mean_diff = state.mean - state.last_mean
    sigma = jnp.sqrt(state.var)

    # 2) Cumulation: update evolution paths

    # Isotropic (step size) evolution path
    state.ps += (
        -config.cs * state.ps
        + config.ps_update_weight / sigma * state.invsqrtC.dot(mean_diff)
    )
    # Anisotropic (covariance) evolution path
    ps_norm_2 = jnp.linalg.norm(state.ps) ** 2  # Temporary constant
    generation = state.it / config.n_samples_per_update
    hsig = int(
        ps_norm_2
        / config.n_params
        / jnp.sqrt(1 - (1 - config.cs) ** (2 * generation))
        < config.hsig_threshold
    )
    state.pc *= 1 - config.cc
    state.pc += (
        hsig
        * jnp.sqrt(config.cc * (2 - config.cc) * config.mueff)
        * mean_diff
        / sigma
    )

    # 3) Update sample distribution covariance

    # Rank-1 update
    rank_one_update = jnp.outer(state.pc, state.pc)

    # Rank-mu update
    noise = (update_samples - state.last_mean) / sigma
    rank_mu_update = noise.T.dot(jnp.diag(config.weights)).dot(noise)

    # Correct variance loss by hsig
    c1a = config.c1 * (1 - (1 - hsig) * config.cc * (2.0 - config.cc))

    if config.active:
        neg_update = samples[ranking[::-1][: config.mu]]
        neg_update -= state.last_mean
        neg_update /= sigma
        neg_rank_mu_update = neg_update.T.dot(jnp.diag(config.weights)).dot(
            neg_update
        )

        state.cov *= 1.0 - c1a - config.cmu + config.neg_cmu * config.alpha_old
        state.cov += rank_one_update * config.c1
        state.cov += rank_mu_update * (
            config.cmu + config.neg_cmu * (1.0 - config.alpha_old)
        )
        state.cov -= neg_rank_mu_update * config.neg_cmu
    else:
        state.cov *= 1.0 - c1a - config.cmu
        state.cov += rank_one_update * config.c1
        state.cov += rank_mu_update * config.cmu

    # NOTE here is a bug: it should be cs / (2 * damps), however, that
    #      breaks unit tests and does not improve results
    log_step_size_update = (config.cs / config.damps) * (
        ps_norm_2 / config.n_params - 1
    )
    # NOTE some implementations of CMA-ES use the denominator
    # np.sqrt(n_params) * (1.0 - 1.0 / (4 * n_params) +
    #                           1.0 / (21 * n_params ** 2))
    # instead of n_params, in this case cs / damps is correct
    # Adapt step size with factor <= exp(0.6)
    state.var = state.var * jnp.exp(min((0.6, log_step_size_update))) ** 2

    if state.it - state.eigen_decomp_updated > config.eigen_update_freq:
        state.invsqrtC = inv_sqrt(state.cov)[0]
        state.eigen_decomp_updated = state.it


def is_cmaes_finished(
    config: CMAESConfig,
    state: CMAESState,
    population: Population,
    logger: LoggerBase | None,
) -> bool:
    """Check if the optimization is finished.

    Parameters
    ----------
    config : CMAESConfig
        Configuration of CMA-ES.

    state : CMAESState
        State of CMA-ES.

    population : Population
        Current population.

    Returns
    -------
    finished : bool
        Is the learning of a behavior finished?
    """
    if state.it <= config.n_samples_per_update:
        return False

    fitness = jnp.asarray(population.fitness)

    total_variance = jnp.max(jnp.diag(state.cov)) * state.var
    max_fitness_dist = jnp.max(pdist(fitness[:, jnp.newaxis]))

    cov_diag = jnp.diag(state.cov)
    condition_number = jnp.max(cov_diag)
    min_cov_diag = jnp.min(cov_diag)
    if min_cov_diag > 0.0:
        condition_number /= min_cov_diag

    if logger is not None:
        logger.record_stat("total_variance", total_variance)
        logger.record_stat("max_fitness_dist", max_fitness_dist)
        logger.record_stat("condition_number", condition_number)

    if not jnp.all(jnp.isfinite(fitness)):
        return True

    # Check for invalid values
    if not (
        jnp.all(jnp.isfinite(state.invsqrtC))
        and jnp.all(jnp.isfinite(state.cov))
        and jnp.all(jnp.isfinite(state.mean))
        and jnp.isfinite(state.var)
    ):
        warnings.warn("[CMA-ES] Stopping: infs or nans", stacklevel=2)
        return True

    if (
        config.min_variance is not None
        and total_variance <= config.min_variance
    ):
        warnings.warn(
            f"[CMA-ES] Stopping: {state.var} < min_variance", stacklevel=2
        )
        return True

    if max_fitness_dist < config.min_fitness_dist:
        warnings.warn(
            f"[CMA-ES] Stopping: {max_fitness_dist} < min_fitness_dist",
            stacklevel=2,
        )
        return True

    if config.max_condition is not None and jnp.max(
        cov_diag
    ) > config.max_condition * jnp.min(cov_diag):
        warnings.warn(
            f"[CMA-ES] Stopping: "
            f"{jnp.max(cov_diag)} / {jnp.min(cov_diag)} > max_condition",
            stacklevel=2,
        )
        return True

    return False


@nnx.jit
def flat_params(net: nnx.Module):
    """Return parameters of a neural net as flat parameter vector.

    Only variables of the type ``nnx.Param`` will be extracted.

    Parameters
    ----------
    net : nnx.Module
        Neural net.

    Returns
    -------
    params : jnp.ndarray, shape (n_params,)
        Flat parameter vector.
    """
    state = nnx.state(net, nnx.Param)
    leaves = jax.tree_util.tree_leaves(state)
    flat_leaves = list(map(lambda x: x.ravel(), leaves))
    return jnp.concatenate(flat_leaves, axis=0)


@nnx.jit
def set_params(net: nnx.Module, params: jnp.ndarray):
    """Set parameters of a neural net (inplace) from a flat parameter vector.

    Only variables of the type ``nnx.Param`` will be updated.

    Parameters
    ----------
    net : nnx.Module
        Neural net.

    params : jnp.ndarray, shape (n_params,)
        Flat parameter vector.
    """
    state = nnx.state(net, nnx.Param)
    leaves = jax.tree_util.tree_leaves(state)
    treedef = jax.tree_util.tree_structure(state)
    n_params_set = 0
    new_leaves = []
    for leaf in leaves:
        n_params_leaf = np.prod(leaf.shape)
        new_leaf = params[n_params_set : n_params_set + n_params_leaf].reshape(
            leaf.shape
        )
        new_leaves.append(new_leaf)
        n_params_set += n_params_leaf
    state = jax.tree_util.tree_unflatten(treedef, new_leaves)
    nnx.update(net, state)


def train_cmaes(
    env: gymnasium.Env,
    policy: nnx.Module,
    total_episodes: int,
    seed: int = 0,
    variance: float = 1.0,
    covariance: ArrayLike | None = None,
    n_samples_per_update: int | None = None,
    active: bool = False,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[nnx.Module, float, bool]:
    """Train policy using Covariance Matrix Adaptation Evolution Strategy.

    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a black-box
    optimizer. We learn after each episode by accumulating the rewards and
    using this return as a fitness value that we maximize with CMA-ES by
    changing the parameters of a deterministic policy network that maps
    observations to actions. Since CMA-ES learns only after
    `n_samples_per_update` episodes, it is supposed to learn slower than
    reinforcement learning algorithms. However, it is a robust algorithm that
    provides a good baseline to compare against.

    See `Wikipedia <http://en.wikipedia.org/wiki/CMA-ES>`_ for details about the
    optimizer. CMA-ES [1]_ is considered to be useful for

    * non-convex,
    * non-separable,
    * ill-conditioned,
    * or noisy

    objective functions. However, in some cases CMA-ES will be outperformed
    by other methods:

    * if the search space dimension is very small (e.g. less than 5),
      downhill simplex or surrogate-assisted methods will be better
    * easy functions (separable, nearly quadratic, etc.) will usually be
      solved faster by NEWUOA
    * multimodal objective functions require restart strategies

    Parameters
    ----------
    env : gymnasium.Env
        Environment.

    policy : nnx.Module
        The policy network should map observations to actions.

    total_episodes : int
        Total number of episodes.

    seed : int
        Random seed.

    variance : float, optional (default: 1.0)
        Initial exploration variance.

    covariance : array-like, optional (default: None)
        Either a diagonal (with shape (n_params,)) or a full covariance matrix
        (with shape (n_params, n_params)). A full covariance can contain
        information about the correlation of variables.

    n_samples_per_update : integer, optional (default: 4+int(3*log(n_params)))
        Number of roll-outs that are required for a parameter update.

    active : bool, optional (default: False)
        Active CMA-ES (aCMA-ES) with negative weighted covariance matrix
        update

    logger : LoggerBase, optional
        Logger for experiment tracking.

    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.

    Returns
    -------
    policy : nnx.Module
        Trained policy network.

    best_fitness : float
        Best fitness observed in any episode.

    stopped : bool
        If the optimizer was stopped before all episodes were run.

    References
    ----------
    .. [1] Hansen, N.; Ostermeier, A. Completely Derandomized Self-Adaptation
        in Evolution Strategies. In: Evolutionary Computation, 9(2), pp.
        159-195. https://www.lri.fr/~hansen/cmaartic.pdf
    """
    init_params = flat_params(policy)
    key = jax.random.key(seed)

    config = CMAESConfig.create(
        active=active,
        bounds=None,
        maximize=True,
        min_variance=2 * jnp.finfo(jnp.float32).eps ** 2,
        min_fitness_dist=2 * jnp.finfo(jnp.float32).eps,
        max_condition=1e7,
        n_params=len(init_params),
        n_samples_per_update=n_samples_per_update,
    )
    state = CMAESState.create(
        key=key,
        initial_params=init_params,
        covariance=covariance,
        variance=variance,
    )
    population = Population.create(
        samples=sample_population(config, state),
    )

    @nnx.jit
    def policy_action(policy, observation):
        return policy(observation)

    stopped = False

    obs, _ = env.reset(seed=seed)
    step_counter = 0
    if logger is not None:
        logger.start_new_episode()
    for _ in trange(total_episodes, disable=not progress_bar):
        set_params(policy, get_next_parameters(config, state, population))
        ret = 0.0
        done = False
        while not done:
            action = np.asarray(policy_action(policy, jnp.asarray(obs)))

            next_obs, reward, termination, truncation, info = env.step(action)
            step_counter += 1
            obs = next_obs
            ret += reward
            done = termination or truncation

        obs, _ = env.reset()
        set_evaluation_feedback(config, state, population, ret)

        if state.it % config.n_samples_per_update == 0:
            stopped = is_cmaes_finished(config, state, population, logger)
            if stopped:
                break
            update_search_distribution(config, state, population)
            population = Population.create(
                samples=sample_population(config, state)
            )

        if logger is not None:
            logger.stop_episode(step_counter)
            logger.start_new_episode()
            logger.record_stat("return", ret)
            logger.record_epoch("policy", policy)

        step_counter = 0

    set_params(policy, state.mean)
    best_fitness = -state.best_fitness
    return namedtuple("CMAESResult", ["policy", "best_fitness", "stopped"])(
        policy, best_fitness, stopped
    )
