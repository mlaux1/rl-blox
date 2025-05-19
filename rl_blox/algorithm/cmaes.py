# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import math

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from flax import nnx
from jax.typing import ArrayLike
from scipy.spatial.distance import pdist

from ..logging.logger import LoggerBase


def inv_sqrt(cov):
    """Compute inverse square root of a covariance matrix."""
    cov = jnp.triu(cov) + jnp.triu(cov, 1).T
    D, B = jnp.linalg.eigh(cov)
    # avoid numerical problems
    D = jnp.maximum(D, jnp.finfo(float).eps)
    D = jnp.sqrt(D)
    return B.dot(jnp.diag(1.0 / D)).dot(B.T), B, D


class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy.

    Parameters
    ----------
    initial_params : array-like, shape = (n_params,), optional (default: 0s)
        Initial parameter vector.

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

    bounds : array-like, shape (n_params, 2), optional (default: None)
        Upper and lower bounds for each parameter.

    maximize : boolean, optional (default: True)
        Maximize return or minimize cost?

    min_variance : float, optional (default: 2 * np.finfo(float).eps ** 2)
        Minimum variance before restart

    min_fitness_dist : float, optional (default: 2 * np.finfo(float).eps)
        Minimum distance between fitness values before restart

    max_condition : float optional (default: 1e7)
        Maximum condition of covariance matrix

    log_to_file: boolean or string, optional (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: boolean, optional (default: False)
        Log to standard output

    random_state : int or RandomState, optional (default: None)
        Seed for the random number generator or RandomState object.

    verbose : int, optional (default: 0)
        Verbosity level.
    """

    def __init__(
        self,
        initial_params: ArrayLike | None = None,
        variance: float = 1.0,
        covariance: ArrayLike | None = None,
        n_samples_per_update: int | None = None,
        active: bool = False,
        bounds: jnp.ndarray | None = None,
        maximize: bool = True,
        min_variance: float = 2 * jnp.finfo(float).eps ** 2,
        min_fitness_dist: float = 2 * jnp.finfo(float).eps,
        max_condition: float = 1e7,
        key: jnp.ndarray | None = None,
        verbose: int = 0,
    ):
        self.initial_params = initial_params
        self.variance = variance
        self.covariance = covariance
        self.n_samples_per_update = n_samples_per_update
        self.active = active
        self.bounds = bounds
        self.maximize = maximize
        self.min_variance = min_variance
        self.min_fitness_dist = min_fitness_dist
        self.max_condition = max_condition
        self.key = key
        self.verbose = verbose

        if self.key is None:
            self.key = jax.random.key(0)

        self.n_params = len(self.initial_params)
        self.it = 0
        self.eigen_decomp_updated = 0

        if self.initial_params is None:
            self.initial_params = jnp.zeros(self.n_params)
        else:
            self.initial_params = jnp.asarray(self.initial_params).copy()
        if self.n_params != len(self.initial_params):
            raise ValueError(
                f"Number of dimensions ({self.n_params}) does not match "
                f"number of initial parameters ({len(self.initial_params)})."
            )

        if self.covariance is None:
            self.covariance = jnp.eye(self.n_params)
        else:
            self.covariance = jnp.asarray(self.covariance).copy()
        if self.covariance.ndim == 1:
            self.covariance = jnp.diag(self.covariance)

        self.best_fitness = jnp.inf
        self.best_fitness_it = self.it
        self.best_params = self.initial_params.copy()

        self.var = self.variance

        if self.n_samples_per_update is None:
            self.n_samples_per_update = 4 + int(3 * math.log(self.n_params))
            if self.verbose:
                print(f"[CMA-ES] {self.n_samples_per_update=}")

        if self.bounds is not None:
            self.bounds = jnp.asarray(self.bounds)

        self.mean: jnp.ndarray = self.initial_params.copy()
        self.cov: jnp.ndarray = self.covariance.copy()

        self.samples: jnp.ndarray = self._sample(self.n_samples_per_update)
        self.fitness: list[float] = []

        # Sample weights for mean recombination
        self.mu: float = self.n_samples_per_update / 2.0
        self.weights: jnp.ndarray = math.log(self.mu + 0.5) - jnp.log1p(
            jnp.arange(int(self.mu))
        )
        self.mu: int = int(self.mu)
        self.weights: jnp.ndarray = self.weights / jnp.sum(self.weights)
        self.mueff = 1.0 / float(jnp.sum(self.weights**2))

        # Time constant for cumulation of the covariance
        self.cc: float = (4 + self.mueff / self.n_params) / (
            self.n_params + 4 + 2 * self.mueff / self.n_params
        )
        # Time constant for cumulation for sigma control
        self.cs: float = (self.mueff + 2) / (self.n_params + self.mueff + 5)
        # Learning rate for rank-one update
        self.c1: float = 2 / ((self.n_params + 1.3) ** 2 + self.mueff)
        # Learning rate for rank-mu update
        self.cmu: float = min(
            1 - self.c1, 2 * self.mueff - 2 + 1.0 / self.mueff
        ) / ((self.n_params + 2) ** 2 + self.mueff)
        # Damping for sigma
        self.damps: float = (
            1
            + 2
            * max(0.0, math.sqrt((self.mueff - 1) / (self.n_params + 1)) - 1)
            + self.cs
        )

        # Misc constants
        self.ps_update_weight: float = math.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        )
        self.hsig_threshold: float = 2 + 4.0 / (self.n_params + 1)
        self.eigen_update_freq: int = int(
            self.n_samples_per_update
            / ((self.c1 + self.cmu) * self.n_params * 10)
        )

        # Evolution path for covariance
        self.pc: jnp.ndarray = jnp.zeros(self.n_params)
        # Evolution path for sigma
        self.ps: jnp.ndarray = jnp.zeros(self.n_params)

        if self.active:
            self.alpha_old: float = 0.5
            self.neg_cmu: float = (
                (1.0 - self.cmu)
                * 0.25
                * self.mueff
                / ((self.n_params + 2) ** 1.5 + 2.0 * self.mueff)
            )

        self.invsqrtC: jnp.ndarray = inv_sqrt(self.cov)[0]
        self.eigen_decomp_updated: int = self.it

    def _sample(self, n_samples):
        self.key, sampling_key = jax.random.split(self.key, 2)
        samples = jax.random.multivariate_normal(
            sampling_key, self.mean, self.var * self.cov, (n_samples,)
        )
        if self.bounds is not None:
            samples = jnp.clip(samples, self.bounds[:, 0], self.bounds[:, 1])
        return samples

    def get_next_parameters(self):
        """Get next individual/parameter vector for evaluation.

        Returns
        -------
        params : array, shape (n_params,)
            Parameter vector
        """
        k = self.it % self.n_samples_per_update
        return self.samples[k]

    def set_evaluation_feedback(self, feedback: ArrayLike):
        """Set feedbacks for the parameter vector.

        Parameters
        ----------
        feedback : list of float
            feedbacks for each step or for the episode, depends on the problem
        """
        k = self.it % self.n_samples_per_update
        fitness_k = float(jnp.sum(feedback))
        if self.maximize:
            fitness_k = -fitness_k

        self.fitness.append(fitness_k)

        if fitness_k <= self.best_fitness:
            self.best_fitness = fitness_k
            self.best_fitness_it = self.it
            self.best_params = self.samples[k]

        self.it += 1

        if self.verbose >= 2:
            print(
                f"[CMA-ES] Iteration #{self.it}, fitness: {fitness_k}, "
                f"variance {self.var}"
            )

        if self.it % self.n_samples_per_update == 0:
            self._update(self.samples, jnp.asarray(self.fitness), self.it)
            self.fitness = []

    def _update(self, samples, fitness, it):
        # 1) Update sample distribution mean

        self.last_mean = self.mean
        ranking = jnp.argsort(fitness, axis=0)
        update_samples = samples[ranking[: self.mu]]
        self.mean = jnp.sum(
            self.weights[:, jnp.newaxis] * update_samples, axis=0
        )

        mean_diff = self.mean - self.last_mean
        sigma = jnp.sqrt(self.var)

        # 2) Cumulation: update evolution paths

        # Isotropic (step size) evolution path
        self.ps += (
            -self.cs * self.ps
            + self.ps_update_weight / sigma * self.invsqrtC.dot(mean_diff)
        )
        # Anisotropic (covariance) evolution path
        ps_norm_2 = jnp.linalg.norm(self.ps) ** 2  # Temporary constant
        generation = it / self.n_samples_per_update
        hsig = int(
            ps_norm_2
            / self.n_params
            / jnp.sqrt(1 - (1 - self.cs) ** (2 * generation))
            < self.hsig_threshold
        )
        self.pc *= 1 - self.cc
        self.pc += (
            hsig
            * jnp.sqrt(self.cc * (2 - self.cc) * self.mueff)
            * mean_diff
            / sigma
        )

        # 3) Update sample distribution covariance

        # Rank-1 update
        rank_one_update = jnp.outer(self.pc, self.pc)

        # Rank-mu update
        noise = (update_samples - self.last_mean) / sigma
        rank_mu_update = noise.T.dot(jnp.diag(self.weights)).dot(noise)

        # Correct variance loss by hsig
        c1a = self.c1 * (1 - (1 - hsig) * self.cc * (2.0 - self.cc))

        if self.active:
            neg_update = samples[ranking[::-1][: self.mu]]
            neg_update -= self.last_mean
            neg_update /= sigma
            neg_rank_mu_update = neg_update.T.dot(jnp.diag(self.weights)).dot(
                neg_update
            )

            self.cov *= 1.0 - c1a - self.cmu + self.neg_cmu * self.alpha_old
            self.cov += rank_one_update * self.c1
            self.cov += rank_mu_update * (
                self.cmu + self.neg_cmu * (1.0 - self.alpha_old)
            )
            self.cov -= neg_rank_mu_update * self.neg_cmu
        else:
            self.cov *= 1.0 - c1a - self.cmu
            self.cov += rank_one_update * self.c1
            self.cov += rank_mu_update * self.cmu

        # NOTE here is a bug: it should be cs / (2 * damps), however, that
        #      breaks unit tests and does not improve results
        log_step_size_update = (self.cs / self.damps) * (
            ps_norm_2 / self.n_params - 1
        )
        # NOTE some implementations of CMA-ES use the denominator
        # np.sqrt(self.n_params) * (1.0 - 1.0 / (4 * self.n_params) +
        #                           1.0 / (21 * self.n_params ** 2))
        # instead of self.n_params, in this case cs / damps is correct
        # Adapt step size with factor <= exp(0.6)
        self.var = self.var * jnp.exp(min((0.6, log_step_size_update))) ** 2

        if it - self.eigen_decomp_updated > self.eigen_update_freq:
            self.invsqrtC = inv_sqrt(self.cov)[0]
            self.eigen_decomp_updated = self.it

        self.samples = self._sample(self.n_samples_per_update)

    def is_behavior_learning_done(self):
        """Check if the optimization is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        if self.it <= self.n_samples_per_update:
            return False

        if not jnp.all(jnp.isfinite(self.fitness)):
            return True

        # Check for invalid values
        if not (
            jnp.all(jnp.isfinite(self.invsqrtC))
            and jnp.all(jnp.isfinite(self.cov))
            and jnp.all(jnp.isfinite(self.mean))
            and jnp.isfinite(self.var)
        ):
            if self.verbose:
                print("[CMA-ES] Stopping: infs or nans")
            return True

        if (
            self.min_variance is not None
            and jnp.max(jnp.diag(self.cov)) * self.var <= self.min_variance
        ):
            if self.verbose:
                print(f"[CMA-ES] Stopping: {self.var} < min_variance")
            return True

        max_dist = jnp.max(pdist(self.fitness[:, jnp.newaxis]))
        if max_dist < self.min_fitness_dist:
            if self.verbose:
                print(f"[CMA-ES] Stopping: {max_dist} < min_fitness_dist")
            return True

        cov_diag = jnp.diag(self.cov)
        if self.max_condition is not None and jnp.max(
            cov_diag
        ) > self.max_condition * jnp.min(cov_diag):
            if self.verbose:
                print(
                    f"[CMA-ES] Stopping: "
                    f"{jnp.max(cov_diag)} / {jnp.min(cov_diag)} > max_condition"
                )
            return True

        return False

    def get_best_parameters(self, method="best"):
        """Get the best parameters.

        Parameters
        ----------
        method : string, optional (default: 'best')
            Either 'best' or 'mean'

        Returns
        -------
        best_params : array-like, shape (n_params,)
            Best parameters
        """
        if method == "best":
            return self.best_params
        else:
            return self.mean

    def get_best_fitness(self):
        """Get the best observed fitness.

        Returns
        -------
        best_fitness : float
            Best fitness (sum of feedbacks) so far. Corresponds to the
            parameters obtained by get_best_parameters(method='best'). For
            maximize=True, this is the highest observed fitness, and for
            maximize=False, this is the lowest observed fitness.
        """
        if self.maximize:
            return -self.best_fitness
        else:
            return self.best_fitness


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
    env,
    policy,
    total_episodes: int,
    seed: int,
    variance: float = 1.0,
    covariance: ArrayLike | None = None,
    n_samples_per_update: int | None = None,
    active: bool = False,
    logger: LoggerBase | None = None,
):
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

    Returns
    -------
    policy : nnx.Module
        Trained policy network.

    References
    ----------
    .. [1] Hansen, N.; Ostermeier, A. Completely Derandomized Self-Adaptation
        in Evolution Strategies. In: Evolutionary Computation, 9(2), pp.
        159-195. https://www.lri.fr/~hansen/cmaartic.pdf
    """
    init_params = flat_params(policy)
    key = jax.random.key(seed)
    opt = CMAES(
        initial_params=init_params,
        variance=variance,
        covariance=covariance,
        n_samples_per_update=n_samples_per_update,
        active=active,
        maximize=True,
        key=key,
        verbose=0,
    )

    @nnx.jit
    def policy_action(policy, observation):
        return policy(observation)

    obs, _ = env.reset(seed=seed)

    step_counter = 0
    if logger is not None:
        logger.start_new_episode()
    for _ in tqdm.trange(total_episodes):
        set_params(policy, opt.get_next_parameters())
        ret = 0.0
        done = False
        while not done:  # episode
            action = np.asarray(policy_action(policy, jnp.asarray(obs)))

            next_obs, reward, termination, truncation, info = env.step(action)
            step_counter += 1
            obs = next_obs
            ret += reward
            done = termination or truncation

        obs, _ = env.reset()
        if logger is not None:
            logger.stop_episode(step_counter)
            logger.start_new_episode()
            logger.record_stat("return", ret)
            logger.record_epoch("policy", policy)
        step_counter = 0
        opt.set_evaluation_feedback(ret)

    print(f"[CMA-ES] {opt.best_fitness=}")
    set_params(policy, opt.get_best_parameters(method="mean"))
    return policy
