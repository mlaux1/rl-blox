import dataclasses
import warnings
from collections import namedtuple
from collections.abc import Callable
from functools import partial

import chex
import gymnasium as gym
import jax
import numpy as np
import optax
from flax import nnx, struct
from jax import numpy as jnp
from jax.typing import ArrayLike
from tqdm.rich import trange

from ..blox.cross_entropy_method import cem_sample, cem_update
from ..blox.probabilistic_ensemble import (
    EnsembleTrainState,
    GaussianMLPEnsemble,
    train_ensemble,
)
from ..blox.replay_buffer import ReplayBuffer
from ..logging.logger import LoggerBase


@struct.dataclass
class PETSMPCConfig:
    """Configuration of Model-Predictive Control (MPC) for PE-TS."""

    plan_horizon: int
    n_particles: int
    n_samples: int
    n_opt_iter: int
    init_with_previous_plan: bool
    reward_model: Callable[[ArrayLike, ArrayLike], jnp.ndarray] = struct.field(
        pytree_node=False
    )
    action_space_shape: tuple[int, ...]
    avg_act: jnp.ndarray
    init_var: jnp.ndarray
    sample_fn: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = struct.field(pytree_node=False)
    update_fn: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray],
    ] = struct.field(pytree_node=False)


@dataclasses.dataclass(frozen=False)
class PETSMPCState:
    """State of Model-Predictive Control (MPC) for PE-TS."""

    dynamics_model: GaussianMLPEnsemble
    prev_plan: jnp.ndarray
    key: jnp.ndarray

    @staticmethod
    def initial_plan(config: PETSMPCConfig) -> jnp.ndarray:
        """Create initial plan for MPC."""
        return jnp.vstack([config.avg_act for _ in range(config.plan_horizon)])


def mpc_action(
    config: PETSMPCConfig,
    state: PETSMPCState,
    optimize_fn: Callable[
        [GaussianMLPEnsemble, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        jnp.ndarray,
    ],
    obs: ArrayLike,
) -> jnp.ndarray:
    """Plan next action with MPC and PE-TS.

    Parameters
    ----------
    config : PETSMPCConfig
        MPC configuration.

    state : PETSMPCState
        MPC state.

    optimize_fn : Callable
        PE-TS optimizer.

    obs : array-like, shape (n_observation_features,)
        Observation.

    Returns
    -------
    action : array, shape (n_action_features,)
        Next action to take.
    """
    obs = jnp.asarray(obs)
    assert obs.ndim == 1, "currently only supports 1d observation spaces"

    state.key, opt_key = jax.random.split(state.key, 2)
    if config.init_with_previous_plan:
        plan = state.prev_plan
    else:
        plan = jnp.broadcast_to(config.avg_act, state.prev_plan.shape)

    plan = optimize_fn(state.dynamics_model, plan, opt_key, obs)

    state.prev_plan = jnp.concatenate(
        (plan[1:], config.avg_act[jnp.newaxis]), axis=0
    )

    return plan[0]


def _pets_optimize(
    config: PETSMPCConfig,
    dynamics_model: GaussianMLPEnsemble,
    mean: jnp.ndarray,
    key: jnp.ndarray,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    """Optimize plan with PE-TS."""
    best_plan = mean
    best_return = -jnp.inf

    key, bootstrap_key = jax.random.split(key, 2)
    model_indices = jax.random.randint(
        bootstrap_key,
        shape=(config.n_particles,),
        minval=0,
        maxval=dynamics_model.n_ensemble,
    )

    var = config.init_var
    for _ in range(config.n_opt_iter):
        mean, var, best_plan, best_return, expected_returns = _pets_opt_iter(
            config,
            dynamics_model,
            key,
            obs,
            model_indices,
            mean,
            var,
            best_plan,
            best_return,
        )

    return mean


def _pets_opt_iter(
    config: PETSMPCConfig,
    dynamics_model: GaussianMLPEnsemble,
    key: jnp.ndarray,
    obs: jnp.ndarray,
    model_indices: jnp.ndarray,
    mean: jnp.ndarray,
    var: jnp.ndarray,
    best_plan: jnp.ndarray,
    best_return: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One iteration of the optimizer."""
    key, sampling_key = jax.random.split(key, 2)
    actions = config.sample_fn(mean, var, sampling_key)
    chex.assert_shape(
        actions,
        (config.n_samples, config.plan_horizon) + config.action_space_shape,
    )
    key, particle_key = jax.random.split(key, 2)
    particle_keys = jax.random.split(
        particle_key, (config.n_samples, config.n_particles)
    )
    chex.assert_shape(particle_keys, (config.n_samples, config.n_particles))
    chex.assert_shape(model_indices, (config.n_particles,))
    chex.assert_shape(
        actions,
        (config.n_samples, config.plan_horizon) + config.action_space_shape,
    )
    chex.assert_shape(obs, (obs.shape[0],))
    trajectories = ts_inf(
        particle_keys,
        model_indices,
        actions,
        obs,
        dynamics_model,
    )
    chex.assert_shape(
        trajectories,
        (
            config.n_samples,
            config.n_particles,
            # initial observation + trajectory:
            config.plan_horizon + 1,
            trajectories.shape[-1],
        ),
    )
    expected_returns = evaluate_plans(
        actions, trajectories, config.reward_model
    )
    chex.assert_shape(expected_returns, (config.n_samples,))
    mean, var = config.update_fn(actions, expected_returns, mean, var)
    best_idx = jnp.argmax(expected_returns)
    best_return = jnp.where(
        expected_returns[best_idx] >= best_return,
        expected_returns[best_idx],
        best_return,
    )
    best_plan = jnp.where(
        expected_returns[best_idx] >= best_return, actions[best_idx], best_plan
    )
    return mean, var, best_plan, best_return, expected_returns


@nnx.jit
@partial(
    jax.vmap,  # over samples for CEM
    # key, model_idx, acts, obs, dynamics_model
    in_axes=(0, None, 0, None, None),
)
@partial(
    jax.vmap,  # over particles for estimation of return
    # key, model_idx, acts, obs, dynamics_model
    in_axes=(0, 0, None, None, None),
)
def ts_inf(
    key: jnp.ndarray,
    model_idx: int,
    acts: jnp.ndarray,
    obs: jnp.ndarray,
    dynamics_model: GaussianMLPEnsemble,
):
    """Trajectory sampling infinity (TSinf).

    Notes
    -----

    Particles do never change the bootstrap during a trial.

    Parameters
    ----------
    keys : array, shape (n_samples, n_particles)
        Keys for random number generator.
    model_idx : array, (n_particles,)
        Each particle will use another base model for sampling.
    acts : array, shape (n_samples, plan_horizon) + action_space.shape
        A sequence of actions to take for each sample of the optimizer.
        Actions at times t:t+T with the horizon T.
    obs : array, shape observation_space.shape
        Observation at time t.
    dynamics_model : GaussianMLPEnsemble
        Dynamics model.

    Returns
    -------
    obs : array, shape (n_samples, n_particles, plan_horizon + 1) + obs.shape
        Sequences of observations sampled with plans.

    Examples
    --------
    >>> from flax import nnx
    >>> import jax
    >>> import chex
    >>> model = GaussianMLPEnsemble(
    ...     5, False, 4, 3, [500, 500, 500], nnx.Rngs(0))
    >>> n_samples = 400
    >>> n_particles = 20
    >>> plan_horizon = 100
    >>> key = jax.random.key(0)
    >>> key, samp_key, model_key, act_key, obs_key = jax.random.split(key, 5)
    >>> sampling_keys = jax.random.split(samp_key, (n_samples, n_particles))
    >>> model_indices = jax.random.randint(
    ...     model_key, (n_particles,), 0, model.n_ensemble)
    >>> acts = jax.random.normal(act_key, (n_samples, plan_horizon, 1))
    >>> obs = jax.random.normal(obs_key, (3,))
    >>> trajectories = ts_inf(sampling_keys, model_indices, acts, obs, model)
    >>> chex.assert_shape(
    ...     trajectories, (n_samples, n_particles, plan_horizon + 1, 3))
    """
    # https://github.com/kchua/handful-of-trials/blob/master/dmbrl/controllers/MPC.py#L318
    observations = [obs]
    sampling_keys = jax.random.split(key, len(acts))
    for act, sampling_key in zip(acts, sampling_keys, strict=False):
        # We sample from one of the base models.
        # https://github.com/kchua/handful-of-trials/blob/master/dmbrl/controllers/MPC.py#L340
        dist = dynamics_model.base_distribution(
            jnp.hstack((obs, act)), model_idx
        )
        delta_obs = dist.sample(seed=sampling_key)[0]
        obs = obs + delta_obs
        observations.append(obs)
    return jnp.array(observations)


def evaluate_plans(
    actions: jnp.ndarray,
    trajectories: jnp.ndarray,
    reward_model: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Evaluate plans based on sampled trajectories.

    Parameters
    ----------
    actions : array, shape (n_samples, plan_horizon) + action.shape
        Action sequences (plans).

    trajectories : array, shape (n_samples, n_par, plan_horizon + 1) + obs.shape
        Sequences of observations sampled with `actions`.
        Note: `n_par` is the abbreviation for `n_particles`. When numpydoc
        allows multiline type documentation, this should be updated
        (issue: https://github.com/numpy/numpydoc/issues/87).

    reward_model : callable
        Mapping from pairs of state and action to reward.

    Returns
    -------
    expected_returns : array, shape (n_samples,)
        Expected returns, summed up over planning horizon, averaged over
        particles.
    """
    n_samples, plan_horizon = actions.shape[:2]
    action_shape = actions.shape[2:]
    n_particles = trajectories.shape[1]

    broadcasted_actions = jnp.broadcast_to(
        actions[:, jnp.newaxis],
        (n_samples, n_particles, plan_horizon) + action_shape,
    )  # broadcast actions along particle axis
    # TODO the reward model could be extended to include next observations
    rewards = reward_model(broadcasted_actions, trajectories[:, :, :-1])
    # sum along plan_horizon axis
    returns = rewards.sum(axis=-1)
    # mean along particle axis
    expected_returns = returns.mean(axis=-1)
    return expected_returns


def create_pets_state(
    env: gym.Env,
    seed: int,
    n_ensemble: int = 5,
    hidden_nodes: tuple[int] | list[int] = (500, 500, 500),
    activation: str = "swish",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    train_size: float = 0.7,
    batch_size: int = 32,
):
    model = GaussianMLPEnsemble(
        n_ensemble=n_ensemble,
        n_features=env.observation_space.shape[0] + env.action_space.shape[0],
        n_outputs=env.observation_space.shape[0],
        shared_head=True,
        hidden_nodes=list(hidden_nodes),
        activation=activation,
        rngs=nnx.Rngs(seed),
    )
    return EnsembleTrainState(
        model=model,
        optimizer=nnx.Optimizer(
            model,
            optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
            wrt=nnx.Param,
        ),
        train_size=train_size,
        batch_size=batch_size,
    )


def train_pets(
    env: gym.Env,
    reward_model: Callable[[ArrayLike, ArrayLike], jnp.ndarray],
    dynamics_model: EnsembleTrainState,
    plan_horizon: int,
    n_particles: int,
    n_samples: int,
    n_opt_iter: int = 5,
    init_with_previous_plan: bool = True,
    seed: int = 1,
    buffer_size: int = 1_000_000,
    total_timesteps: int = 1_000_000,
    learning_starts: int = 100,
    learning_starts_gradient_steps: int = 100,
    n_steps_per_iteration: int = 100,
    gradient_steps: int = 10,
    replay_buffer: ReplayBuffer | None = None,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[
    PETSMPCConfig,
    PETSMPCState,
    Callable[
        [GaussianMLPEnsemble, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        jnp.ndarray,
    ],
    ReplayBuffer,
]:
    r"""Probabilistic Ensemble - Trajectory Sampling (PE-TS).

    Each probabilistic neural network of the probabilistic ensemble (PE)
    dynamics model captures aleatoric uncertainty (inherent variance of the
    observed data). The ensemble captures epistemic uncertainty through
    bootstrap disagreement far from data. The trajectory sampling (TS)
    propagation technique uses this dynamics model to resample each particle
    (with associated bootstrap) according to its probabilistic prediction at
    each point in time, up until a given planning horizon. At each time step,
    the model-predictive control (MPC) algorithm computes an optimal action
    sequence, applies the first action in the sequence, and repeats until the
    task horizon.

    Parameters
    ----------
    env
        gymnasium environment.
    reward_model
        Vectorized implementation of the environment's reward function.
        The first argument should be an array of actions (act). The second
        argument should be the current observation (obs), in which these
        actions will be executed. For each action it should return the reward
        associated with the pair of the observation and action.
    dynamics_model
        Probabilistic ensemble dynamics model.
    plan_horizon
        Planning horizon: number of time steps to predict with dynamics model.
    n_particles
        Number of particles to compute the expected returns.
    n_samples
        Number of action samples per time step.
    n_opt_iter, optional
        Number of iterations of the optimization algorithm.
    init_with_previous_plan
        Initialize optimizer in each step with previous plan shifted by one
        time step.
    seed, optional
        Seed for random number generators in Jax and NumPy.
    buffer_size, optional
        Size of dataset for training of dynamics model.
    total_timesteps, optional
        Number of steps to execute in the environment.
    learning_starts, optional
        Learning starts after this number of random steps was taken in the
        environment. Should correspond to the expected number of steps in one
        episode.
    learning_start_gradient_steps
        Number of gradient steps used after learning_starts steps.
    n_steps_per_iteration
        Number of steps to take in the environment before we refine the model.
        Should correspond to the expected number of steps in one episode.
    gradient_steps, optional
        Number of gradient steps during one training phase.
    replay_buffer : ReplayBuffer
        Replay buffer.
    logger : logger.LoggerBase, optional
        Experiment logger.
    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.


    Returns
    -------
    mpc_config : PETSMPCConfig
        MPC config.
    mpc_state : PETSMPCState
        MPC state.
    optimizer_fn : callable
        PETS action optimizer.
    replay_buffer : ReplayBuffer
        Replay buffer.

    Notes
    -----
    The original PE-TS algorithm can be summarized as follows.

    * Parameters

      * environment with specfic task horizon
      * :math:`r(o, a)` - reward model (`reward_model`)
      * :math:`f` - probabilistic ensemble (PE) dynamics model
      * :math:`T` - planning horizon (`plan_horizon`) for trajectory sampling
        (TS)
      * :math:`CEM(\cdot)` - cross entropy method (CEM) optimizer
      * :math:`N` - number of samples for CEM (`n_samples`)
      * :math:`P` - number of particles for trajectory sampling
        (`n_particles`)
    * Initialize dataset :math:`\mathcal{D}` (``replay_buffer``) with a random
      controller.
    * for trial :math:`k=1` to K do

      * Train dynamics model :math:`f` given :math:`\mathcal{D}` (see
        :func:`update_dynamics_model`)
      * for time :math:`t=0` to task horizon do

        * for actions samples :math:`a_{t:t+T} \sim CEM(\cdot)`,
          1 to :math:`N` do (see :func:`mpc_action`)

          * Propagate observation particles :math:`o_{\tau}^p` using TS with
            :math:`f,a_{t:t+T}` (see :func:`ts_inf`)
          * Evaluate actions as
            :math:`\sum_{\tau=t}^{t+T} \frac{1}{P} \sum_{p=1}^P
            r(o_{\tau}^p, a_{\tau})` (see :func:`evaluate_plans`)
          * Update :math:`CEM(\cdot)` distribution.
      * Execute first action :math:`a_t^*` (only) from optimal actions
        :math:`a_{t:t+T}^*`.
      * Record outcome:
        :math:`\mathcal{D} \leftarrow \mathcal{D} \cup
        \{s_t, a_t^*, s_{t+1}\}`

    This implementation modifies the original algorithm. We change the
    episode-driven training to step-driven training so that you can specify
    an initial number of random steps (`learning_starts`), a total number of
    steps (`total_timesteps`), and a number of steps that will be recorded
    before the dynamics model is trained for an epoch
    (`n_steps_per_iteration`).

    In addition, the number of gradient steps to train the initial model
    (`learning_start_gradient_steps`) and to train the model in each iteration
    (`gradient_steps`) can be specified separately.

    The model-predictive controller can be configured to not initialize with
    the last solution (`init_with_previous_plan=False`) if desired.

    References
    ----------
    .. [1] Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine.
           2018. Deep reinforcement learning in a handful of trials using
           probabilistic dynamics models. In Proceedings of the 32nd
           International Conference on Neural Information Processing Systems
           (NeurIPS'18). Curran Associates Inc., Red Hook, NY, USA, 4759–4770.
           https://papers.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html
    """
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    action_space: gym.spaces.Box = env.action_space

    if replay_buffer is None:
        replay_buffer = ReplayBuffer(buffer_size)

    # Initialize model-predictive control: configuration, state, and optimizer
    sample_fn, update_fn = _init_mpc_optimizer_cem(
        env.action_space, plan_horizon, n_samples
    )
    mpc_config = PETSMPCConfig(
        plan_horizon=plan_horizon,
        n_particles=n_particles,
        n_samples=n_samples,
        n_opt_iter=n_opt_iter,
        init_with_previous_plan=init_with_previous_plan,
        reward_model=reward_model,
        action_space_shape=env.action_space.shape,
        avg_act=jnp.asarray(
            0.5 * (env.action_space.high + env.action_space.low)
        ),
        init_var=jnp.array(
            [
                (env.action_space.high - env.action_space.low) ** 2 / 16.0
                for _ in range(plan_horizon)
            ]
        ),
        sample_fn=sample_fn,
        update_fn=update_fn,
    )
    mpc_state = PETSMPCState(
        dynamics_model=dynamics_model.model,
        prev_plan=PETSMPCState.initial_plan(mpc_config),
        key=jax.random.key(seed),
    )
    try:
        mpc_optimize_fn = nnx.jit(partial(_pets_optimize, mpc_config))
        mpc_optimize_fn(
            mpc_state.dynamics_model,
            mpc_state.prev_plan,
            mpc_state.key,
            jnp.zeros(env.observation_space.shape),
        )
    except jax.errors.ConcretizationTypeError as e:
        warnings.warn(
            f"nnx.jit failed. MPC will be slow. Check if your reward "
            f"model is JIT-compilable. Error message: {e}",
            stacklevel=2,
        )
        mpc_optimize_fn = partial(_pets_optimize, mpc_config)

    n_epochs = learning_starts_gradient_steps

    env.action_space.seed(seed)

    obs, _ = env.reset(seed=seed)
    if logger is not None:
        logger.start_new_episode()
    steps_per_episode = 0

    for t in trange(total_timesteps, disable=not progress_bar):
        if (
            t >= learning_starts
            and (t - learning_starts) % n_steps_per_iteration == 0
        ):
            D_obs, D_acts, _, D_next_obs, _ = replay_buffer.sample_batch(
                len(replay_buffer), rng
            )
            key, train_key = jax.random.split(key)
            dynamics_model_loss = update_dynamics_model(
                dynamics_model, D_obs, D_acts, D_next_obs, train_key, n_epochs
            )
            n_epochs = gradient_steps
            if logger is not None:
                logger.record_stat(
                    "dynamics model loss", dynamics_model_loss, step=t
                )
                logger.record_epoch(
                    "dynamics_model", dynamics_model.model, step=t
                )

        if t < learning_starts:
            action = action_space.sample()
        else:
            action = mpc_action(mpc_config, mpc_state, mpc_optimize_fn, obs)

        next_obs, reward, termination, truncation, info = env.step(action)
        steps_per_episode += 1

        replay_buffer.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            termination=termination,
        )

        if termination or truncation:
            if logger is not None:
                if "episode" in info:
                    logger.record_stat("return", info["episode"]["r"], step=t)
                logger.stop_episode(steps_per_episode)
                logger.start_new_episode()

            steps_per_episode = 0
            obs, _ = env.reset()
            mpc_state.prev_plan = PETSMPCState.initial_plan(mpc_config)

        obs = next_obs

    return namedtuple(
        "PETSResult",
        ["mpc_config", "mpc_state", "mpc_optimize_fn", "replay_buffer"],
    )(mpc_config, mpc_state, mpc_optimize_fn, replay_buffer)


def _init_mpc_optimizer_cem(
    action_space: gym.spaces.Box,
    plan_horizon: int,
    n_samples: int,
    n_elite: int | None = None,
    alpha: float = 0.1,
) -> tuple[
    Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray],
    ],
]:
    """Init CEM optimizer for MPC."""
    lower_bound = jnp.vstack([action_space.low for _ in range(plan_horizon)])
    upper_bound = jnp.vstack([action_space.high for _ in range(plan_horizon)])
    _sample = jax.jit(
        partial(
            cem_sample,
            n_population=n_samples,
            lb=lower_bound,
            ub=upper_bound,
        )
    )
    if n_elite is None:
        n_elite = int(0.1 * n_samples)
    _update_search_distribution = jax.jit(
        partial(
            cem_update,
            n_elite=n_elite,
            alpha=alpha,
        )
    )
    return _sample, _update_search_distribution


def update_dynamics_model(
    dynamics_model: EnsembleTrainState,
    observations: ArrayLike,
    actions: ArrayLike,
    next_observations: ArrayLike,
    train_key: jnp.ndarray,
    n_epochs: int,
) -> jnp.ndarray:
    """Train dynamics model.

    Parameters
    ----------
    dynamics_model : GaussianMLPEnsemble
        Dynamics model to train.

    observations : array-like, shape (n_samples, n_observation_features)
        Observations.

    actions : array-like, shape (n_samples, n_action_features)
        Actions

    next_observations : array-like, shape (n_samples, n_observation_features)
        Next observations.

    train_key : array
        Random key for training.

    n_epochs : int
        Number of epochs to train.

    Returns
    -------
    loss : array, shape ()
        Mean loss of batches during last epoch.
    """
    observations = jnp.asarray(observations)
    actions = jnp.asarray(actions)
    next_observations = jnp.asarray(next_observations)

    chex.assert_equal_shape((observations, next_observations))
    chex.assert_equal_shape_prefix((observations, actions), prefix_len=1)

    observations_actions = jnp.hstack((observations, actions))

    chex.assert_shape(
        observations_actions,
        (observations.shape[0], observations.shape[1] + actions.shape[1]),
    )

    return train_ensemble(
        model=dynamics_model.model,
        optimizer=dynamics_model.optimizer,
        train_size=dynamics_model.train_size,
        X=observations_actions,
        # This is configurable in the original implementation, although it
        # is the same for every environment used in the experiments. We
        # assume that we are dealing with continuous state vectors and
        # predict the delta in the transition.
        Y=next_observations - observations,
        n_epochs=n_epochs,
        batch_size=dynamics_model.batch_size,
        key=train_key,
    )
