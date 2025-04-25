from collections import deque
from collections.abc import Callable
from functools import partial

import chex
import gymnasium as gym
import jax
import numpy as np
import optax
from flax import nnx, struct
from gymnasium.wrappers import RecordEpisodeStatistics
from jax import numpy as jnp
from jax.typing import ArrayLike

from ...model.cross_entropy_method import cem_sample, cem_update
from ...model.probabilistic_ensemble import (
    EnsembleTrainState,
    GaussianMLPEnsemble,
    store_checkpoint,
    train_ensemble,
)


class ReplayBuffer:
    buffer: deque[tuple[ArrayLike, ArrayLike, float, ArrayLike, bool]]

    def __init__(self, n_samples):
        self.buffer = deque(maxlen=n_samples)

    def add_sample(self, observation, action, reward, next_observation, done):
        self.buffer.append(
            (
                observation,
                action,
                reward,
                next_observation,
                done,
            )
        )

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        indices = rng.integers(0, len(self.buffer), batch_size)
        observations = jnp.vstack([self.buffer[i][0] for i in indices])
        actions = jnp.stack([self.buffer[i][1] for i in indices])
        rewards = jnp.hstack([self.buffer[i][2] for i in indices])
        next_observations = jnp.vstack([self.buffer[i][3] for i in indices])
        dones = jnp.hstack([self.buffer[i][4] for i in indices])
        return observations, actions, rewards, next_observations, dones

    def mean_reward(self, n_steps):
        if n_steps > len(self.buffer):
            return jnp.inf
        return jnp.mean(
            jnp.hstack([self.buffer[i][2] for i in np.arange(-n_steps, 0)])
        )

    def __len__(self):
        return len(self.buffer)


@struct.dataclass
class MPCConfig:
    """Configuration of Model-Predictive Control (MPC)."""

    plan_horizon: int
    n_particles: int
    n_samples: int
    n_opt_iter: int
    init_with_previous_plan: bool
    verbose: int
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


class ModelPredictiveControl:
    """Model-Predictive Control (MPC).

    Parameters
    ----------
    action_space
        Action space of the environment.
    reward_model
        Vectorized implementation of the environment's reward function.
        The first argument should be an array of actions (act). The second
        argument should be the current observation (obs), in which these
        actions will be executed. For each action it should return the reward
        associated with the pair of the observation and action.
    dynamics_model
        Learned model of the environment's dynamic.
    plan_horizon
        Horizon in which the controller predicts states and optimizes actions.
    n_particles
        Number of particles to compute the expected returns.
    n_samples
        Number of sampled paths from the dynamics model.
    n_opt_iter
        Number of iterations of the optimization algorithm.
    init_with_previous_plan
        Initialize optimizer in each step with previous plan shifted by one
        time step.
    seed
        Seed for random number generator.
    verbose
        Verbosity level.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        reward_model: Callable[[ArrayLike, ArrayLike], jnp.ndarray],
        dynamics_model: GaussianMLPEnsemble,
        plan_horizon: int,
        n_particles: int,
        n_samples: int,
        n_opt_iter: int,
        init_with_previous_plan: bool,
        seed: int,
        verbose: int = 0,
    ):
        sample_fn, update_fn = _init_mpc_optimizer_cem(
            action_space, plan_horizon, n_samples
        )
        self.config = MPCConfig(
            plan_horizon=plan_horizon,
            n_particles=n_particles,
            n_samples=n_samples,
            n_opt_iter=n_opt_iter,
            init_with_previous_plan=init_with_previous_plan,
            verbose=verbose,
            reward_model=reward_model,
            action_space_shape=action_space.shape,
            avg_act=jnp.asarray(0.5 * (action_space.high + action_space.low)),
            init_var=jnp.array(
                [
                    (action_space.high - action_space.low) ** 2 / 16.0
                    for _ in range(plan_horizon)
                ]
            ),
            sample_fn=sample_fn,
            update_fn=update_fn,
        )

        self.dynamics_model = dynamics_model
        self.key = jax.random.key(seed)
        self.start_episode()

    def start_episode(self) -> None:
        """Tell MPC that a new episode started."""
        self.prev_plan = jnp.vstack(
            [self.config.avg_act for _ in range(self.config.plan_horizon)]
        )

    def action(self, obs: ArrayLike) -> jnp.ndarray:
        """Plan next action.

        Parameters
        ----------
        obs : array-like, shape (n_observation_features,)
            Observation.

        Returns
        -------
        action : array, shape (n_action_features,)
            Next action to take.
        """
        obs = jnp.asarray(obs)
        assert obs.ndim == 1

        best_plan = self._optimize_actions(obs)

        self.prev_plan = jnp.concatenate(
            (best_plan[1:], self.config.avg_act[jnp.newaxis]), axis=0
        )

        return best_plan[0]

    def _optimize_actions(self, obs):
        best_plan = self.prev_plan
        best_return = -jnp.inf

        self.key, bootstrap_key = jax.random.split(self.key, 2)
        model_indices = jax.random.randint(
            bootstrap_key,
            shape=(self.config.n_particles,),
            minval=0,
            maxval=self.dynamics_model.n_ensemble,
        )

        if self.config.init_with_previous_plan:
            mean = self.prev_plan
        else:
            mean = jnp.broadcast_to(self.config.avg_act, self.prev_plan.shape)
        var = jnp.copy(self.config.init_var)

        for i in range(self.config.n_opt_iter):
            mean, var, best_plan, best_return, expected_returns = (
                self._opt_iter(
                    obs, model_indices, mean, var, best_plan, best_return
                )
            )

            if self.config.verbose >= 2:
                print(
                    f"[PETS/MPC] it #{i + 1}, return "
                    f"{jnp.mean(expected_returns)} +- "
                    f"{jnp.std(expected_returns)}, "
                    f"[{expected_returns.min()}, {expected_returns.max()}]"
                )
        if self.config.verbose >= 1:
            print(f"[PETS/MPC] Best return [{best_return}]")

        return mean

    def _opt_iter(self, obs, model_indices, mean, var, best_plan, best_return):
        self.key, sampling_key = jax.random.split(self.key, 2)
        actions = self.config.sample_fn(mean, var, sampling_key)
        assert not jnp.any(jnp.isnan(actions))
        chex.assert_shape(
            actions,
            (self.config.n_samples, self.config.plan_horizon)
            + self.config.action_space_shape,
        )
        self.key, particle_key = jax.random.split(self.key, 2)
        particle_keys = jax.random.split(
            particle_key, (self.config.n_samples, self.config.n_particles)
        )
        chex.assert_shape(
            particle_keys, (self.config.n_samples, self.config.n_particles)
        )
        chex.assert_shape(model_indices, (self.config.n_particles,))
        chex.assert_shape(
            actions,
            (self.config.n_samples, self.config.plan_horizon)
            + self.config.action_space_shape,
        )
        chex.assert_shape(obs, (obs.shape[0],))
        trajectories = ts_inf(
            particle_keys,
            model_indices,
            actions,
            obs,
            self.dynamics_model,
        )
        chex.assert_shape(
            trajectories,
            (
                self.config.n_samples,
                self.config.n_particles,
                # initial observation + trajectory:
                self.config.plan_horizon + 1,
                trajectories.shape[-1],
            ),
        )
        expected_returns = evaluate_plans(
            actions, trajectories, self.config.reward_model
        )
        chex.assert_shape(expected_returns, (self.config.n_samples,))
        mean, var = self.config.update_fn(actions, expected_returns, mean, var)
        best_idx = jnp.argmax(expected_returns)
        if expected_returns[best_idx] >= best_return:
            best_return = expected_returns[best_idx]
            best_plan = actions[best_idx]
        return mean, var, best_plan, best_return, expected_returns


def create_pets_state(
    env: gym.Env,
    seed: int,
    n_ensemble: int = 5,
    hidden_nodes: tuple[int] | list[int] = (500, 500, 500),
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
        rngs=nnx.Rngs(seed),
    )
    return EnsembleTrainState(
        model=model,
        optimizer=nnx.Optimizer(
            model,
            optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        ),
        train_size=train_size,
        batch_size=batch_size,
    )


def train_pets(
    env: gym.Env,
    reward_model: Callable[[ArrayLike, ArrayLike], jnp.ndarray],
    dynamics_model: EnsembleTrainState,
    planning_horizon: int,
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
    save_checkpoints: bool = False,
    checkpoint_path_prefix: str = "/tmp",
    verbose: int = 0,
) -> ModelPredictiveControl:
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

    Algorithm:

    * Initialize dataset :math:`\mathcal{D}` with a random controller.
    * for trial :math:`k=1` to K do
        * Train a PE dynamics model :math:`f` given
          :math:`\mathcal{D}`.
        * for time :math:`t=0` to T_p (task horizon) do
            * for actions samples :math:`a_{t:t+T} \sim CEM(\cdot)`,
              1 to `plan_horizon` do
                * Propagate state particles :math:`s_{\tau}^p` using TS and
                  :math:`f|\left{\mathcal{D},a_{t:t+T}\right}`
                * Evaluate actions as
                  :math:`\sum_{\tau=t}^{t+T} \frac{1}{P} \sum_{p=1}^P
                  r(s_{\tau}^p, a_{\tau})`
                * Update :math:`CEM(\cdot)` distribution.
        * Execute first action :math:`a_t^*` (only) from optimal actions
          :math:`a_{t:t+T}^*`.
        * Record outcome:
          :math:`\mathcal{D} \leftarrow \mathcal{D} \cup
          \left{s_t, a_t^*, s_{t+1}\right}`

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
    planning_horizon
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
    save_checkpoints, optional
        Save checkpoint each time we update the model.
    checkpoint_path_prefix, optional
        Prefix of path at which we store checkpoints after each model update.
        Note that the path has to be absolute. '/tmp/' is recommended and used
        as default. Model checkpoints will be stored in
        '/tmp/pets_dynamics_model_iteration' in this case.
    verbose, optional
        Verbosity level.

    Returns
    -------
    mpc
        Model-predictive control based on dynamics model.

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

    if verbose:
        env = RecordEpisodeStatistics(env)

    rb = ReplayBuffer(buffer_size)

    mpc = ModelPredictiveControl(
        action_space,
        reward_model,
        dynamics_model.model,
        planning_horizon,
        n_particles,
        n_samples,
        n_opt_iter,
        init_with_previous_plan,
        seed,
        verbose=verbose - 1,
    )

    n_epochs = learning_starts_gradient_steps

    env.action_space.seed(seed)

    obs, _ = env.reset(seed=seed)

    for t in range(total_timesteps):
        if (
            t >= learning_starts
            and (t - learning_starts) % n_steps_per_iteration == 0
        ):
            D_obs, D_acts, _, D_next_obs, _ = rb.sample_batch(len(rb), rng)
            if verbose >= 2:
                print("[PETS] start training")
            key, train_key = jax.random.split(key)
            loss = update_dynamics_model(
                dynamics_model, D_obs, D_acts, D_next_obs, train_key, n_epochs
            )
            if verbose >= 2:
                print(f"[PETS] training done; {loss=}")
            n_epochs = gradient_steps
            if save_checkpoints:  # TODO use logging interface
                store_checkpoint(
                    f"{checkpoint_path_prefix}/pets_dynamics_model_{t}",
                    dynamics_model.model,
                )

        if t < learning_starts:
            action = action_space.sample()
        else:
            action = mpc.action(obs)

        next_obs, reward, termination, truncation, info = env.step(action)

        rb.add_sample(obs, action, reward, next_obs, termination)

        if termination or truncation:
            if verbose >= 1:
                print(f"{t=}, {info=}")
            mpc.start_episode()
            obs, _ = env.reset()

        obs = next_obs

    return mpc


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
    obs : array, shape (n_samples, n_particles, plan_horizon + 1)
          + observation_space.shape
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
    return jnp.vstack(observations)


def evaluate_plans(
    actions: jnp.ndarray,
    trajectories: jnp.ndarray,
    reward_model: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Evaluate plans based on sampled trajectories.

    Parameters
    ----------
    actions : array, shape (n_samples, plan_horizon) + action_space.shape
        Action sequences (plans).

    trajectories : array,
            shape (n_samples, n_particles, plan_horizon + 1)
            + observation_space.shape
        Sequences of observations sampled with plans.

    reward_model : callable
        Mapping from pairs of state and action to reward.

    Returns
    -------
    expected_returns : array, shape (n_samples,)
        Expected returns, summed up over planning horizon, averaged over
        particles.
    """
    assert not jnp.any(jnp.isnan(trajectories))
    n_samples, plan_horizon = actions.shape[:2]
    action_shape = actions.shape[2:]
    n_particles = trajectories.shape[1]

    broadcasted_actions = np.broadcast_to(
        actions[:, jnp.newaxis],
        (n_samples, n_particles, plan_horizon) + action_shape,
    )  # broadcast along particle axis
    rewards = reward_model(broadcasted_actions, trajectories[:, :, :-1])
    # sum along plan_horizon axis
    returns = rewards.sum(axis=-1)
    # mean along particle axis
    expected_returns = returns.mean(axis=-1)
    return expected_returns


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
    dynamics_model
        Dynamics model to train.

    observations : array-like, shape (n_samples, n_observation_features)
        Observations.

    actions : array-like, shape (n_samples, n_action_features)
        Actions

    next_observations : array-like, shape (n_samples, n_observation_features)
        Next observations.

    train_key : jnp.ndarray
        Random key for training.

    n_epochs : int
        Number of epochs to train.

    Returns
    -------
    loss
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
