from collections import deque
from collections.abc import Callable
from functools import partial

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ...model.cross_entropy_method import optimize_cem
from ...model.gaussian_mlp_ensemble import EnsembleOfGaussianMlps


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
    task_horizon
        Horizon in which the controller predicts states and optimizes actions.
    n_samples
        Number of sampled paths from the dynamics model.
    seed
        Seed for random number generator.
    verbose
        Verbosity level.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        reward_model: Callable[[ArrayLike, ArrayLike], jnp.ndarray],
        dynamics_model: EnsembleOfGaussianMlps,
        task_horizon: int,
        n_samples: int,
        seed: int,
        verbose: int = 0,
    ):
        self.action_space = action_space
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.task_horizon = task_horizon
        self.n_samples = n_samples
        self.verbose = verbose

        self.key = jax.random.PRNGKey(seed)

        # https://github.com/kchua/handful-of-trials/blob/master/dmbrl/controllers/MPC.py#L132
        self.init_var = (
            self.action_space.high - self.action_space.low
        ) ** 2 / 16.0

    def action(self, last_act: ArrayLike, obs: ArrayLike) -> jnp.ndarray:
        last_act = jnp.asarray(last_act)
        obs = jnp.asarray(obs)
        return self._trajectory_sampling_inf(last_act, obs)

    def _trajectory_sampling_inf(self, last_act: jnp.ndarray, obs: jnp.ndarray):
        """TSinf refers to particle bootstraps never changing during a trial."""
        assert last_act.ndim == 1
        assert obs.ndim == 1

        if self.verbose >= 5:
            print("[PETS/MPC] start trajectory sampling")
        self.key, bootstrap_key = jax.random.split(self.key, 2)
        model_indices = jax.random.randint(
            bootstrap_key,
            shape=(self.n_samples,),
            minval=0,
            maxval=self.dynamics_model.n_base_models,
        )
        if self.verbose >= 6:
            print(f"[PETS/MPC] samples indices: {model_indices}")
        actions_per_bootstrap = []
        rewards_per_bootstrap = []
        observations = jnp.vstack([obs for _ in range(self.n_samples)])
        actions = jnp.vstack([last_act for _ in range(self.n_samples)])
        for t in range(self.task_horizon):
            if self.verbose >= 6:
                print(f"[PETS/MPC] predict step: {t=}")
            actions_per_step = []
            rewards_per_step = []
            for i in range(self.n_samples):
                act, rew = (
                    self._cem_action_with_max_reward(  # vmap and jit, check source https://github.com/kchua/handful-of-trials/blob/master/dmbrl/controllers/MPC.py#L132
                        actions[i], observations[i]
                    )
                )
                actions_per_step.append(act)
                rewards_per_step.append(rew)

            actions = jnp.vstack(actions_per_step)
            obs_act = jnp.hstack((observations, actions))
            next_observations = self.dynamics_model.vmap_base_predict(
                obs_act, model_indices
            )

            observations = next_observations

            actions_per_bootstrap.append(actions_per_step)
            rewards_per_bootstrap.append(rewards_per_step)

        best_bootstrap = _find_best_bootstrap(jnp.array(rewards_per_bootstrap))

        return actions_per_bootstrap[0][best_bootstrap]

    def _cem_action_with_max_reward(self, last_act: jnp.ndarray, obs: jnp.ndarray):
        # https://github.com/kchua/handful-of-trials/blob/master/dmbrl/controllers/MPC.py#L214C9-L214C76
        self.key, cem_key = jax.random.split(self.key, 2)
        act = optimize_cem(
            partial(self.reward_model, obs=obs),
            last_act,
            key=cem_key,
            # TODO make configurable
            init_var=self.init_var,
            n_iter=5,
            n_population=400,
            n_elite=40,
            alpha=0.1,
            lower_bound=self.action_space.low,
            upper_bound=self.action_space.high,
        )
        rew = self.reward_model(act, obs)
        return act, rew

    def fit(
        self,
        observations: ArrayLike,
        actions: ArrayLike,
        next_observations: ArrayLike,
        n_epochs: int,
    ) -> "ModelPredictiveControl":
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

        if self.verbose >= 5:
            print("[PETS/MPC] start training")
        self.dynamics_model.fit(
            observations_actions, next_observations, n_epochs
        )
        if self.verbose >= 5:
            print("[PETS/MPC] training done")

        return self


@jax.jit
def _find_best_bootstrap(rewards_per_bootstrap: jnp.ndarray) -> jnp.ndarray:
    returns_per_bootstrap = jnp.sum(rewards_per_bootstrap, axis=0)
    best_bootstrap = jnp.argmax(returns_per_bootstrap)
    return best_bootstrap


def train_pets(
    env: gym.Env,
    reward_model: Callable[[ArrayLike, ArrayLike], jnp.ndarray],
    dynamics_model: EnsembleOfGaussianMlps,
    task_horizon: int,
    n_samples: int,
    seed: int = 1,
    buffer_size: int = 1_000_000,
    total_timesteps: int = 1_000_000,
    learning_starts: int = 100,
    batch_size: int = 256,
    n_epochs_per_iteration: int = 10,
    verbose: int = 0,
) -> ModelPredictiveControl:
    r"""Probabilistic Ensemble - Trajectory Sampling (PE-TS).

    Each probabilistic neural network of the probabilistic ensemble (PE)
    dynamics model captures aleatoric uncertainty (inherent variance of the
    observed data). The ensemble captures epistemic uncertainty through
    bootstrap disagreement far from data. The trajectory sampling (TS)
    propagation technique uses this dynamics model to re-sample each particle
    (with associated bootstrap) according to its probabilistic prediction at
    each point in time, up until a given horizon. At each time step, the
    model-predictive control (MPC) algorithm computes an optimal action
    sequence, applies the first action in the sequence, and repeats until the
    task-horizon.

    Algorithm:

    * Initialize dataset :math:`\mathcal{D}` with a random controller.
    * for trial :math:`k=1` to K do
        * Train a PE dynamics model :math:`f` given
          :math:`\mathcal{D}`.
        * for time :math:`t=0` to T (`task_horizon`) do
            * for actions samples :math:`a_{t:t+T} \sim CEM(\cdot)`,
              1 to `n_samples` do
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
    task_horizon
        Task horizon: number of time steps to predict with dynamics model.
    n_samples
        Number of action samples per time step.
    seed, optional
        Seed for random number generators in Jax and NumPy.
    buffer_size, optional
        Size of dataset for training of dynamics model.
    total_timesteps, optional
        Number of steps to execute in the environment.
    learning_starts
        Learning starts after this number of random steps was taken in the
        environment.
    batch_size
        Size of a batch during gradient computation.
    n_epochs_per_iteration
        Number of epochs to train in each iteration.
    verbose
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

    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    action_space: gym.spaces.Box = env.action_space
    rb = ReplayBuffer(buffer_size)

    mpc = ModelPredictiveControl(
        action_space,
        reward_model,
        dynamics_model,
        task_horizon,
        n_samples,
        seed,
        verbose=verbose - 1,
    )

    obs, _ = env.reset(seed=seed)
    action = None

    for t in range(total_timesteps):
        if verbose >= 5 and t % 100 == 0:
            print(f"[PETS] {t=}")
        if t >= learning_starts:
            D_obs, D_acts, D_rews, D_next_obs, D_dones = rb.sample_batch(
                batch_size, rng
            )
            mpc.fit(D_obs, D_acts, D_next_obs, n_epochs_per_iteration)

        if t < learning_starts:
            action = action_space.sample()
        else:
            assert action is not None
            action = mpc.action(action, obs)

        next_obs, reward, termination, truncation, info = env.step(action)

        rb.add_sample(obs, action, reward, next_obs, termination)

        if termination or truncation:
            if verbose:
                print(
                    f"{t=}, length={info['episode']['l']}, "
                    f"return={info['episode']['r']}"
                )
            obs, _ = env.reset()

        obs = next_obs

    return mpc
