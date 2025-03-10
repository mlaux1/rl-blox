from collections import deque
from collections.abc import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from rl_blox.model.gaussian_mlp_ensemble import EnsembleOfGaussianMlps


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

    TODO
    """

    def __init__(
        self,
        reward_model: Callable,
        dynamics_model: EnsembleOfGaussianMlps,
        task_horizon: int,
        n_samples: int,
        seed: int,
    ):
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.task_horizon = task_horizon
        self.n_samples = n_samples

        self.key = jax.random.PRNGKey(seed)

    def action(self, obs: ArrayLike) -> jnp.ndarray:
        raise NotImplementedError()

    def fit(
        self,
        observations: ArrayLike,
        actions: ArrayLike,
        next_observations: ArrayLike,
    ) -> "ModelPredictiveControl":
        X = jnp.hstack((observations, actions))
        Y = jnp.asarray(next_observations)
        self.dynamics_model.fit(X, Y, n_epochs=1)
        return self


def train_pets(
    env: gym.Env,
    reward_model: Callable,
    dynamics_model: EnsembleOfGaussianMlps,
    task_horizon: int,
    n_samples: int,
    seed: int = 1,
    buffer_size: int = 1_000_000,
    total_timesteps: int = 1_000_000,
    learning_starts: int = 25_000,
    batch_size: int = 256,
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
        * for time :math:`t=0` to `task_horizon` do
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
        Reward function for the environment.
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
        reward_model, dynamics_model, task_horizon, n_samples, seed
    )

    obs, _ = env.reset(seed=seed)

    for t in range(total_timesteps):
        if t > learning_starts:
            obs, acts, rews, next_obs, dones = rb.sample_batch(batch_size, rng)
            mpc.fit(obs, acts, next_obs)

        if t < learning_starts:
            action = action_space.sample()
        else:
            action = mpc.action(obs)

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
