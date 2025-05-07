import dataclasses
from collections import deque

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from ..blox.value_policy import get_epsilon_greedy_action, get_greedy_action


@dataclasses.dataclass(frozen=False)
class Counter:
    transition_counter: list[list[list[int]]]
    """Maps o,a,o' to counter."""

    reward_history: list[list[list[list[float]]]]
    """Maps o,a,o' to list of rewards."""


def counter_update(
    counter: Counter,
    obs: int,
    act: int,
    reward: float,
    next_obs: int,
) -> Counter:
    counter.transition_counter[obs][act][next_obs] += 1
    counter.reward_history[obs][act][next_obs].append(reward)
    return counter


@dataclasses.dataclass(frozen=False)
class ForwardModel:
    transition: jnp.ndarray
    """Probability of transition o,a,o'."""

    reward: jnp.ndarray
    """Average reward for transition o,a,o'."""


def model_update(
    model: ForwardModel,
    counter: Counter,
    obs: int,
    act: int,
    next_obs: int,
):
    model.transition = model.transition.at[obs, act, next_obs].set(
        counter.transition_counter[obs][act][next_obs]
        / sum(counter.transition_counter[obs][act])
    )
    model.reward = model.reward.at[obs, act, next_obs].set(
        np.mean(counter.reward_history[obs][act][next_obs])
    )
    return model


def planning(
    model: ForwardModel,
    obs_buffer: jnp.ndarray,
    act_buffer: jnp.ndarray,
    n_planning_steps: int,
    key: jnp.ndarray,
    gamma: float,
    learning_rate: float,
    q_table: jnp.ndarray,
):
    sampling_keys = jax.random.split(key, n_planning_steps)
    for skey in sampling_keys:
        sample_idx = jax.random.randint(skey, (), 0, len(obs_buffer))
        obs = obs_buffer[sample_idx]
        act = act_buffer[sample_idx]
        # we could also sample instead of taking argmax
        next_obs = jnp.argmax(model.transition[obs, act])
        reward = model.reward[obs, act, next_obs]
        q_table = q_learning_update(
            int(obs),
            int(act),
            float(reward),
            int(next_obs),
            gamma,
            learning_rate,
            q_table,
        )
    return q_table


def q_learning_update(
    obs: int,
    act: int,
    reward: float,
    next_obs: int,
    gamma: float,
    learning_rate: float,
    q_table: jnp.ndarray,
) -> jnp.ndarray:
    next_act = get_greedy_action(None, q_table, next_obs)
    q_target = reward + gamma * q_table[next_obs, next_act] - q_table[obs, act]
    return q_table.at[obs, act].set(
        q_table[obs, act] + learning_rate * q_target
    )


def train_dynaq(
    env: gym.Env,
    q_table: jnp.ndarray,
    gamma: float = 0.99,
    learning_rate: float = 0.1,
    epsilon: float = 0.05,
    n_planning_steps: int = 5,
    total_timesteps: int = 1_000_000,
    buffer_size: int = 100,
    seed: int = 0,
):
    """Train tabular Dyna-Q for discrete state and action spaces.

    Dyna-Q integrates trial-and-error learning and planning into a process
    operating alternately on the environment and on a model of the environment.
    The model is learned online in parallel to learning a policy. Policy
    learning is based on reinforcement learning and planning. The Q-function
    is updated based on actual and simulated transitions.

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium environment.

    q_table : array
        Tabular action-value function.

    gamma : float, optional
        Discount factor.

    learning_rate : float, optional
        Learning rate for value function update.

    epsilon : float, optional
        Exploration probability for epsilon-greedy policy.

    n_planning_steps : int, optional
        Number of planning steps.

    total_timesteps : int, optional
        The number of environment steps to train for.

    buffer_size : int, optional
        The number of previous observations and actions that should be stored
        for random sampling during planning.

    seed : int, optional
        Seed for random number generator.
    """
    key = jax.random.key(seed)
    n_states, n_actions = q_table.shape
    counter = Counter(
        transition_counter=[
            [[0 for _ in range(n_states)] for _ in range(n_actions)]
            for _ in range(n_states)
        ],
        reward_history=[
            [[[] for _ in range(n_states)] for _ in range(n_actions)]
            for _ in range(n_states)
        ],
    )
    model = ForwardModel(
        transition=jnp.zeros((n_states, n_actions, n_states)),
        reward=jnp.zeros((n_states, n_actions, n_states)),
    )
    obs_buffer = deque(maxlen=buffer_size)
    act_buffer = deque(maxlen=buffer_size)

    obs, _ = env.reset(seed=seed)
    obs = int(obs)
    for _ in range(total_timesteps):
        key, sampling_key = jax.random.split(key, 2)
        act = int(
            get_epsilon_greedy_action(sampling_key, q_table, obs, epsilon)
        )
        next_obs, reward, terminated, truncated, _ = env.step(act)
        reward = float(reward)
        next_obs = int(next_obs)

        # TODO do we need buffers?
        # store sample in replay buffer
        obs_buffer.append(obs)
        act_buffer.append(act)

        # direct RL (Q-learning)
        q_table = q_learning_update(
            obs, act, reward, next_obs, gamma, learning_rate, q_table
        )

        counter = counter_update(counter, obs, act, reward, next_obs)
        model = model_update(model, counter, obs, act, next_obs)
        key, sampling_key = jax.random.split(key, 2)
        q_table = planning(
            model,
            jnp.asarray(obs_buffer, dtype=int),
            jnp.asarray(act_buffer, dtype=int),
            n_planning_steps,
            sampling_key,
            gamma,
            learning_rate,
            q_table,
        )

        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()
