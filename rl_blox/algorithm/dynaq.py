import dataclasses
from collections import deque

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.rich import trange

from ..blox.value_policy import epsilon_greedy_policy, greedy_policy
from ..logging.logger import LoggerBase


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
    model_transition: jnp.ndarray,
    model_reward: jnp.ndarray,
    obs_buffer: jnp.ndarray,
    act_buffer: jnp.ndarray,
    n_planning_steps: int,
    key: jnp.ndarray,
    gamma: float,
    learning_rate: float,
    q_table: jnp.ndarray,
):
    key, sampling_key = jax.random.split(key, 2)
    samples = jax.random.randint(
        sampling_key, (n_planning_steps,), 0, len(obs_buffer)
    )
    observations = obs_buffer[samples]
    actions = act_buffer[samples]
    for obs, act in zip(observations, actions, strict=False):
        # we could also sample instead of taking argmax
        next_obs = jnp.argmax(model_transition[obs, act])
        reward = model_reward[obs, act, next_obs]
        q_table = q_learning_update(
            obs,
            act,
            reward,
            next_obs,
            gamma,
            learning_rate,
            q_table,
        )
    return q_table


@jax.jit
def q_learning_update(
    obs: int,
    act: int,
    reward: float,
    next_obs: int,
    gamma: float,
    learning_rate: float,
    q_table: jnp.ndarray,
) -> jnp.ndarray:
    next_act = greedy_policy(q_table, next_obs)
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
    buffer_size: int = 1_000,
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> jnp.ndarray:
    """Train tabular Dyna-Q for discrete state and action spaces.

    Dyna-Q [1]_ integrates trial-and-error learning and planning into a process
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

    buffer_size : int, optional
        The number of previous observations and actions that should be stored
        for random sampling during planning.

    total_timesteps : int, optional
        The number of environment steps to train for.

    seed : int, optional
        Seed for random number generator.

    logger : LoggerBase, optional
        Logger.

    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.

    Returns
    -------
    q_table : array
        Tabular action-value function of the optimal policy.

    References
    ----------
    .. [1] Sutton, Richard S. (1990). Integrated architecture for learning,
       planning, and reacting based on approximating dynamic programming.
       In Proceedings of the Seventh International Conference (1990) on Machine
       Learning. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA,
       pp. 216–224. http://incompleteideas.net/papers/sutton-90.pdf
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
    accumulated_reward = 0.0
    for t in trange(total_timesteps, disable=not progress_bar):
        key, sampling_key = jax.random.split(key, 2)
        act = int(epsilon_greedy_policy(q_table, obs, epsilon, sampling_key))
        next_obs, reward, terminated, truncated, _ = env.step(act)
        reward = float(reward)
        next_obs = int(next_obs)

        accumulated_reward += reward

        # TODO do we need buffers?
        # store sample in replay buffer
        obs_buffer.append(obs)
        act_buffer.append(act)

        # direct RL
        q_table = q_learning_update(
            obs, act, reward, next_obs, gamma, learning_rate, q_table
        )

        counter = counter_update(counter, obs, act, reward, next_obs)
        model = model_update(model, counter, obs, act, next_obs)
        key, sampling_key = jax.random.split(key, 2)
        q_table = planning(
            model.transition,
            model.reward,
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
            if logger is not None:
                logger.record_stat("return", accumulated_reward, step=t)
            obs, _ = env.reset()
            accumulated_reward = 0.0

    return q_table
