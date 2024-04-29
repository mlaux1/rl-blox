import gymnasium
import jax.numpy as jnp
from jax import Array, jit, random
from jax.random import PRNGKey
from jax.typing import ArrayLike
from tqdm import tqdm

from ...policy.value_policy import get_epsilon_greedy_action, get_greedy_action
from ...tools.error_functions import td_error


def double_q_learning(
        key: PRNGKey,
        env: gymnasium.Env,
        q_table1: ArrayLike,
        q_table2: ArrayLike,
        alpha: float,
        epsilon: float,
        num_episodes: int,
        gamma: float = 0.9999,

) -> Array:

    ep_rewards = jnp.zeros(num_episodes)

    for i in tqdm(range(num_episodes)):
        key, subkey = random.split(key)
        q_table, ep_reward = _q_learning_episode(
            subkey, env, q_table1, q_table2, alpha, epsilon, gamma)
        ep_rewards = ep_rewards.at[i].add(ep_reward)

    return q_table, ep_rewards


def _q_learning_episode(
        key: PRNGKey,
        env: gymnasium.Env,
        q_table1: ArrayLike,
        q_table2: ArrayLike,
        alpha: float,
        epsilon: float,
        gamma: float = 0.9999,
) -> float:
    raise NotImplementedError
