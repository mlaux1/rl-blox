import gymnasium
import jax
import tqdm
from jax.typing import ArrayLike

from ..blox.value_policy import get_epsilon_greedy_action, get_greedy_action
from ..logging.logger import LoggerBase
from ..util.error_functions import td_error


def train_q_learning(
    env: gymnasium.Env,
    q_table: ArrayLike,
    learning_rate: float = 0.1,
    epsilon: float = 0.05,
    gamma: float = 0.99,
    total_timesteps: int = 100_000,
    seed: int = 1,
    logger: LoggerBase | None = None,
) -> ArrayLike:
    key = jax.random.key(seed)

    if logger is not None:
        logger.start_new_episode()

    observation, _ = env.reset()
    steps_per_episode = 0

    for i in tqdm.trange(total_timesteps):
        steps_per_episode += 1
        key, subkey1, subkey2 = jax.random.split(key, 3)

        action = get_epsilon_greedy_action(
            subkey1, q_table, observation, epsilon
        )

        next_observation, reward, terminated, truncated, info = env.step(
            int(action)
        )

        next_action = get_greedy_action(subkey2, q_table, next_observation)

        q_table = _update_policy(
            q_table,
            observation,
            action,
            reward,
            next_observation,
            next_action,
            gamma,
            terminated,
            learning_rate,
        )

        if terminated or truncated:
            if logger is not None:
                logger.record_stat("return", info["episode"]["r"], step=i)
                logger.stop_episode(steps_per_episode)
            observation, _ = env.reset()
            steps_per_episode = 0
        else:
            observation = next_observation

    return q_table


@jax.jit
def _update_policy(
    q_table,
    observation,
    action,
    reward,
    next_observation,
    next_action,
    gamma,
    terminated,
    learning_rate,
):
    val = q_table[observation, action]
    next_val = (1 - terminated) * q_table[next_observation, next_action]
    error = td_error(reward, gamma, val, next_val)
    q_table = q_table.at[observation, action].add(learning_rate * error)

    return q_table
