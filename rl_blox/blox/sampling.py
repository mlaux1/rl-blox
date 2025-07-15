import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..logging.logger import LoggerBase
from .function_approximator.policy_head import StochasticPolicyBase
from .replay_buffer import EpisodeBuffer


def sample_trajectories(
    env: gym.Env,
    policy: StochasticPolicyBase,
    key: jnp.ndarray,
    logger: LoggerBase,
    train_after_episode: bool,
    total_steps: int,
) -> EpisodeBuffer:
    """Sample trajectories with stochastic policy.

    Parameters
    ----------
    env : gym.Env
        Environment in which we collect samples.

    policy : StochasticPolicyBase
        Policy from which we sample actions.

    key : array
        Pseudo random number generator key for action sampling.

    logger : Logger
        Logs average return.

    train_after_episode : bool
        Collect exactly one episode of samples.

    total_steps : int
        Collect a minimum of total_steps, but continues to the end of the
        episode.

    Returns
    -------
    dataset : EpisodeBuffer
        Collected samples organized in episodes.
    """
    if key is None:
        key = jax.random.key(0)

    dataset = EpisodeBuffer()
    dataset.start_episode()

    if logger is not None:
        logger.start_new_episode()

    @nnx.jit
    def sample(policy, observation, subkey):
        return policy.sample(observation, subkey)

    steps_per_episode = 0
    observation, _ = env.reset()

    while True:
        key, subkey = jax.random.split(key)
        action = np.asarray(sample(policy, jnp.array(observation), subkey))

        next_observation, reward, terminated, truncated, _ = env.step(action)

        steps_per_episode += 1
        done = terminated or truncated

        dataset.add_sample(observation, action, next_observation, reward)

        observation = next_observation

        if done:
            if logger is not None:
                logger.stop_episode(steps_per_episode)
                logger.start_new_episode()
            steps_per_episode = 0

            if train_after_episode or len(dataset) >= total_steps:
                break

            observation, _ = env.reset()
            dataset.start_episode()

    if logger is not None:
        logger.record_stat(
            "average return",
            dataset.average_return(),
            episode=logger.n_episodes - 1,
        )
    return dataset
