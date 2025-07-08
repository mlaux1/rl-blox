import jax
import numpy as np
from flax import nnx
from gymnasium.vector import VectorEnv
from tqdm import trange

from ..logging.logger import LoggerBase


def train_a2c(
    envs: VectorEnv,
    actor: nnx.Module,
    critic: nnx.Module,
    t_max: int = 100,
    seed: int = 0,
    policy_gradient_steps: int = 1,
    vf_gradient_steps: int = 1,
    gamma: float = 0.9999,
    total_timesteps: int = 100_000,
    logger: LoggerBase | None = None,
) -> tuple[nnx.Module, nnx.Module]:
    """Advantage Actor-Critic (A2C)

    Parameters
    ----------
    envs : VectorEnv
        The environments to train in.
    actor : nnx.Module
        The policy network.
    critic : nnx.Module
        The value function estimation network.
    """

    key = jax.random.key(seed)
    rng = np.random.default_rng(seed)

    # initialise stuff
    obs, _ = envs.reset()
    terminations = [np.array([False, False])]
    truncations = [np.array([False, False])]
    rewards = []
    actions = []
    observations = []

    for i in trange(t_max):
        # for each episode
        # act and collect transitions in each env until termination
        acts = envs.action_space.sample()
        obs, rews, terms, truncs, _ = envs.step(acts)
        observations.append(obs)
        actions.append(acts)
        rewards.append(rews)
        terminations.append(np.logical_or(terms, terminations[-1]))
        truncations.append(np.logical_and(truncs, truncations[-1]))

        # backwards through gathered trajectories
        # compute advantage estimates and accumulate gradients

        # perform policy and value function update

    return
