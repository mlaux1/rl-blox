import jax
import numpy as np
from flax import nnx
from tqdm import trange

from ..logging.logger import LoggerBase


def train_a2c(
    env,
    policy: nnx.Module,
    policy_optimizer,
    value_function: nnx.Module,
    vf_optimizer,
    learning_rate,
    seed: int = 0,
    policy_gradient_steps: int = 1,
    vf_gradient_steps: int = 1,
    gamma: float = 0.9999,
    total_timesteps: float = 1e6,
    logger: LoggerBase | None = None,
) -> tuple[nnx.Module, nnx.Module]:
    """Advantage Actor-Critic (A2C)

    Parameters
    ----------
    env : gym.Env
        The environment to train in.
    """

    key = jax.random.key(seed)
    rng = np.random.default_rng(seed)

    # initialise stuff

    for i in trange(total_timesteps):
        print("Stepping")

        # for each episode
        # act and collect transitions in each env until termination

        # backwards through gathered trajectories
        # compute advantage estimates and accumulate gradients

        # perform policy and value function update

    policy = policy
    value_function = value_function

    return policy, value_function
