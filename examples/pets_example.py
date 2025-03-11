import gymnasium as gym
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from rl_blox.algorithms.model_based.pets import train_pets
from rl_blox.model.gaussian_mlp_ensemble import EnsembleOfGaussianMlps

MAX_TORQUE: float = 2.0


def pendulum_reward(act: ArrayLike, obs: ArrayLike) -> jnp.ndarray:
    obs = jnp.asarray(obs)
    act = jnp.asarray(act)

    theta = obs[..., 0]
    theta_dot = obs[..., 1]
    act = jnp.clip(act, -MAX_TORQUE, MAX_TORQUE)[..., 0]

    costs = norm_angle(theta) ** 2 + 0.1 * theta_dot ** 2 + 0.001 * (act ** 2)
    return -costs


def norm_angle(angle: jnp.ndarray) -> jnp.ndarray:
    return ((angle + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi


env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
key = jax.random.PRNGKey(42)
dynamics_model = EnsembleOfGaussianMlps.create(
    env.action_space.shape[0],  # TODO determine automatically
    [500, 500, 500],
    5,
    key,
)
mpc = train_pets(
    env,
    pendulum_reward,
    dynamics_model,
    task_horizon=150,
    n_samples=20,
    seed=20,
    learning_starts=100,
    verbose=1,
)
