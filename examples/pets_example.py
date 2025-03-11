import gymnasium as gym
import jax

from rl_blox.algorithms.model_based.pets import train_pets
from rl_blox.algorithms.model_based.pets_reward_models import pendulum_reward
from rl_blox.model.gaussian_mlp_ensemble import EnsembleOfGaussianMlps


env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
key = jax.random.PRNGKey(42)
dynamics_model = EnsembleOfGaussianMlps.create(
    env.observation_space.shape[0],  # TODO determine automatically
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
    verbose=10,
)
