import gymnasium as gym
import jax
import optax
from flax import nnx

from rl_blox.algorithms.model_based.pets import train_pets
from rl_blox.algorithms.model_based.pets_reward_models import pendulum_reward
from rl_blox.model.probabilistic_ensemble import EnsembleTrainState, GaussianMlpEnsemble


env_name = "Pendulum-v1"
env = gym.make(env_name, render_mode="human")
seed = 1
model = GaussianMlpEnsemble(  # TODO refactor initialization
    n_ensemble=5,
    n_features=env.observation_space.shape[0] + env.action_space.shape[0],
    n_outputs=env.observation_space.shape[0],
    shared_head=True,
    hidden_nodes=[500, 500, 500],
    rngs=nnx.Rngs(seed),
)
dynamics_model = EnsembleTrainState(
    model=model,
    optimizer=nnx.Optimizer(model, optax.adam(learning_rate=1e-3)),
    train_size=0.7,
    batch_size=32,
)
mpc = train_pets(
    env,
    pendulum_reward,
    dynamics_model,
    task_horizon=50,
    n_samples=400,
    n_opt_iter=20,
    seed=seed,
    learning_starts=500,
    verbose=20,
)
