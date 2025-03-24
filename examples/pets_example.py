import gymnasium as gym
import optax
from flax import nnx

from rl_blox.algorithms.model_based.pets import train_pets
from rl_blox.algorithms.model_based.pets_reward_models import pendulum_reward
from rl_blox.model.probabilistic_ensemble import (
    EnsembleTrainState,
    GaussianMLPEnsemble,
)


env_name = "Pendulum-v1"
env = gym.make(env_name, render_mode="human")
seed = 1
model = GaussianMLPEnsemble(  # TODO refactor initialization
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
    task_horizon=150,
    n_particles=20,
    n_samples=400,
    batch_size=256,  # TODO batch size to sample from replay buffer
    n_opt_iter=5,
    seed=seed,
    learning_starts=600,  # 200 steps = one episode
    learning_starts_gradient_steps=200,
    n_steps_per_iteration=200,  # 200 steps = one episode
    gradient_steps=50,
    total_timesteps=800,
    verbose=20,
)
env.close()
