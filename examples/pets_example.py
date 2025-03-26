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
    optimizer=nnx.Optimizer(
        model, optax.adamw(learning_rate=1e-3, weight_decay=0.001)
    ),
    train_size=0.7,
    batch_size=32,
    regularization=0.0,
)
mpc = train_pets(
    env,
    pendulum_reward,
    dynamics_model,
    task_horizon=80,
    n_particles=20,
    n_samples=400,
    n_opt_iter=10,
    init_with_previous_plan=False,
    seed=seed,
    learning_starts=600,  # 200 steps = one episode
    learning_starts_gradient_steps=300,
    n_steps_per_iteration=200,  # 200 steps = one episode
    gradient_steps=10,
    total_timesteps=2_001,
    save_checkpoints=True,
    verbose=15,
)
env.close()
