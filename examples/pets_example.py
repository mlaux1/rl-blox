import gymnasium as gym

from rl_blox.algorithms.model_based.pets import create_pets_state, train_pets
from rl_blox.algorithms.model_based.pets_reward_models import pendulum_reward

env_name = "Pendulum-v1"
env = gym.make(env_name, render_mode="human")
seed = 1

dynamics_model = create_pets_state(env, seed=seed)
mpc = train_pets(
    env,
    pendulum_reward,
    dynamics_model,
    planning_horizon=25,
    n_particles=20,
    n_samples=400,
    n_opt_iter=10,
    init_with_previous_plan=False,
    seed=seed,
    learning_starts=200,  # 200 steps = one episode
    learning_starts_gradient_steps=300,
    n_steps_per_iteration=200,  # 200 steps = one episode
    gradient_steps=10,
    total_timesteps=2_001,
    save_checkpoints=True,
    verbose=2,
)
env.close()
