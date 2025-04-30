import gymnasium as gym

from rl_blox.algorithms.model_based.pets import create_pets_state, train_pets
from rl_blox.algorithms.model_based.pets_reward_models import pendulum_reward
from rl_blox.logging.logger import AIMLogger

env_name = "Pendulum-v1"
env = gym.make(env_name, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
seed = 1

hparams = dict(
    plan_horizon=25,
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
)

logger = AIMLogger()
logger.define_experiment(env_name, "PE-TS", hparams=hparams)

dynamics_model = create_pets_state(env, seed=seed)
mpc_config, mpc_state, optimizer_fn = train_pets(
    env,
    pendulum_reward,
    dynamics_model,
    logger=logger,
    **hparams,
)
env.close()
