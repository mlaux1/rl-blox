import gymnasium as gym

from rl_blox.algorithm.sac import create_sac_state, train_sac
from rl_blox.algorithm.pets import create_pets_state, train_pets
from rl_blox.algorithm.pets_reward_models import pendulum_reward

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
verbose = 1
env = gym.wrappers.RecordEpisodeStatistics(env)

hparams_sac_model = dict(
    policy_hidden_nodes=[128, 128],
    policy_learning_rate=3e-4,
    q_hidden_nodes=[512, 512],
    q_learning_rate=1e-3,
    seed=seed,
)
hparams_sac = dict(
    total_timesteps=11_000,
    buffer_size=11_000,
    gamma=0.99,
    learning_starts=5_000,
)
hparams_pets = dict(
    plan_horizon=25,
    n_particles=20,
    n_samples=400,
    n_opt_iter=10,
    init_with_previous_plan=False,
    seed=seed,
    learning_starts=0,
    learning_starts_gradient_steps=300,
    n_steps_per_iteration=200,
    gradient_steps=10,
    total_timesteps=2_000,
)

if verbose:
    print("Using SAC to collect experience.")
sac_state = create_sac_state(env, **hparams_sac_model)
sac_result = train_sac(
    env,
    sac_state.policy,
    sac_state.policy_optimizer,
    sac_state.q,
    sac_state.q_optimizer,
    **hparams_sac,
)
env.close()
policy, _, q, _, _, _, replay_buffer = sac_result

env = gym.make(env_name, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
seed = 1

if verbose:
    print("Train PETS with SAC replay buffer.")
dynamics_model = create_pets_state(env, seed=seed)
mpc_config, mpc_state, optimizer_fn, _ = train_pets(
    env,
    pendulum_reward,
    dynamics_model,
    replay_buffer=replay_buffer,
    **hparams_pets,
)
env.close()
