import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.a2c import train_a2c
from rl_blox.algorithm.reinforce import create_policy_gradient_continuous_state
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

# env_name = "Pendulum-v1"
# env_name = "HalfCheetah-v4"
env_name = "InvertedPendulum-v5"
num_envs = 8
seed = 42
total_timesteps_for_test = 900_000
# Parallelization
envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make(env_name) for _ in range(num_envs)]
)
# env.reset(seed=seed)

hparams_model = dict(
    policy_shared_head=True,
    policy_hidden_nodes=[32, 32],
    policy_learning_rate=3e-4,
    value_network_hidden_nodes=[128, 128],
    value_network_learning_rate=1e-2,
    seed=seed,
)
hparams_algorithm = dict(
    total_timesteps=total_timesteps_for_test,
    gamma=0.99,
    steps_per_update=5_000,
    # train_after_episode=False,
    seed=seed,
    num_envs=num_envs,
    lmbda=0.95,
    value_loss_coef=0.5,
    entropy_coef=0.01,
)

# logger = StandardLogger(verbose=2)
logger = None


# logger = LoggerList([StandardLogger(verbose=2), AIMLogger()])
# logger.define_experiment(
#     env_name=env_name,
#     algorithm_name="A2C",
#     hparams=hparams_model | hparams_algorithm,
# )
# logger.define_checkpoint_frequency("value_function", 10)

ac_state = create_policy_gradient_continuous_state(envs, **hparams_model)

train_a2c(
    envs,
    ac_state.policy,
    ac_state.policy_optimizer,
    ac_state.value_function,
    ac_state.value_function_optimizer,
    **hparams_algorithm,
    logger=logger,
)
envs.close()

# Evaluation
# Using only one environment for evaluation and rendering
# env = gym.make(env_name, render_mode="human")
# env = gym.wrappers.RecordEpisodeStatistics(env)
# while True:
#     done = False
#     infos = {}
#     obs, _ = env.reset()
#     while not done:
#         action = np.asarray(ac_state.policy(jnp.asarray(obs)))
#         next_obs, reward, termination, truncation, infos = env.step(action)
#         done = termination or truncation
#         obs = np.asarray(next_obs)
#     if "final_info" in infos:
#         for info in infos["final_info"]:
#             print(f"episodic_return={info['episode']['r']}")
#             break
