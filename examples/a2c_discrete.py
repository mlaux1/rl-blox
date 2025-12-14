import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.a2c import train_a2c
from rl_blox.algorithm.reinforce import create_policy_gradient_discrete_state
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

env_name = "CartPole-v1"
seed = 42
num_envs = 4

hparams_model = dict(
    policy_hidden_nodes=[64, 64],
    policy_learning_rate=3e-4,
    value_network_hidden_nodes=[256, 256],
    value_network_learning_rate=1e-2,
    seed=seed,
)

hparams_algorithm = dict(
    policy_gradient_steps=20,
    value_gradient_steps=20,
    total_timesteps=100_000,
    gamma=0.99,
    gae_lambda=0.95,
    steps_per_update=1_000,
    log_frequency=5_000,
    seed=seed,
)


def make_env():
    return gym.make(env_name)


envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])

envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

logger = None

ac_state = create_policy_gradient_discrete_state(envs, **hparams_model)

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

eval_env = gym.make(env_name, render_mode="human")
eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)

while True:
    done = False
    infos = {}
    obs, _ = eval_env.reset()
    while not done:
        action = np.argmax(np.asarray(ac_state.policy(jnp.asarray(obs))))
        next_obs, reward, termination, truncation, infos = eval_env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
eval_env.close()
