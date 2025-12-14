import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.a2c import train_a2c
from rl_blox.algorithm.reinforce import create_policy_gradient_continuous_state
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

env_name = "InvertedPendulum-v5"
seed = 42
num_envs = 4

hparams_model = dict(
    policy_shared_head=True,
    policy_hidden_nodes=[32, 32],
    policy_learning_rate=3e-4,
    value_network_hidden_nodes=[128, 128],
    value_network_learning_rate=1e-2,
    seed=seed,
)

hparams_algorithm = dict(
    policy_gradient_steps=5,
    value_gradient_steps=5,
    total_timesteps=500_000,
    gamma=0.99,
    gae_lambda=0.95,
    steps_per_update=500,
    log_frequency=None,
    seed=seed,
)


def make_env():
    return gym.make(env_name)


envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

logger = LoggerList([StandardLogger(verbose=2), AIMLogger()])
logger.define_experiment(
    env_name=env_name,
    algorithm_name="A2C",
    hparams=hparams_model | hparams_algorithm,
)
logger.define_checkpoint_frequency("policy", 10)
logger.define_checkpoint_frequency("value_function", 10)

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

eval_env = gym.make(env_name, render_mode="human")
eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)

while True:
    done = False
    infos = {}
    obs, _ = eval_env.reset()
    while not done:
        action = np.asarray(ac_state.policy(jnp.asarray(obs)))
        next_obs, reward, termination, truncation, infos = eval_env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
eval_env.close()
