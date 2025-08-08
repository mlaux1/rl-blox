import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.a2c import train_a2c
from rl_blox.algorithm.reinforce import create_policy_gradient_continuous_state
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

env_name = "Pendulum-v1"
# env_name = "HalfCheetah-v4"
# env_name = "InvertedPendulum-v5"
env1 = gym.make(env_name, g=9.81)
env2 = gym.make(env_name, g=9.9)
env_set = [env1, env2]

seed = 42

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
    total_timesteps=900_000,
    gamma=0.99,
    steps_per_update=5_000,
    train_after_episode=False,
    seed=seed,
)

logger = LoggerList([StandardLogger(verbose=2), AIMLogger()])
logger.define_experiment(
    env_name=env_name,
    algorithm_name="A2C",
    hparams=hparams_model | hparams_algorithm,
)
logger.define_checkpoint_frequency("value_function", 10)

a2c_state = create_policy_gradient_continuous_state(env1, **hparams_model)

train_a2c(
    env_set,
    a2c_state.policy,
    a2c_state.policy_optimizer,
    a2c_state.value_function,
    a2c_state.value_function_optimizer,
    **hparams_algorithm,
    logger=logger,
)

env1.close()
env2.close()

# Evaluation
env = gym.make(env_name, render_mode="human", g=9.81)
env = gym.wrappers.RecordEpisodeStatistics(env)
while True:
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        action = np.asarray(a2c_state.policy(jnp.asarray(obs)))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
