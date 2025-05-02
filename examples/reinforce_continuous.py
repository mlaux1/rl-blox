import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.reinforce import (
    create_policy_gradient_continuous_state,
    train_reinforce,
)
from rl_blox.logging import logger

# env_name = "Pendulum-v1"
# env_name = "HalfCheetah-v4"
env_name = "InvertedPendulum-v5"
env = gym.make(env_name)
env.reset(seed=43)

hparams_model = dict(
    policy_shared_head=True,
    policy_hidden_nodes=[64, 64],
    policy_learning_rate=3e-4,
    value_network_hidden_nodes=[256, 256],
    value_network_learning_rate=1e-2,
    seed=42,
)
hparams_algorithm = dict(
    policy_gradient_steps=5,
    value_gradient_steps=5,
    total_timesteps=500_000,
    steps_per_update=5000,
    gamma=0.99,
    train_after_episode=False,
)

logger = logger.LoggerList(
    [logger.StandardLogger(verbose=2), logger.AIMLogger()]
)
logger.define_experiment(
    env_name=env_name,
    algorithm_name="REINFORCE",
    hparams=hparams_model | hparams_algorithm,
)

reinforce_state = create_policy_gradient_continuous_state(env, **hparams_model)

train_reinforce(
    env,
    reinforce_state.policy,
    reinforce_state.policy_optimizer,
    reinforce_state.value_function,
    reinforce_state.value_function_optimizer,
    **hparams_algorithm,
    key=reinforce_state.key,
    logger=logger,
)
env.close()

# Evaluation
env = gym.make(env_name, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
while True:
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        action = np.asarray(reinforce_state.policy(jnp.asarray(obs)))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
