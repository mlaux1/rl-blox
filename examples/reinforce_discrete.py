import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from rl_blox.algorithms.model_free.reinforce import (
    create_policy_gradient_discrete_state,
    train_reinforce_epoch,
)
from rl_blox.logging import logger

env_name = "CartPole-v1"
# env_name = "MountainCar-v0"  # never reaches the goal -> never learns
env = gym.make(env_name)
env.reset(seed=42)

hparams = dict(
    policy_hidden_nodes=[64, 64],
    policy_learning_rate=1e-4,
    value_network_hidden_nodes=[256, 256],
    value_network_learning_rate=1e-2,
    seed=42,
)

logger = logger.LoggerList(
    [logger.StandardLogger(verbose=2), logger.AIMLogger()]
)
logger.define_experiment(
    env_name=env_name, algorithm_name="REINFORCE", hparams=hparams
)

reinforce_state = create_policy_gradient_discrete_state(env, **hparams)

n_epochs = 100
key = reinforce_state.key
for _ in tqdm.trange(n_epochs):
    key, subkey = jax.random.split(key, 2)
    train_reinforce_epoch(
        env,
        reinforce_state.policy,
        reinforce_state.policy_optimizer,
        reinforce_state.value_function,
        reinforce_state.value_function_optimizer,
        policy_gradient_steps=20,
        value_gradient_steps=20,
        total_steps=1000,
        gamma=1.0,
        train_after_episode=False,
        key=subkey,
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
        action = np.argmax(np.asarray(reinforce_state.policy(jnp.asarray(obs))))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
