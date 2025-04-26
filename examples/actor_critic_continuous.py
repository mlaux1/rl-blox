import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from rl_blox.algorithms.model_free.actor_critic import (
    train_ac_epoch,
)
from rl_blox.algorithms.model_free.reinforce import (
    create_policy_gradient_continuous_state,
)
from rl_blox.logging import logger

# env_name = "Pendulum-v1"
# env_name = "HalfCheetah-v4"
env_name = "InvertedPendulum-v5"
env = gym.make(env_name)
env.reset(seed=43)

hparams = dict(
    policy_shared_head=True,
    policy_hidden_nodes=[32, 32],
    policy_learning_rate=3e-4,
    value_network_hidden_nodes=[50, 50],
    value_network_learning_rate=1e-2,
    seed=42,
)

logger = logger.Logger(use_aim=True, verbose=2)
logger.define_experiment(
    env_name=env_name, algorithm_name="Actor-Critic", hparams=hparams
)
logger.define_checkpoint_frequency("value_function", 10)

ac_state = create_policy_gradient_continuous_state(env, **hparams)

n_epochs = 375
key = ac_state.key
for _ in tqdm.trange(n_epochs):
    key, subkey = jax.random.split(key, 2)
    train_ac_epoch(
        env,
        ac_state.policy,
        ac_state.policy_optimizer,
        ac_state.value_function,
        ac_state.value_function_optimizer,
        policy_gradient_steps=5,
        value_gradient_steps=5,
        total_steps=5000,
        gamma=0.99,
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
        action = np.asarray(ac_state.policy(jnp.asarray(obs)))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
