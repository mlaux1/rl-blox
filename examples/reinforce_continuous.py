import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from rl_blox.algorithms.model_free.reinforce_flax import (
    create_reinforce_continuous_state,
    train_reinforce_epoch,
)

# env_name = "Pendulum-v1"
# env_name = "HalfCheetah-v4"
env_name = "InvertedPendulum-v5"
env = gym.make(env_name)
env.reset(seed=43)

reinforce_state = create_reinforce_continuous_state(
    env,
    policy_shared_head=True,
    policy_hidden_nodes=[16, 32],
    policy_learning_rate=1e-4,
    value_network_hidden_nodes=[50, 50],
    value_network_learning_rate=1e-2,
    seed=42,
)

n_epochs = 50
for i in range(n_epochs):
    print(f"Epoch #{i + 1}")
    train_reinforce_epoch(
        env,
        reinforce_state.policy,
        reinforce_state.policy_optimizer,
        reinforce_state.value_function,
        reinforce_state.value_function_optimizer,
        policy_gradient_steps=50,
        value_gradient_steps=50,
        total_steps=1000,
        gamma=0.99,
        train_after_episode=False,
        verbose=2,
    )

# Evaluation
env = gym.make(env_name, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
while True:
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        mean_action, _ = reinforce_state.policy(jnp.asarray(obs))
        action = np.asarray(mean_action)
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
