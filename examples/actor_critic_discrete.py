import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithms.model_free.actor_critic import (
    train_ac_epoch,
)
from rl_blox.algorithms.model_free.reinforce import (
    create_policy_gradient_discrete_state,
)

env_name = "CartPole-v1"
# env_name = "MountainCar-v0"  # never reaches the goal -> never learns
env = gym.make(env_name)
env.reset(seed=42)

ac_state = create_policy_gradient_discrete_state(
    env,
    policy_hidden_nodes=[32],
    policy_learning_rate=1e-4,
    value_network_hidden_nodes=[100, 100],
    value_network_learning_rate=1e-2,
    seed=42,
)

n_epochs = 100
for i in range(n_epochs):
    print(f"Epoch #{i + 1}")
    train_ac_epoch(
        env,
        ac_state.policy,
        ac_state.policy_optimizer,
        ac_state.value_function,
        ac_state.value_function_optimizer,
        policy_gradient_steps=10,
        value_gradient_steps=50,
        total_steps=5000,
        gamma=1.0,
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
        action = np.argmax(np.asarray(ac_state.policy(jnp.asarray(obs))))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
