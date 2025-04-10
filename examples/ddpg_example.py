import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithms.model_free.ddpg import create_ddpg_state, train_ddpg

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)

ddpg_state = create_ddpg_state(
    env,
    policy_hidden_nodes=[256, 256],
    policy_learning_rate=3e-4,
    q_hidden_nodes=[256, 256],
    q_learning_rate=3e-4,
    seed=seed,
)

policy, policy_target, policy_optimizer, q, q_target, q_optimizer = train_ddpg(
    env,
    ddpg_state.policy,
    ddpg_state.policy_optimizer,
    ddpg_state.q,
    ddpg_state.q_optimizer,
    gradient_steps=1,
    seed=seed,
    total_timesteps=31_000,
    verbose=1,
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
        action = np.asarray(policy(jnp.asarray(obs)))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        q_value = q(jnp.concatenate((obs, action)))
        print(f"{q_value=}")
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
