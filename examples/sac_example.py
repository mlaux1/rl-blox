import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithms.model_free.sac import (
    create_sac_state,
    train_sac,
)

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)

sac_state = create_sac_state(
    env,
    policy_hidden_nodes=[256, 256],
    policy_learning_rate=3e-4,
    q_hidden_nodes=[256, 256],
    q_learning_rate=1e-3,
    seed=seed,
)
sac_result = train_sac(
    env,
    sac_state.policy,
    sac_state.policy_optimizer,
    sac_state.q1,
    sac_state.q1_optimizer,
    sac_state.q2,
    sac_state.q2_optimizer,
    total_timesteps=10_000,
    buffer_size=1_000_000,
    gamma=0.99,
    learning_starts=5_000,
    verbose=1,
)
policy, _, q1, _, _, q2, _, _, _ = sac_result

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
        q1_value = q1(jnp.concatenate((obs, action), axis=-1))
        q2_value = q2(jnp.concatenate((obs, action), axis=-1))
        q_value = np.minimum(q1_value, q2_value)
        print(f"{q_value=}")
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
