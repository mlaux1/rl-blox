import gymnasium as gym
import numpy as np
from flax import nnx

from rl_blox.algorithms.model_free.ddpg import MLP
from rl_blox.algorithms.model_free.sac import (
    GaussianMlpPolicyNetwork,
    mean_action,
    train_sac,
)

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)

policy = GaussianMlpPolicyNetwork.create([256, 256], env)
q = MLP(env.observation_space.shape[0] + env.action_space.shape[0], 1, [256, 256], nnx.Rngs(seed))
policy, policy_params, q, q1_params, q2_params = train_sac(
    env,
    policy,
    q,
    seed=seed,
    total_timesteps=8_000,
    buffer_size=1_000_000,
    gamma=0.99,
    learning_starts=5_000,
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
        action = np.asarray(mean_action(policy, policy_params, obs)[0])
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        q1_value = q.apply(q1_params, obs, action)
        q2_value = q.apply(q2_params, obs, action)
        q_value = np.minimum(q1_value, q2_value)
        print(f"{q_value=}")
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
