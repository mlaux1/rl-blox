import numpy as np
import gymnasium as gym
from modular_rl.algorithms.model_free.sac import train_sac, GaussianMlpPolicyNetwork, SoftMlpQNetwork, mean_actions

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)
envs = gym.vector.SyncVectorEnv([lambda: env])
policy = GaussianMlpPolicyNetwork.create([256, 256], envs)
q = SoftMlpQNetwork(hidden_nodes=[256, 256])
policy, policy_params, q, q1_params, q2_params = train_sac(
    envs,
    policy,
    q,
    seed=seed,
    total_timesteps=8_000,
    buffer_size=1_000_000,
    gamma=0.99,
    learning_starts=5_000
)
envs.close()

# Evaluation
env = gym.make(env_name, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
while True:
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        action = np.asarray(mean_actions(policy, policy_params, obs)[0])
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
