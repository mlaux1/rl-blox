import gymnasium as gym
import numpy as np

from modular_rl.algorithms.model_free.ddpg import train_ddpg, DeterministicMlpPolicyNetwork, MlpQNetwork


env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)
envs = gym.vector.SyncVectorEnv([lambda: env])
policy = DeterministicMlpPolicyNetwork.create([256, 256], envs)
q = MlpQNetwork(hidden_nodes=[256, 256])
policy, policy_params, q, q_params = train_ddpg(
    envs,
    policy,
    q,
    gradient_steps=1,
    seed=seed,
    total_timesteps=31_000,
    verbose=1
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
        action = np.asarray(policy.apply(policy_params, obs)[0])
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        q_value = q.apply(q_params, obs, action)
        print(f"{q_value=}")
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
