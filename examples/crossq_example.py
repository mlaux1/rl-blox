import numpy as np
import gymnasium as gym
from rl_blox.algorithms.model_free.ext.crossq import train_crossq


env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)
model = train_crossq(
    env,
    total_timesteps=15_000,
    seed=seed,
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
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = next_obs
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
