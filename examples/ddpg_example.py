import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from flax import nnx

from rl_blox.algorithms.model_free.ddpg import (
    MLP,
    DeterministicPolicy,
    train_ddpg,
)

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)
policy_net = MLP(
    env.observation_space.shape[0],
    env.action_space.shape[0],
    [256, 256],
    nnx.Rngs(seed),
)
policy = DeterministicPolicy(policy_net, env.action_space)
q = MLP(
    env.observation_space.shape[0] + env.action_space.shape[0],
    1,
    [256, 256],
    nnx.Rngs(seed),
)
policy, policy_target, policy_optimizer, q, q_target, q_optimizer = train_ddpg(
    env,
    policy,
    q,
    gradient_steps=1,
    seed=seed,
    total_timesteps=31_000,
    verbose=2,
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
        action = np.asarray(policy(jnp.asarray(obs)[jnp.newaxis])[0])
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        q_value = q(jnp.concatenate((obs, action)))
        print(f"{q_value=}")
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
