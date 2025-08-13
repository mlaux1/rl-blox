import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.multi_task.uts_sac import TaskSet, train_uts_sac
from rl_blox.algorithm.sac import create_sac_state

env_name = "Pendulum-v1"
seed = 1
verbose = 1

train_contexts = jnp.array([[10.0], [10.1], [9.9]])

train_envs = [
    gym.make(env_name, g=10.0),
    gym.make(env_name, g=10.1),
    gym.make(env_name, g=9.9),
]

train_set = TaskSet(train_contexts, train_envs)

hparams_models = dict(
    policy_hidden_nodes=[128, 128],
    policy_learning_rate=3e-4,
    q_hidden_nodes=[512, 512],
    q_learning_rate=1e-3,
    seed=seed,
)
hparams_algorithm = dict(
    total_timesteps=100_000,
    exploring_starts=0,
    episodes_per_task=1,
)

sac_state = create_sac_state(train_envs[0], **hparams_models)
sac_result = train_uts_sac(
    train_set,
    sac_state.policy,
    sac_state.policy_optimizer,
    sac_state.q,
    sac_state.q_optimizer,
    **hparams_algorithm,
)


for env in train_envs:
    train_envs[env].close()

policy, _, q, _, _, _, _ = sac_result

# Evaluation
env1 = gym.make(env_name, render_mode="human", g=10.0)
# env2 = gym.make(env_name, render_mode="human", g=9.9)
# env3 = gym.make(env_name, render_mode="human", g=10.0)
# env4 = gym.make(env_name, render_mode="human", g=10.5)

while True:
    env = env1
    done = False
    obs, _ = env.reset()
    while not done:
        action = np.asarray(policy(jnp.asarray(obs))[0])
        next_obs, reward, termination, truncation, info = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
