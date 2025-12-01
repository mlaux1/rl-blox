import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.a2c import train_a2c
from rl_blox.algorithm.reinforce import create_policy_gradient_discrete_state

env_name = "CartPole-v1"
num_envs = 8
seed = 42
total_timesteps_for_test = 50_000

envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make(env_name) for _ in range(num_envs)]
)

hparams_model = dict(
    policy_hidden_nodes=[64, 64],
    policy_learning_rate=1.5e-2,
    value_network_hidden_nodes=[64, 64],
    value_network_learning_rate=2.5e-5,
    seed=seed,
)
hparams_algorithm = dict(
    total_timesteps=total_timesteps_for_test,
    gamma=0.99,
    steps_per_update=128,
    num_envs=num_envs,
    lmbda=0.95,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    seed=seed,
)

logger = None

ac_state = create_policy_gradient_discrete_state(envs, **hparams_model)

train_a2c(
    envs,
    ac_state.policy,
    ac_state.policy_optimizer,
    ac_state.value_function,
    ac_state.value_function_optimizer,
    **hparams_algorithm,
    logger=logger,
)
envs.close()

# eval_env = gym.make(env_name, render_mode="human")
# while True:
#     done = False
#     obs, _ = eval_env.reset()
#     while not done:
#         logits = ac_state.policy(jnp.asarray(obs))
#         action = np.asarray(jnp.argmax(logits, axis=-1))

#         next_obs, reward, termination, truncation, _ = eval_env.step(action)
#         done = termination or truncation
#         obs = np.asarray(next_obs)
