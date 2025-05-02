import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.actor_critic import train_ac
from rl_blox.algorithm.reinforce import create_policy_gradient_discrete_state
from rl_blox.logging import logger

env_name = "CartPole-v1"
# env_name = "MountainCar-v0"  # never reaches the goal -> never learns
env = gym.make(env_name)
env.reset(seed=42)

hparams_model = dict(
    policy_hidden_nodes=[64, 64],
    policy_learning_rate=3e-4,
    value_network_hidden_nodes=[256, 256],
    value_network_learning_rate=1e-2,
    seed=42,
)
hparams_algorithm = dict(
    policy_gradient_steps=20,
    value_gradient_steps=20,
    total_timesteps=100_000,
    gamma=1.0,
    steps_per_update=1_000,
    train_after_episode=False,
)

logger = logger.LoggerList(
    [logger.StandardLogger(verbose=2), logger.AIMLogger()]
)
logger.define_experiment(
    env_name=env_name,
    algorithm_name="Actor-Critic",
    hparams=hparams_model | hparams_algorithm,
)

ac_state = create_policy_gradient_discrete_state(env, **hparams_model)

train_ac(
    env,
    ac_state.policy,
    ac_state.policy_optimizer,
    ac_state.value_function,
    ac_state.value_function_optimizer,
    key=ac_state.key,
    logger=logger,
    **hparams_algorithm,
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
        action = np.argmax(np.asarray(ac_state.policy(jnp.asarray(obs))))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
