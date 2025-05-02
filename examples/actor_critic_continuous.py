import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.actor_critic import train_ac
from rl_blox.algorithm.reinforce import create_policy_gradient_continuous_state
from rl_blox.logging import logger

# env_name = "Pendulum-v1"
# env_name = "HalfCheetah-v4"
env_name = "InvertedPendulum-v5"
env = gym.make(env_name)
env.reset(seed=43)

hparams_model = dict(
    policy_shared_head=True,
    policy_hidden_nodes=[32, 32],
    policy_learning_rate=3e-4,
    value_network_hidden_nodes=[128, 128],
    value_network_learning_rate=1e-2,
    seed=42,
)
hparams_algorithm = dict(
    policy_gradient_steps=5,
    value_gradient_steps=5,
    total_timesteps=900_000,
    gamma=0.99,
    steps_per_update=5_000,
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
logger.define_checkpoint_frequency("value_function", 10)

ac_state = create_policy_gradient_continuous_state(env, **hparams_model)

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
        action = np.asarray(ac_state.policy(jnp.asarray(obs)))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
