import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.td3 import create_td3_state, train_td3
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

env_name = "Hopper-v5"
env = gym.make(env_name)

seed = 1
verbose = 1
env = gym.wrappers.RecordEpisodeStatistics(env)

hparams_models = dict(
    policy_hidden_nodes=[256, 256],
    policy_learning_rate=1e-3,
    q_hidden_nodes=[256, 256],
    q_learning_rate=1e-3,
    seed=seed,
)
hparams_algorithm = dict(
    policy_delay=2,
    exploration_noise=0.2,
    noise_clip=0.5,
    gradient_steps=1,
    total_timesteps=1_000_000,
    buffer_size=1_000_000,
    learning_starts=25_000,
    batch_size=256,
    seed=seed,
)

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = LoggerList(
    [AIMLogger(), StandardLogger(checkpoint_dir=".td3_example", verbose=0)]
)
logger.define_experiment(
    env_name=env_name,
    algorithm_name="TD3",
    hparams=hparams_models | hparams_algorithm,
)
n_policy_epochs = (
    hparams_algorithm["total_timesteps"] - hparams_algorithm["learning_starts"]
) // hparams_algorithm["policy_delay"]
logger.define_checkpoint_frequency("policy", n_policy_epochs)

td3_state = create_td3_state(env, **hparams_models)

td3_result = train_td3(
    env,
    td3_state.policy,
    td3_state.policy_optimizer,
    td3_state.q,
    td3_state.q_optimizer,
    logger=logger,
    **hparams_algorithm,
)
env.close()
policy, _, _, q, _, _, _ = td3_result

# Evaluation
env = gym.make(env_name, render_mode="human")
returns = []
for i in range(10):
    done = False
    infos = {}
    obs, _ = env.reset(seed=i)
    accumulated_reward = 0.0
    while not done:
        action = np.asarray(policy(jnp.asarray(obs)))
        next_obs, reward, termination, truncation, infos = env.step(action)
        accumulated_reward += reward
        done = termination or truncation
        if verbose >= 2:
            q_value = float(
                q(jnp.concatenate((obs, action), axis=-1)).squeeze()
            )
            print(f"{q_value=:.3f}")
        obs = np.asarray(next_obs)
    returns.append(accumulated_reward)
env.close()
print(f"{returns=}")
print(
    f"{np.mean(returns)} +- {np.std(returns)}, "
    f"[min={min(returns)}, max={max(returns)}]"
)
