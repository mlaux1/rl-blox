from functools import partial

import gymnasium as gym
import numpy as np
import optax
from flax import nnx

from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.algorithm.plr import train_plr
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.multitask import DiscreteTaskSet
from rl_blox.blox.q_policy import greedy_policy
from rl_blox.blox.replay_buffer import ReplayBuffer
from rl_blox.blox.state_value import greedy_value
from rl_blox.logging.logger import AIMLogger

env_name = "MountainCar-v0"
seed = 2
verbose = 1
gamma = 0.99


def set_context(env, context):
    env.unwrapped.goal_velocity = context[0]


base_env = gym.make(env_name)
contexts = np.linspace(0, 0.3, 11)[:, np.newaxis]
train_set = DiscreteTaskSet(base_env, set_context, contexts, context_aware=True)

hparams_algorithm = dict(
    total_timesteps=200_000,
    episodes_per_task=1,
    warmup_episodes=1,
    beta=0.1,
    rho=0.1,
    gamma=gamma,
    gae_lambda=0.95,
    seed=seed,
)

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="PLR-DDQN",
    hparams=hparams_algorithm,
)

env = train_set.get_task(0)
q_net = MLP(
    env.observation_space.shape[0],
    int(env.action_space.n),
    activation="relu",
    hidden_nodes=[128, 128],
    rngs=nnx.Rngs(seed),
)
train_st = partial(
    train_ddqn,
    q_net=q_net,
    replay_buffer=ReplayBuffer(buffer_size=100_000, discrete_actions=True),
    optimizer=nnx.Optimizer(q_net, optax.adam(0.003), wrt=nnx.Param),
    q_target_net=nnx.clone(q_net),
    gamma=gamma,
)

# PLR scores tasks with the state values of the Q-network that DDQN trains
result = train_plr(
    train_set,
    train_st,
    partial(greedy_value, q_net),
    **hparams_algorithm,
    logger=logger,
)

# Evaluation
q_net = result.result.q_net
base_env = gym.make(env_name, render_mode="human")
test_set = DiscreteTaskSet(base_env, set_context, contexts, context_aware=True)

for task_id in range(len(test_set)):
    print(f"Evaluating task {task_id}")
    env = test_set.get_task(task_id)
    done = False
    obs, _ = env.reset()
    while not done:
        action = int(greedy_policy(q_net, obs))
        obs, reward, termination, truncation, info = env.step(action)
        done = termination or truncation
