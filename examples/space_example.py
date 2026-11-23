from functools import partial

import gymnasium as gym
import numpy as np
import optax
from flax import nnx

from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.algorithm.space import train_space
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.multitask import DiscreteTaskSet
from rl_blox.blox.q_policy import greedy_policy
from rl_blox.blox.replay_buffer import ReplayBuffer
from rl_blox.blox.state_value import greedy_value
from rl_blox.logging.logger import AIMLogger

env_name = "CartPole-v1"
seed = 2
verbose = 1
gamma = 0.99


def set_context(env, context):
    # Tasks differ in the half-length of the pole
    env.unwrapped.length = context[0]
    env.unwrapped.polemass_length = env.unwrapped.masspole * context[0]


# Short, medium and long pole, as in the paper's motivating example
base_env = gym.make(env_name)
contexts = np.array([[0.25], [0.5], [1.0]])
train_set = DiscreteTaskSet(base_env, set_context, contexts, context_aware=True)

hparams_algorithm = dict(
    total_timesteps=100_000,
    episodes_per_task=1,
    # The paper's values for CartPole
    eta=0.025,
    kappa=1,
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
    algorithm_name="SPaCE-DDQN",
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
    replay_buffer=ReplayBuffer(buffer_size=50_000, discrete_actions=True),
    optimizer=nnx.Optimizer(q_net, optax.adam(0.003), wrt=nnx.Param),
    q_target_net=nnx.clone(q_net),
    gamma=gamma,
)

# SPaCE evaluates tasks with the start-state values of the Q-network
result = train_space(
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
