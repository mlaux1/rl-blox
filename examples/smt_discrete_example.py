from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.algorithm.nature_dqn import train_nature_dqn
from rl_blox.algorithm.smt import train_smt
from rl_blox.blox.embedding.task_embedding import MTMLPQNetwork
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.multitask import DiscreteTaskSet
from rl_blox.blox.replay_buffer import MultiTaskReplayBuffer, ReplayBuffer
from rl_blox.logging.logger import AIMLogger

env_name = "MountainCar-v0"


def set_context(env, context):
    env.unwrapped.goal_velocity = context


base_env = gym.make(env_name)
contexts = np.linspace(0, 0.3, 11)[:, np.newaxis]

train_set = DiscreteTaskSet(base_env, set_context, contexts, context_aware=True)

seed = 2
verbose = 2
# Backbone algorithm to use for SMT: "DDQN", "NDQN"
backbone = "DDQN"
context_in_observation = False

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = AIMLogger()
logger.define_experiment(
    env_name="MountainCar-v0",
    algorithm_name=f"SMT-{backbone}",
    hparams={},
)

env = train_set.get_task(0)
if context_in_observation:
    q_net = MLP(
        n_features=env.observation_space.shape[0],
        n_outputs=int(env.action_space.n),
        activation="relu",
        hidden_nodes=[128, 128],
        rngs=nnx.Rngs(seed),
    )
else:
    q_net = MTMLPQNetwork(
        n_tasks=len(train_set),
        task_embedding_dim=10,
        n_features=env.observation_space.shape[0],
        n_outputs=int(env.action_space.n),
        activation="relu",
        hidden_nodes=[128, 128],
        rngs=nnx.Rngs(seed),
    )
q_target_net = nnx.clone(q_net)
replay_buffer = MultiTaskReplayBuffer(
    ReplayBuffer(buffer_size=100_000, discrete_actions=True),
    len(train_set),
)
optimizer = nnx.Optimizer(q_net, optax.adam(0.003), wrt=nnx.Param)
if backbone == "DDQN":
    train_st = partial(
        train_ddqn,
        q_net=q_net,
        optimizer=optimizer,
        q_target_net=q_target_net,
    )
elif backbone == "NDQN":
    train_st = partial(
        train_nature_dqn,
        q_net=q_net,
        optimizer=optimizer,
        q_target_net=q_target_net,
    )
else:
    raise NotImplementedError(f"Unknown backbone '{backbone}'")

result = train_smt(
    train_set,
    train_st,
    replay_buffer,
    solved_threshold=-110.0,
    unsolvable_threshold=-200.0,
    task_selectables=None if context_in_observation else [q_net],
    b1=170_000,
    b2=30_000,
    learning_starts=0,
    scheduling_interval=20,
    logger=logger,
    seed=seed,
)

# Evaluation
result_st = result[0]
q_net = result_st.q_net


def policy(obs):
    return int(jnp.argmax(q_net(obs)))


base_env = gym.make(env_name, render_mode="human")
test_set = DiscreteTaskSet(base_env, set_context, contexts, context_aware=True)

for task_id in range(len(test_set)):
    print(f"Evaluating task {task_id}")
    env = test_set.get_task(task_id)
    if not context_in_observation:
        q_net.select_task(task_id)
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        action = policy(obs)
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
