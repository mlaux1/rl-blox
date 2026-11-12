from functools import partial

import gymnasium as gym
import jax
import optax
import pytest
from flax import nnx

from rl_blox.algorithm.actor_critic import train_ac
from rl_blox.algorithm.cmaes import train_cmaes
from rl_blox.algorithm.ddpg import create_ddpg_state, train_ddpg
from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.algorithm.dqn import train_dqn
from rl_blox.algorithm.nature_dqn import train_nature_dqn
from rl_blox.algorithm.per import train_ddqn_per
from rl_blox.algorithm.reinforce import (
    create_policy_gradient_continuous_state,
    train_reinforce,
)
from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac
from rl_blox.algorithm.td3 import create_td3_state, train_td3
from rl_blox.algorithm.td3_lap import train_td3_lap
from rl_blox.algorithm.td7 import create_td7_state, train_td7
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import (
    DeterministicTanhPolicy,
)
from rl_blox.blox.replay_buffer import (
    LAP,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)

compilations = []


def record_compilation(event, duration, **kwargs):
    if event == "/jax/core/compile/backend_compile_duration":
        compilations.append(event)


jax.monitoring.register_event_duration_secs_listener(record_compilation)


def discrete(train, buffer, **kwargs):
    env = gym.make("CartPole-v1")
    q_net = MLP(4, 2, [8], "relu", nnx.Rngs(0))
    optimizer = nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param)
    replay_buffer = buffer(100, discrete_actions=True)
    return partial(
        train,
        q_net,
        env,
        replay_buffer,
        optimizer,
        batch_size=8,
        total_timesteps=50,
        progress_bar=False,
        **kwargs,
    )


def continuous(train, create, buffer, **kwargs):
    env = gym.make("Pendulum-v1")
    state = create(env, policy_hidden_nodes=[8], q_hidden_nodes=[8], seed=0)
    return partial(
        train,
        env,
        *state,
        batch_size=8,
        total_timesteps=60,
        learning_starts=20,
        replay_buffer=buffer(100),
        progress_bar=False,
        **kwargs,
    )


def sac():
    env = gym.make("Pendulum-v1")
    state = create_sac_state(
        env, policy_hidden_nodes=[8], q_hidden_nodes=[8], seed=0
    )
    return partial(
        train_sac,
        env,
        *state,
        batch_size=8,
        total_timesteps=60,
        learning_starts=20,
        replay_buffer=ReplayBuffer(100),
        # Passed on between calls like in the task schedulers' examples
        entropy_control=EntropyControl(env, 0.2, True, 1e-3),
        progress_bar=False,
    )


def td7():
    env = gym.make("Pendulum-v1")
    state = create_td7_state(
        env,
        n_embedding_dimensions=8,
        state_embedding_hidden_nodes=[8],
        state_action_embedding_hidden_nodes=[8],
        policy_sa_encoding_nodes=8,
        policy_hidden_nodes=[8],
        q_sa_encoding_nodes=8,
        q_hidden_nodes=[8],
        seed=0,
    )
    return partial(
        train_td7,
        env,
        *state,
        batch_size=8,
        total_timesteps=60,
        learning_starts=20,
        target_delay=10,
        use_checkpoints=False,
        replay_buffer=LAP(100),
        progress_bar=False,
    )


def policy_gradient(train):
    # Pendulum episodes have a fixed length: batches of different lengths
    # would cause recompilations of the updates
    env = gym.make("Pendulum-v1")
    state = create_policy_gradient_continuous_state(
        env, policy_hidden_nodes=[8], value_network_hidden_nodes=[8], seed=0
    )
    return partial(
        train,
        env,
        *state,
        total_timesteps=400,
        steps_per_update=200,
        progress_bar=False,
    )


def cmaes():
    env = gym.make("Pendulum-v1")
    policy = DeterministicTanhPolicy(
        MLP(3, 1, [8], "relu", rngs=nnx.Rngs(0)), env.action_space
    )
    return partial(
        train_cmaes,
        env,
        policy,
        total_episodes=4,
        seed=0,
        n_samples_per_update=4,
        progress_bar=False,
    )


CASES = {
    "DQN": lambda: discrete(train_dqn, ReplayBuffer),
    "DDQN": lambda: discrete(
        train_ddqn, ReplayBuffer, target_update_frequency=10
    ),
    "Nature DQN": lambda: discrete(
        train_nature_dqn, ReplayBuffer, target_update_frequency=10
    ),
    "PER": lambda: discrete(
        train_ddqn_per, PrioritizedReplayBuffer, target_update_frequency=10
    ),
    "DDPG": lambda: continuous(train_ddpg, create_ddpg_state, ReplayBuffer),
    "TD3": lambda: continuous(train_td3, create_td3_state, ReplayBuffer),
    "TD3-LAP": lambda: continuous(train_td3_lap, create_td3_state, LAP),
    "SAC": sac,
    "TD7": td7,
    "REINFORCE": lambda: policy_gradient(train_reinforce),
    "actor-critic": lambda: policy_gradient(train_ac),
    "CMA-ES": cmaes,
}


@pytest.mark.parametrize("make_train", CASES.values(), ids=CASES.keys())
def test_second_call_compiles_nothing(make_train):
    # The task schedulers call train_* once per episode
    train = make_train()
    train()
    n_compilations = len(compilations)

    train()

    assert len(compilations) == n_compilations
