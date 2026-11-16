import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx
from numpy.testing import assert_allclose

from rl_blox.algorithm.mrq import create_mrq_state
from rl_blox.algorithm.sac import create_sac_state
from rl_blox.algorithm.td3 import create_td3_state
from rl_blox.blox.embedding.model_based_encoder import (
    policy_with_encoder_value,
)
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.state_value import greedy_value, policy_value

N_SAMPLES = 5


def make_observations(n_features):
    return jax.random.normal(jax.random.key(0), (N_SAMPLES, n_features))


def test_greedy_value_is_maximum_q_value():
    q_net = MLP(4, 3, [8], "relu", rngs=nnx.Rngs(0))
    observation = make_observations(4)

    value = greedy_value(q_net, observation)

    assert value.shape == (N_SAMPLES,)
    assert_allclose(value, jnp.max(q_net(observation), axis=-1), rtol=1e-5)


def test_policy_value_of_deterministic_policy():
    env = gym.make("Pendulum-v1")
    state = create_td3_state(env, policy_hidden_nodes=[8], q_hidden_nodes=[8])
    observation = make_observations(3)

    value = policy_value(state.q, state.policy, observation)

    action = state.policy(observation)
    expected = state.q(jnp.concatenate((observation, action), axis=-1))
    assert value.shape == (N_SAMPLES,)
    assert_allclose(value, expected[:, 0], rtol=1e-5)


def test_policy_value_uses_mean_action_of_gaussian_policy():
    env = gym.make("Pendulum-v1")
    state = create_sac_state(env, policy_hidden_nodes=[8], q_hidden_nodes=[8])
    observation = make_observations(3)

    value = policy_value(state.q, state.policy, observation)

    mean_action, _ = state.policy(observation)
    expected = state.q(jnp.concatenate((observation, mean_action), axis=-1))
    assert value.shape == (N_SAMPLES,)
    assert_allclose(value, expected[:, 0], rtol=1e-5)


def test_policy_with_encoder_value():
    env = gym.make("Pendulum-v1")
    state = create_mrq_state(
        env,
        policy_hidden_nodes=[8],
        q_hidden_nodes=[8],
        encoder_zs_dim=8,
        encoder_za_dim=8,
        encoder_zsa_dim=8,
        encoder_hidden_nodes=[8],
    )
    observation = make_observations(3)

    value = policy_with_encoder_value(
        state.q, state.policy_with_encoder, observation
    )

    encoder = state.policy_with_encoder.encoder
    zs = encoder.encode_zs(observation)
    zsa = encoder.encode_zsa(zs, state.policy_with_encoder.policy(zs))
    assert value.shape == (N_SAMPLES,)
    assert_allclose(value, state.q(zsa)[:, 0], rtol=1e-5)
