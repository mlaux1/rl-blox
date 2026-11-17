import numpy as np
from numpy.testing import assert_allclose

from rl_blox.algorithm.mrq import create_mrq_state, mrq_value, train_mrq
from rl_blox.blox.embedding.model_based_encoder import (
    policy_with_encoder_value,
)
from rl_blox.blox.replay_buffer import SubtrajectoryReplayBufferPER


def test_td7(pendulum_env):
    seed = 0
    state = create_mrq_state(pendulum_env, seed=seed)
    train_mrq(
        pendulum_env,
        state.policy_with_encoder,
        state.encoder_optimizer,
        state.policy_optimizer,
        state.q,
        state.q_optimizer,
        state.the_bins,
        total_timesteps=10,
        seed=seed,
    )


def test_mrq_value_has_units_of_rewards(pendulum_env):
    state = create_mrq_state(
        pendulum_env,
        policy_hidden_nodes=[8],
        q_hidden_nodes=[8],
        encoder_zs_dim=8,
        encoder_za_dim=8,
        encoder_zsa_dim=8,
        encoder_hidden_nodes=[8],
    )
    replay_buffer = SubtrajectoryReplayBufferPER(10, horizon=1)
    observation = np.random.default_rng(0).normal(size=(3, 3))
    scaled_value = policy_with_encoder_value(
        state.q, state.policy_with_encoder, observation
    )

    # Reward scale 1 while the buffer is empty, as in train_mrq
    value = mrq_value(
        state.q, state.policy_with_encoder, replay_buffer, observation
    )
    assert_allclose(value, scaled_value)

    for reward in [-2.0, 4.0]:
        replay_buffer.add_sample(
            observation=np.zeros(3),
            action=np.zeros(1),
            reward=reward,
            next_observation=np.zeros(3),
            terminated=False,
            truncated=False,
        )
    value = mrq_value(
        state.q, state.policy_with_encoder, replay_buffer, observation
    )
    assert_allclose(value, 3.0 * scaled_value, rtol=1e-6)
