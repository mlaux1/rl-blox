from rl_blox.algorithm.mrq import create_mrq_state, train_mrq


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
