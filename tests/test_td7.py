import gymnasium as gym

from rl_blox.algorithm.td7 import create_td7_state, train_td7


def test_td7():
    env = gym.make("Pendulum-v1")

    seed = 0
    td7_state = create_td7_state(env, seed=seed)
    train_td7(
        env,
        embedding=td7_state.embedding,
        embedding_optimizer=td7_state.embedding_optimizer,
        actor=td7_state.actor,
        actor_optimizer=td7_state.actor_optimizer,
        critic=td7_state.critic,
        critic_optimizer=td7_state.critic_optimizer,
        total_timesteps=10,
        seed=seed,
    )
    env.close()
