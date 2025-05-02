import gymnasium as gym

from rl_blox.algorithm.pets import create_pets_state, train_pets
from rl_blox.algorithm.pets_reward_models import pendulum_reward


def test_pets():
    env = gym.make("Pendulum-v1")

    dynamics_model = create_pets_state(env, seed=0)
    train_pets(
        env,
        pendulum_reward,
        dynamics_model,
        plan_horizon=25,
        n_particles=20,
        n_samples=400,
        total_timesteps=10,
    )

    env.close
