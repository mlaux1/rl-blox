import gymnasium as gym
import jax
import jax.numpy as jnp

from ..blox.value_policy import get_epsilon_greedy_action


class ForwardModel:
    def __init__(self):
        jnp.array([])


def train_dynaq(
    env: gym.Env,
    q_table: jnp.ndarray,
    epsilon: float = 0.05,
    n_planning_steps: int = 5,
    total_timesteps: int = 1_000_000,
    seed: int = 0,
):
    """Train tabular Dyna-Q for discrete state and action spaces.

    Dyna-Q integrates trial-and-error learning and planning into a process
    operating alternately on the environment and on a model of the environment.
    The model is learned online in parallel to learning a policy. Policy
    learning is based on reinforcement learning and planning. The Q-function
    is updated based on actual and simulated transitions.

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium environment.

    q_table : array
        Tabular action-value function.

    epsilon : float, optional
        Exploration probability for epsilon-greedy policy.

    n_planning_steps : int, optional
        Number of planning steps.

    total_timesteps : int, optional
        The number of environment steps to train for.

    seed : int, optional
        Seed for random number generator.
    """
    key = jax.random.key(seed)
    model = ForwardModel()

    obs, _ = env.reset(seed=seed)
    for _ in range(total_timesteps):
        key, sampling_key = jax.random.split(key)
        act = int(get_epsilon_greedy_action(sampling_key, q_table, obs, epsilon))
        obs, reward, terminated, truncated, _ = env.step(act)
        # direct RL
        # TODO
        # model learning
        # TODO
        # planning
        for _ in range(n_planning_steps):
            # TODO
            print("plan plan plan")
        if terminated or truncated:
            obs, _ = env.reset()
