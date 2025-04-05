import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .reinforce import (
    MLP,
    EpisodeDataset,
    ProbabilisticPolicyBase,
    policy_gradient_pseudo_loss,
    train_value_function,
)


@nnx.jit
def actor_critic_policy_gradient(
    policy: ProbabilisticPolicyBase,
    value_function: nnx.Module,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    gamma_discount: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    r"""Actor-critic policy gradient.

    Parameters
    ----------
    policy
        Probabilistic policy that we want to update and has been used for
        exploration.
    value_function
        Estimated value function.
    observations
        Samples that were collected with the policy.
    actions
        Samples that were collected with the policy.
    next_observations
        Samples that were collected with the policy.
    rewards
        Samples that were collected with the policy.
    gamma_discount
        Discounting for individual steps of the episode.
    gamma
        Discount factor.

    Returns
    -------
    loss
        Actor-critic policy gradient pseudo loss.
    grad
        Actor-critic policy gradient.
    """
    v = value_function(observations).squeeze()
    v_next = value_function(next_observations).squeeze()
    td_bootstrap_estimate = rewards + gamma * v_next - v
    weights = gamma_discount * td_bootstrap_estimate

    return nnx.value_and_grad(policy_gradient_pseudo_loss, argnums=3)(
        observations, actions, weights, policy
    )


def train_ac_epoch(
    env: gym.Env,
    policy: ProbabilisticPolicyBase,
    policy_optimizer: nnx.Optimizer,
    value_function: MLP,
    value_function_optimizer: nnx.Optimizer,
    policy_gradient_steps: int = 1,
    value_gradient_steps: int = 1,
    total_steps: int = 1000,
    gamma: float = 1.0,
    train_after_episode: bool = False,
    verbose: int = 0,
):
    """Train with actor-critic for one epoch.

    Parameters
    ----------
    env : gym.Env
        Environment.

    policy : nnx.Module
        Probabilistic policy network. Maps observations to probability
        distribution over actions.

    policy_optimizer : nnx.Optimizer
        Optimizer for policy network.

    value_function : nnx.Module or None, optional
        Policy network. Maps observations to expected returns.

    value_function_optimizer : nnx.Optimizer or None, optional
        Optimizer for value function network.

    policy_gradient_steps : int, optional
        Number of gradient descent steps for the policy network.

    value_gradient_steps : int, optional
        Number of gradient descent steps for the value network.

    total_steps : int, optional
        Number of samples to collect before updating the policy. Alternatively
        you can train after each episode.

    gamma : float, optional
        Discount factor for rewards.

    train_after_episode : bool, optional
        Train after each episode. Alternatively you can train after collecting
        a certain number of samples.

    verbose : int, optional
        Verbosity level.
    """
    dataset = EpisodeDataset()

    dataset.start_episode()
    observation, _ = env.reset()
    while True:
        action = policy.sample(jnp.array(observation))

        next_observation, reward, terminated, truncated, _ = env.step(
            np.asarray(action)
        )

        done = terminated or truncated

        dataset.add_sample(observation, action, next_observation, reward)

        observation = next_observation

        if done:
            if train_after_episode or len(dataset) >= total_steps:
                break

            env = env
            observation, _ = env.reset()
            dataset.start_episode()

    if verbose:
        print(
            f"[Actor-Critic] Average return in sampled "
            f"dataset: {dataset.average_return():.3f}"
        )

    observations, actions, next_observations, returns, gamma_discount = (
        dataset.prepare_policy_gradient_dataset(env.action_space, gamma)
    )
    rewards = jnp.hstack(
        [
            jnp.hstack([r for _, _, _, r in episode])
            for episode in dataset.episodes
        ]
    )

    p_loss = train_policy_actor_critic(
        policy,
        policy_optimizer,
        policy_gradient_steps,
        value_function,
        observations,
        actions,
        next_observations,
        rewards,
        gamma_discount,
        gamma,
    )
    if verbose >= 2:
        print(f"[Actor-Critic] Policy loss: {p_loss:.3f}")

    if value_function is not None:
        assert value_function_optimizer is not None
        v_loss = train_value_function(
            value_function,
            value_function_optimizer,
            value_gradient_steps,
            observations,
            returns,
        )
        if verbose >= 2:
            print(f"[Actor-Critic] Value function loss: {v_loss:.3f}")


def train_policy_actor_critic(
    policy,
    policy_optimizer,
    policy_gradient_steps,
    value_function,
    observations,
    actions,
    next_observations,
    rewards,
    gamma_discount,
    gamma,
):
    p_loss = 0.0
    for _ in range(policy_gradient_steps):
        p_loss, p_grad = actor_critic_policy_gradient(
            policy,
            value_function,
            observations,
            actions,
            next_observations,
            rewards,
            gamma_discount,
            gamma,
        )
        policy_optimizer.update(p_grad)
    return p_loss
