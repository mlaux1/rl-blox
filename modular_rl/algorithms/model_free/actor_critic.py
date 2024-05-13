from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ...policy.differentiable import GaussianNNPolicy, NeuralNetwork
from .reinforce import (EpisodeDataset, ValueFunctionApproximation,
                        gaussian_policy_gradient_pseudo_loss,
                        softmax_policy_gradient_pseudo_loss)


def ac_policy_gradient(
        policy: NeuralNetwork,
        value_function: ValueFunctionApproximation,
        states: jax.Array, actions: jax.Array, next_states: jax.Array,
        rewards: jax.Array, gamma_discount: jax.Array, gamma: float
) -> jax.Array:
    V = value_function.predict(states)
    V_next = value_function.predict(next_states)
    TD_bootstrap_estimate = rewards + gamma * V_next - V
    weights = gamma_discount * TD_bootstrap_estimate

    if isinstance(policy, GaussianNNPolicy):  # TODO find another way without if-else
        return jax.grad(
            partial(gaussian_policy_gradient_pseudo_loss, states, actions, weights)
        )(policy.theta)
    else:
        return jax.grad(
            partial(softmax_policy_gradient_pseudo_loss, states, actions, weights)
        )(policy.theta)


def train_ac_epoch(train_env, policy, policy_trainer, render_env, value_function, batch_size, gamma, train_after_episode=False):
    dataset = EpisodeDataset()
    if render_env is not None:
        env = render_env
    else:
        env = train_env

    dataset.start_episode()
    observation, _ = env.reset()
    while True:
        action = policy.sample(jnp.array(observation))
        next_observation, reward, terminated, truncated, _ = env.step(np.asarray(action))

        done = terminated or truncated

        dataset.add_sample(observation, action, next_observation, reward)

        observation = next_observation

        if done:
            if train_after_episode or len(dataset) >= batch_size:
                break

            env = train_env
            observation, _ = env.reset()
            dataset.start_episode()

    print(f"{dataset.average_return()=}")

    states, actions, next_states, returns, gamma_discount = dataset.prepare_policy_gradient_dataset(
        env.action_space, gamma)
    rewards = jnp.hstack([jnp.hstack([r for _, _, _, r in episode])
                          for episode in dataset.episodes])

    policy_trainer.update(
        ac_policy_gradient, value_function, states, actions, next_states, rewards,
        gamma_discount, gamma)

    if value_function is not None:
        value_function.update(states, returns)
