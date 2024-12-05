import gymnasium as gym
import jax
import optax

from modular_rl.algorithms.model_free.actor_critic import train_ac_epoch
from modular_rl.algorithms.model_free.reinforce import (
    PolicyTrainer, ValueFunctionApproximation)
from modular_rl.policy.differentiable import GaussianNNPolicy

#env_name = "Pendulum-v1"
#env_name = "HalfCheetah-v4"
env_name = "InvertedPendulum-v4"
train_env = gym.make(env_name)
train_env.reset(seed=43)
#render_env = gym.make(env_name, render_mode="human")
render_env = None

observation_space = train_env.observation_space
action_space = train_env.action_space
policy = GaussianNNPolicy(observation_space, action_space, [16, 32], jax.random.PRNGKey(42))

value_function = ValueFunctionApproximation(
    observation_space, [50, 50], jax.random.PRNGKey(43),
    n_train_iters_per_update=1)

policy_trainer = PolicyTrainer(policy, optimizer=optax.adamw, learning_rate=1e-4)

n_epochs = 5000
for i in range(n_epochs):
    print(f"Epoch #{i + 1}")
    train_ac_epoch(
        train_env, policy, policy_trainer, render_env, value_function,
        batch_size=1000, gamma=0.99, train_after_episode=False)
