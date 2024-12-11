import gymnasium as gym
import jax
import optax
from rl_blox.algorithms.model_free.actor_critic import train_ac_epoch
from rl_blox.algorithms.model_free.reinforce import (
    PolicyTrainer,
    ValueFunctionApproximation,
)
from rl_blox.policy.differentiable import SoftmaxNNPolicy

env_name = "CartPole-v1"
# env_name = "MountainCar-v0"  # never reaches the goal -> never learns
train_env = gym.make(env_name)
train_env.reset(seed=42)
render_env = gym.make(env_name, render_mode="human")
render_env.reset(seed=42)
# render_env = None

observation_space = train_env.observation_space
action_space = train_env.action_space
policy = SoftmaxNNPolicy(
    observation_space, action_space, [32], jax.random.PRNGKey(42)
)

value_function = ValueFunctionApproximation(
    observation_space,
    [100, 100],
    jax.random.PRNGKey(43),
    n_train_iters_per_update=1,
)

policy_trainer = PolicyTrainer(policy, optimizer=optax.adam, learning_rate=1e-2)

n_epochs = 50
for i in range(n_epochs):
    train_ac_epoch(
        train_env,
        policy,
        policy_trainer,
        render_env,
        value_function,
        batch_size=5000,
        gamma=1.0,
    )
