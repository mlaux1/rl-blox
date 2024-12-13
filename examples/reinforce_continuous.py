import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from rl_blox.algorithms.model_free.reinforce import (
    ValueFunctionApproximation,
    train_reinforce_epoch,
)
from rl_blox.policy.differentiable import GaussianMlpPolicyNetwork

# env_name = "Pendulum-v1"
# env_name = "HalfCheetah-v4"
env_name = "InvertedPendulum-v5"
train_env = gym.make(env_name)
train_env.reset(seed=43)
# render_env = gym.make(env_name, render_mode="human")
render_env = None

key = jax.random.PRNGKey(42)
policy_key, value_key = jax.random.split(key, 2)

obs, _ = train_env.reset(seed=42)

observation_space = train_env.observation_space
action_space = train_env.action_space
policy = GaussianMlpPolicyNetwork.create(
    [16, 32],
    action_dim=np.prod(action_space.shape),
    action_scale=jnp.array((action_space.high - action_space.low) / 2.0),
    action_bias=jnp.array((action_space.high + action_space.low) / 2.0)
)
policy_state = TrainState.create(
    apply_fn=policy.apply,
    params=policy.init(policy_key, obs),
    tx=optax.adamw(learning_rate=1e-4),
)
policy.apply = jax.jit(policy.apply)

value_function = ValueFunctionApproximation(hidden_nodes=[50, 50])
value_function_state = TrainState.create(
    apply_fn=value_function.apply,
    params=value_function.init(value_key, obs),
    tx=optax.adam(learning_rate=1e-2),
)
value_function.apply = jax.jit(value_function.apply)

n_epochs = 5000
for i in range(n_epochs):
    print(f"Epoch #{i + 1}")
    train_reinforce_epoch(
        train_env,
        policy,
        policy_state,
        render_env,
        value_function,
        value_function_state,
        batch_size=1000,
        gamma=0.99,
        train_after_episode=False,
    )
