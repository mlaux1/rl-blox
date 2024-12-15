import gymnasium as gym
import jax
from flax.training.train_state import TrainState
import optax
from rl_blox.algorithms.model_free.reinforce import (
    PolicyTrainer,
    ValueFunctionApproximation,
    train_reinforce_epoch,
)
from rl_blox.policy.differentiable import GaussianNNPolicy

# env_name = "Pendulum-v1"
# env_name = "HalfCheetah-v4"
env_name = "InvertedPendulum-v5"
train_env = gym.make(env_name)
train_env.reset(seed=43)
# render_env = gym.make(env_name, render_mode="human")
render_env = None

observation_space = train_env.observation_space
action_space = train_env.action_space
policy = GaussianNNPolicy(
    observation_space, action_space, [16, 32], jax.random.PRNGKey(42)
)

value_key = jax.random.PRNGKey(43)
obs, _ = train_env.reset(seed=42)
value_function = ValueFunctionApproximation(hidden_nodes=[50, 50])
value_function_state = TrainState.create(
    apply_fn=value_function.apply,
    params=value_function.init(value_key, obs),
    tx=optax.adam(learning_rate=1e-2),
)
value_function.apply = jax.jit(value_function.apply)

policy_trainer = PolicyTrainer(
    policy, optimizer=optax.adamw, learning_rate=1e-4
)

n_epochs = 5000
for i in range(n_epochs):
    print(f"Epoch #{i + 1}")
    train_reinforce_epoch(
        train_env,
        policy,
        policy_trainer,
        render_env,
        value_function,
        value_function_state,
        batch_size=1000,
        gamma=0.99,
        train_after_episode=False,
        n_train_iters_per_update=1
    )
