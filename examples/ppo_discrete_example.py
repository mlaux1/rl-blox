import gymnasium as gym
import jax
import optax
from flax import nnx

from rl_blox.algorithm.ppo import select_action_deterministic, train_ppo
from rl_blox.blox.function_approximator.mlp import MLP

# TODO change environment to one with changable cntext. This was chosen for simplicity
env_name = "CartPole-v1"
seed = 1
test_episodes = 10

env = gym.make(env_name)

actor_hparam = {
    "hidden_layers": [64, 64],
    "activation": "relu",
    "learning_rate": 3e-4,
}

critic_hparam = {
    "hidden_layers": [64, 64],
    "activation": "relu",
    "learning_rate": 1e-3,
}

features = env.observation_space.shape[0]
actions = int(env.action_space.n)

actor = MLP(
    features,
    actions,
    actor_hparam["hidden_layers"],
    actor_hparam["activation"],
    nnx.Rngs(seed),
)

critic = MLP(
    features,
    1,
    critic_hparam["hidden_layers"],
    critic_hparam["activation"],
    nnx.Rngs(seed),
)

optimizer_actor = nnx.Optimizer(
    actor, optax.adam(actor_hparam["learning_rate"]), wrt=nnx.Param
)
optimizer_critic = nnx.Optimizer(
    critic, optax.adam(critic_hparam["learning_rate"]), wrt=nnx.Param
)

env, actor, critic, optimizer_actor, optimizer_critic = train_ppo(
    env, actor, critic, optimizer_actor, optimizer_critic
)

env.close()

# Evaluation

env = gym.make(env_name, render_mode="human")

key = jax.random.key(seed)
obs, _ = env.reset()
finished_episodes = 0
reward_sum = 0

while finished_episodes < test_episodes:
    key, subkey = jax.random.split(key)
    (
        action,
        _,
    ) = select_action_deterministic(actor, obs, subkey)
    next_obs, reward, terminated, truncated, _ = env.step(int(action))
    done = terminated or truncated
    reward_sum += reward

    if done:
        finished_episodes += 1
        next_obs, _ = env.reset()

print(f"Return: {reward_sum}")
