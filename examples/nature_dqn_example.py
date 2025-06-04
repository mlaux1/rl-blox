import gymnasium as gym
import jax.numpy as jnp
import optax
from flax import nnx

from rl_blox.algorithm.nature_dqn import train_nature_dqn
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.replay_buffer import ReplayBuffer

# Set up environment
env_name = "CartPole-v1"
env = gym.make(env_name)
seed = 42
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)

# Initialise the Q-Network
q_net = MLP(
    env.observation_space.shape[0],
    int(env.action_space.n),
    [32, 32],
    "relu",
    nnx.Rngs(seed),
)

# Initialise the replay buffer
rb = ReplayBuffer(200_000)

# initialise optimiser
optimizer = nnx.Optimizer(q_net, optax.rprop(0.003))

# Train
q, _, _ = train_nature_dqn(
    q_net,
    env,
    rb,
    optimizer,
    seed=seed,
    total_timesteps=250_000,
)
env.close()

# Show the final policy
eval_env = gym.make(env_name, render_mode="human")
obs, _ = eval_env.reset()

while True:
    action = int(jnp.argmax(q([obs])))
    next_obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, _ = eval_env.reset()
    else:
        obs = next_obs
