import gymnasium as gym
import jax.numpy as jnp
from flax import nnx

from rl_blox.algorithms.model_free.dqn import MLP, train_dqn
from rl_blox.policy.replay_buffer import ReplayBuffer

# Set up environment
env_name = "CartPole-v1"
env = gym.make(env_name)
seed = 42
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)

# Initialise the Q-Network
nnx_rngs = nnx.Rngs(seed)
q_net = MLP(4, 10, env.action_space.n, nnx_rngs)

# Initialse the replay buffer
rb = ReplayBuffer(30_000)

# Train
q = train_dqn(
    q_net,
    env,
    rb,
    learning_rate=0.003,
    seed=seed,
    total_timesteps=30_000,
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
