import gymnasium as gym
import jax.numpy as jnp
import optax
from flax import nnx

from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.replay_buffer import ReplayBuffer
from rl_blox.logging.logger import AIMLogger

# Set up environment
env_name = "CartPole-v1"
env = gym.make(env_name)
seed = 42
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)

hparams_model = dict(
    activation="relu",
    hidden_nodes=[128, 128],
)
hparams_algorithm = dict(
    buffer_size=50_000,
    total_timesteps=100_000,
    learning_rate=0.003,
    seed=seed,
)

logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="DDQN",
    hparams=hparams_model | hparams_algorithm,
)
# Initialise the Q-Network
q_net = MLP(
    env.observation_space.shape[0],
    int(env.action_space.n),
    rngs=nnx.Rngs(seed),
    **hparams_model,
)

# Initialise the replay buffer
rb = ReplayBuffer(hparams_algorithm.pop("buffer_size"), discrete_actions=True)

# initialise optimiser
optimizer = nnx.Optimizer(
    q_net, optax.adam(hparams_algorithm.pop("learning_rate")), wrt=nnx.Param
)

# Train
result_dict = train_ddqn(
    q_net,
    env,
    rb,
    optimizer,
    **hparams_algorithm,
    logger=logger,
)
env.close()

q = result_dict.q_net

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
