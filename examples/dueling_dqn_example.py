import gymnasium as gym
import jax.numpy as jnp
import optax
from flax import nnx

from rl_blox.algorithm.dueling_dqn import train_dueling_dqn
from rl_blox.blox.dueling_qnet import DuelingQNet
from rl_blox.blox.replay_buffer import ReplayBuffer
from rl_blox.logging.logger import AIMLogger


"""
Network architecture as defined in this example:

                 Input (4)
                 │
                 [Shared Hidden Layers]
                 │
 ┌───────────────┴───────────────┐
 │                               │
Advantage Stream             State-Value Stream
(128→32→2)                   (128→16→1)
 │                               │
 └───────────────┬───────────────┘
                 │
            Aggregation
                 │
            Q-values (2)
"""

# Set up environment
env_name = "CartPole-v1"
env = gym.make(env_name)
seed = 42
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)

hparams_model = dict(
    activation="relu",
    shared_nodes=[128, 128],
    advantage_nodes=[64],
    state_value_nodes=[64],
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
    algorithm_name="Dueling_DQN",
    hparams=hparams_model | hparams_algorithm,
)
# Initialise the Q-Network
q_net = DuelingQNet(
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
q, _, _ = train_dueling_dqn(
    q_net,
    env,
    rb,
    optimizer,
    **hparams_algorithm,
    logger=logger,
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
