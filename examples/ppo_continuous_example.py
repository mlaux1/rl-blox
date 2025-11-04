import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from rl_blox.algorithm.ppo import train_ppo
from rl_blox.blox.function_approximator.gaussian_mlp import GaussianMLP
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import GaussianTanhPolicy
from rl_blox.logging.logger import AIMLogger

env_name = "Pendulum-v1"
seed = 1
test_episodes = 10

hparams_model = {
    "actor_hidden_layers": [64, 64],
    "actor_activation": "relu",
    "actor_learning_rate": 3e-4,
    "critic_hidden_layers": [64, 64],
    "critic_activation": "relu",
    "critic_learning_rate": 1e-3,
}
hparams_algorithm = dict(
    num_envs=64,
    batch_size=256,
    iterations=300,
    epochs=1,
    seed=seed,
)

envs = gym.make_vec(
    env_name,
    num_envs=hparams_algorithm["num_envs"],
    vectorization_mode="sync",
    vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
)

features = envs.observation_space.shape[1]
n_action_dims = envs.single_action_space.shape[0]

actor = GaussianMLP(
    True,
    features,
    n_action_dims,
    hparams_model["actor_hidden_layers"],
    hparams_model["actor_activation"],
    nnx.Rngs(seed),
)
actor = GaussianTanhPolicy(actor, action_space=envs.single_action_space)

critic = MLP(
    features,
    1,
    hparams_model["critic_hidden_layers"],
    hparams_model["critic_activation"],
    nnx.Rngs(seed),
)

optimizer_actor = nnx.Optimizer(
    actor, optax.adam(hparams_model["actor_learning_rate"]), wrt=nnx.Param
)
optimizer_critic = nnx.Optimizer(
    critic, optax.adam(hparams_model["critic_learning_rate"]), wrt=nnx.Param
)

logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="PPO",
    hparams=hparams_model | hparams_algorithm,
)

actor, critic, optimizer_actor, optimizer_critic = train_ppo(
    envs,
    actor,
    critic,
    optimizer_actor,
    optimizer_critic,
    iterations=hparams_algorithm["iterations"],
    epochs=hparams_algorithm["epochs"],
    logger=logger,
    batch_size=hparams_algorithm["batch_size"],
)

envs.close()

# Evaluation

env = gym.make(env_name, render_mode="human")

obs, _ = env.reset(seed=seed)

while True:
    action = np.array(actor(jnp.array(obs)))
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
