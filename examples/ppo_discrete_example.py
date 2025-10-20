import gymnasium as gym
import jax
import optax
from flax import nnx

from rl_blox.algorithm.ppo import train_ppo
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import SoftmaxPolicy
from rl_blox.logging.logger import AIMLogger, StandardLogger, LoggerList


env_name = "CartPole-v1"
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
    num_envs=32,
    iterations=3000,
    epochs=1,
    seed=seed,
)

envs = gym.make_vec("CartPole-v1", num_envs=hparams_algorithm['num_envs'], vectorization_mode="sync")

features = envs.observation_space.shape[1]
actions = int(envs.single_action_space.n)

actor = MLP(
    features,
    actions,
    hparams_model["actor_hidden_layers"],
    hparams_model["actor_activation"],
    nnx.Rngs(seed),
)
actor = SoftmaxPolicy(actor)

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
    epochs=hparams_algorithm["epochs"],
    logger=logger,
)

envs.close()

# Evaluation

env = gym.make(env_name, render_mode="human")

key = jax.random.key(seed)
obs, _ = env.reset()
finished_episodes = 0
reward_sum = 0

while finished_episodes < test_episodes:
    key, subkey = jax.random.split(key)
    action = actor.sample(obs, subkey) 
    next_obs, reward, terminated, truncated, _ = env.step(int(action))
    done = terminated or truncated
    reward_sum += reward

    if done:
        finished_episodes += 1
        next_obs, _ = env.reset()

print(f"Return: {reward_sum}")
