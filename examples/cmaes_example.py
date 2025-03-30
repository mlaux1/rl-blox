import gymnasium as gym
from flax import nnx

from rl_blox.algorithms.model_free.cmaes import train_cmaes, MLPPolicy


env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)
policy = MLPPolicy(env, [32], nnx.Rngs(seed))
policy = train_cmaes(
    env,
    policy,
    300,
    seed
)
env.close()
