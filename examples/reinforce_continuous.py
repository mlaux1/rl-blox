import gymnasium as gym
import numpy as np
import optax
import jax.numpy as jnp
from flax import nnx

from rl_blox.algorithms.model_free.reinforce_flax import PolicyTrainer, GaussianMLP, MLP, train_reinforce_epoch

# env_name = "Pendulum-v1"
# env_name = "HalfCheetah-v4"
env_name = "InvertedPendulum-v5"
env = gym.make(env_name)
env.reset(seed=43)

observation_space = env.observation_space
action_space = env.action_space
policy = GaussianMLP(
    shared_head=True,
    n_features=observation_space.shape[0],
    n_outputs=action_space.shape[0],
    hidden_nodes=[16, 32],
    rngs=nnx.Rngs(43),
)

value_function = MLP(
    n_features=observation_space.shape[0],
    n_outputs=1,
    hidden_nodes=[50, 50],
    rngs=nnx.Rngs(44),
)
v_opt = nnx.Optimizer(value_function, optax.adamw(learning_rate=1e-2))

policy_trainer = PolicyTrainer(policy, optimizer=optax.adamw(learning_rate=1e-4))

n_epochs = 5000
for i in range(n_epochs):
    print(f"Epoch #{i + 1}")
    train_reinforce_epoch(
        env,
        policy,
        policy_trainer,
        value_function,
        v_opt,
        batch_size=1000,
        gamma=0.99,
        train_after_episode=False,
    )

# Evaluation
env = gym.make(env_name, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
while True:
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        mean_action, _ = policy(jnp.asarray(obs))
        action = np.asarray(mean_action)
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
