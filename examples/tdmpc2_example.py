import gymnasium as gym
import torch

from rl_blox.algorithm.ext.tdmpc2 import train_tdmpc2
from rl_blox.logging.logger import AIMLogger

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
verbose = 2
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)

hparams = dict(
    obs="state",
    # eval
    eval_episodes=3,
    eval_freq=2_000,
    steps=4_000,
    # training
    lr=3e-4,
    buffer_size=1_000_000,
    exp_name="default",
    # planning
    mpc=True,
    iterations=6,
    num_samples=512,
    num_elites=64,
    num_pi_trajs=24,
    horizon=3,
    temperature=0.5,
    # architecture
    model_size=1,  # 1, 5, 19, 48, 317
    # misc
    seed=seed,
    # speedups
    compile=False,
)

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="TD-MPC2",
    hparams=hparams,
)

agent = train_tdmpc2(
    env=env,
    task=env_name,
    logger=logger,
    **hparams,
)
env.close()

# Evaluation
env = gym.make(env_name, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
while True:
    done = False
    infos = {}
    obs, _ = env.reset()
    t = 0
    ep_return = 0.0
    while not done:
        action = agent.act(torch.from_numpy(obs), t0=t == 0, eval_mode=True)
        obs, reward, termination, truncation, info = env.step(action)
        done = termination or truncation
        ep_return += reward
        t += 1
    print(f"{ep_return=}")
