from collections import namedtuple

import gymnasium as gym
import torch

from rl_blox.algorithms.model_based.ext.tdmpc2 import train_tdmpc2

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)

agent = train_tdmpc2(
    env=env,
    task=env_name,
    obs="state",
    checkpoint="???",
    # eval
    eval_episodes=10,
    eval_freq=2_000,
    steps=4_000,
    # training
    batch_size=256,
    reward_coef=0.1,
    value_coef=0.1,
    consistency_coef=20,
    rho=0.5,
    lr=3e-4,
    enc_lr_scale=0.3,
    grad_clip_norm=20,
    tau=0.01,
    discount_denom=5,
    discount_min=0.95,
    discount_max=0.995,
    buffer_size=1_000_000,
    exp_name="default",
    data_dir="???",
    # planning
    mpc=True,
    iterations=6,
    num_samples=512,
    num_elites=64,
    num_pi_trajs=24,
    horizon=3,
    min_std=0.05,
    max_std=2,
    temperature=0.5,
    # actor
    log_std_min=-10,
    log_std_max=2,
    entropy_coef=1e-4,
    # critic
    num_bins=101,
    vmin=-10,
    vmax=+10,
    # architecture
    model_size=1,  # 1, 5, 19, 48, 317
    num_enc_layers=2,
    enc_dim=256,
    num_channels=32,
    mlp_dim=512,
    latent_dim=512,
    task_dim=96,
    num_q=5,
    dropout=0.01,
    simnorm_dim=8,
    # logging
    wandb_project="???",
    wandb_entity="???",
    wandb_silent=False,
    enable_wandb=True,
    save_csv=True,
    # misc
    save_agent=True,
    seed=seed,
    # convenience
    work_dir="???",
    task_title="???",
    multitask=False,
    tasks="???",
    obs_shape="???",
    action_dim="???",
    episode_length="???",
    obs_shapes="???",
    action_dims="???",
    episode_lengths="???",
    seed_steps="???",
    bin_size="???",
    # speedups
    compile=False,
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
