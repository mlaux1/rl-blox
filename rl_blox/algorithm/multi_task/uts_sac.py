import random

import gymnasium as gym
from flax import nnx

from ...blox.double_qnet import ContinuousClippedDoubleQNet
from ...blox.function_approximator.policy_head import StochasticPolicyBase
from ...blox.replay_buffer import ReplayBuffer
from ..sac import EntropyControl, train_sac


def train_uts_sac(
    envs: list[gym.Env],
    policy: StochasticPolicyBase,
    policy_optimizer: nnx.Optimizer,
    q_net: ContinuousClippedDoubleQNet,
    q_optimizer: nnx.Optimizer,
    time_steps_per_epoch: int,
    epochs: int,
    seed: int = 1,
    exploring_starts: int = 0,
) -> tuple[
    nnx.Module,
    nnx.Optimizer,
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    EntropyControl,
    ReplayBuffer,
]:
    replay_buffer = None
    q_target = None
    entropy_control = None

    for i in range(epochs):
        env = random.choice(envs)
        (
            policy,
            policy_optimizer,
            q_net,
            q_target,
            q_optimizer,
            entropy_control,
            replay_buffer,
        ) = train_sac(
            env,
            policy,
            policy_optimizer,
            q_net,
            q_optimizer,
            seed=seed,
            total_timesteps=time_steps_per_epoch,
            replay_buffer=replay_buffer,
            q_target=q_target,
            entropy_control=entropy_control,
            learning_starts=exploring_starts,
        )
        print(f"Epoch {i} complete.")

    return (
        policy,
        policy_optimizer,
        q_net,
        q_target,
        q_optimizer,
        entropy_control,
        replay_buffer,
    )
