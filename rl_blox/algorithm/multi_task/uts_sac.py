import random
from dataclasses import dataclass

import gymnasium as gym
from flax import nnx

from ...blox.double_qnet import ContinuousClippedDoubleQNet
from ...blox.function_approximator.policy_head import StochasticPolicyBase
from ...blox.replay_buffer import ReplayBuffer
from ..sac import EntropyControl, train_sac


@dataclass(frozen=True)
class EnvSpec:
    name: str
    id: int
    context: float


def train_uts_sac(
    envs: dict[EnvSpec, gym.Env],
    policy: StochasticPolicyBase,
    policy_optimizer: nnx.Optimizer,
    q_net: ContinuousClippedDoubleQNet,
    q_optimizer: nnx.Optimizer,
    total_timesteps: int = 100_000,
    episodes_per_task: int = 1,
    seed: int = 1,
    exploring_starts: int = 0,
    progress_bar: bool = True,
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
    steps_so_far = 0
    episodes_so_far = 0

    while steps_so_far < total_timesteps:
        spec, env = random.choice(list(envs.items()))
        print(f"Selected env {spec.id} with context {spec.context}.")

        (
            policy,
            policy_optimizer,
            q_net,
            q_target,
            q_optimizer,
            entropy_control,
            replay_buffer,
            ep_steps,
        ) = train_sac(
            env,
            policy,
            policy_optimizer,
            q_net,
            q_optimizer,
            seed=seed,
            total_timesteps=total_timesteps - steps_so_far,
            max_episodes=episodes_per_task,
            replay_buffer=replay_buffer,
            q_target=q_target,
            entropy_control=entropy_control,
            learning_starts=exploring_starts,
            progress_bar=progress_bar,
        )

        steps_so_far += ep_steps
        episodes_so_far += 1
        print(
            f"Episode {episodes_so_far} completed after {
                ep_steps
            } steps in env {env}. Total: {steps_so_far}."
        )

    return (
        policy,
        policy_optimizer,
        q_net,
        q_target,
        q_optimizer,
        entropy_control,
        replay_buffer,
    )
