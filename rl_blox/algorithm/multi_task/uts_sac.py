import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike
from tqdm.rich import tqdm

from ...blox.double_qnet import ContinuousClippedDoubleQNet
from ...blox.function_approximator.policy_head import StochasticPolicyBase
from ...blox.replay_buffer import ReplayBuffer
from ..sac import EntropyControl, train_sac


class TaskSet:
    """A collection of tasks (environments)."""

    def __init__(self, contexts: ArrayLike, envs: list[gym.Env]):
        assert len(contexts) == len(envs)
        self.contexts = contexts
        self.task_envs = envs

    def get_context(self, task_id: int) -> jnp.ndarray:
        assert task_id >= 0 and task_id < len(self.contexts)
        return self.contexts[task_id]

    def get_task_env(self, task_id: int) -> gym.Env:
        assert task_id >= 0 and task_id < len(self.contexts)
        return self.task_envs[task_id]

    def __len__(self) -> int:
        return len(self.contexts)


def train_uts_sac(
    envs: TaskSet,
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
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    key = jax.random.key(seed)

    while steps_so_far < total_timesteps:
        key, skey = jax.random.split(key)
        env_id = jax.random.choice(key, jnp.arange(len(envs))).item()
        env = envs.get_task_env(env_id)
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
            progress_bar=False,
        )

        steps_so_far += ep_steps
        episodes_so_far += 1
        progress.update(ep_steps)

    return (
        policy,
        policy_optimizer,
        q_net,
        q_target,
        q_optimizer,
        entropy_control,
        replay_buffer,
    )
