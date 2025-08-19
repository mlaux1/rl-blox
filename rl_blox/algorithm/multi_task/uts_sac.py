import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from gymnasium.wrappers import TransformObservation
from jax.typing import ArrayLike
from tqdm.rich import tqdm

from ...blox.double_qnet import ContinuousClippedDoubleQNet
from ...blox.function_approximator.policy_head import StochasticPolicyBase
from ...blox.replay_buffer import ReplayBuffer
from ...logging.logger import LoggerBase
from ..sac import EntropyControl, train_sac


class TaskSet:
    """A collection of tasks (environments)."""

    def __init__(
        self, contexts: list[ArrayLike], envs: list[gym.Env], context_aware=True
    ):
        assert len(contexts) == len(envs)
        self.contexts = contexts
        self.task_envs = envs
        self.context_aware = context_aware

        if context_aware:
            for i in range(len(contexts)):
                assert isinstance(
                    self.task_envs[i].observation_space, gym.spaces.Box
                )
                new_low = np.concatenate(
                    [self.task_envs[i].observation_space.low, self.contexts[i]]
                )
                new_high = np.concatenate(
                    [self.task_envs[i].observation_space.high, self.contexts[i]]
                )
                new_obs_space = gym.spaces.Box(low=new_low, high=new_high)
                self.task_envs[i] = TransformObservation(
                    self.task_envs[i],
                    lambda obs: np.concatenate([obs, self.contexts[i]]),
                    new_obs_space,
                )

    def get_context(self, task_id: int) -> jnp.ndarray:
        assert 0 <= task_id < len(self.contexts)
        return self.contexts[task_id]

    def get_task_env(self, task_id: int) -> gym.Env:
        assert 0 <= task_id < len(self.contexts)
        return self.task_envs[task_id]

    def get_task_and_context(self, task_id: int) -> tuple[gym.Env, jnp.ndarray]:
        assert 0 <= task_id < len(self.contexts)
        return self.task_envs[task_id], self.contexts[task_id]

    def __len__(self) -> int:
        return len(self.contexts)


class PrioritisedTaskSampler:
    """A sampler that returns envs from a TaskSet given priorities."""

    def __init__(
        self,
        task_set: TaskSet,
        priorities: ArrayLike | None = None,
    ):
        self.task_set = task_set
        self.priorities = priorities

    def sample(self, key) -> tuple[gym.Env, jnp.ndarray]:
        env_id = jax.random.choice(
            key, jnp.arange(len(self.task_set)), p=self.priorities
        ).item()
        return self.task_set.get_task_and_context(env_id)

    def update_priorities(self, priorities) -> None:
        self.priorities = priorities


def train_uts_sac(
    envs: TaskSet,
    policy: StochasticPolicyBase,
    policy_optimizer: nnx.Optimizer,
    q_net: ContinuousClippedDoubleQNet,
    q_optimizer: nnx.Optimizer,
    total_timesteps: int = 100_000,
    episodes_per_task: int = 1,
    seed: int = 1,
    exploring_starts: int = 1_000,
    progress_bar: bool = True,
    logger: LoggerBase = None,
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
    task_sampler = PrioritisedTaskSampler(envs)

    while steps_so_far < total_timesteps:
        key, skey = jax.random.split(key)
        env, context = task_sampler.sample(skey)
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
            learning_starts=exploring_starts - steps_so_far,
            progress_bar=False,
            logger=logger,
            step_offset=steps_so_far + 1,
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
