# Original source code of https://github.com/nicklashansen/tdmpc2 in one file

# Dependencies:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install torchrl tensordict termcolor
# pip install array-api-compat # for gymnasium.wrappers.NumpyToTorch

# MIT License
#
# Copyright (c) Nicklas Hansen (2023).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from collections import namedtuple

from rl_blox.logging.logger import LoggerBase
from rl_blox.logging.timer import Timer

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TORCH_LOGS"] = "+recompiles"
import warnings

warnings.filterwarnings("ignore")
import datetime
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import NumpyToTorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from tensordict.nn import TensorDictParams
from tensordict.tensordict import TensorDict
from termcolor import colored
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from tqdm.rich import trange

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


MODEL_SIZE = {  # parameters (M)
    1: {
        "enc_dim": 256,
        "mlp_dim": 384,
        "latent_dim": 128,
        "num_enc_layers": 2,
        "num_q": 2,
    },
    5: {
        "enc_dim": 256,
        "mlp_dim": 512,
        "latent_dim": 512,
        "num_enc_layers": 2,
        "num_q": 5,
    },
    19: {
        "enc_dim": 1024,
        "mlp_dim": 1024,
        "latent_dim": 768,
        "num_enc_layers": 3,
        "num_q": 5,
    },
    48: {
        "enc_dim": 1792,
        "mlp_dim": 1792,
        "latent_dim": 768,
        "num_enc_layers": 4,
        "num_q": 5,
    },
    317: {
        "enc_dim": 4096,
        "mlp_dim": 4096,
        "latent_dim": 1376,
        "num_enc_layers": 5,
        "num_q": 8,
    },
}

TASK_SET = {
    "mt30": [
        # 19 original dmcontrol tasks
        "walker-stand",
        "walker-walk",
        "walker-run",
        "cheetah-run",
        "reacher-easy",
        "reacher-hard",
        "acrobot-swingup",
        "pendulum-swingup",
        "cartpole-balance",
        "cartpole-balance-sparse",
        "cartpole-swingup",
        "cartpole-swingup-sparse",
        "cup-catch",
        "finger-spin",
        "finger-turn-easy",
        "finger-turn-hard",
        "fish-swim",
        "hopper-stand",
        "hopper-hop",
        # 11 custom dmcontrol tasks
        "walker-walk-backwards",
        "walker-run-backwards",
        "cheetah-run-backwards",
        "cheetah-run-front",
        "cheetah-run-back",
        "cheetah-jump",
        "hopper-hop-backwards",
        "reacher-three-easy",
        "reacher-three-hard",
        "cup-spin",
        "pendulum-spin",
    ],
    "mt80": [
        # 19 original dmcontrol tasks
        "walker-stand",
        "walker-walk",
        "walker-run",
        "cheetah-run",
        "reacher-easy",
        "reacher-hard",
        "acrobot-swingup",
        "pendulum-swingup",
        "cartpole-balance",
        "cartpole-balance-sparse",
        "cartpole-swingup",
        "cartpole-swingup-sparse",
        "cup-catch",
        "finger-spin",
        "finger-turn-easy",
        "finger-turn-hard",
        "fish-swim",
        "hopper-stand",
        "hopper-hop",
        # 11 custom dmcontrol tasks
        "walker-walk-backwards",
        "walker-run-backwards",
        "cheetah-run-backwards",
        "cheetah-run-front",
        "cheetah-run-back",
        "cheetah-jump",
        "hopper-hop-backwards",
        "reacher-three-easy",
        "reacher-three-hard",
        "cup-spin",
        "pendulum-spin",
        # meta-world mt50
        "mw-assembly",
        "mw-basketball",
        "mw-button-press-topdown",
        "mw-button-press-topdown-wall",
        "mw-button-press",
        "mw-button-press-wall",
        "mw-coffee-button",
        "mw-coffee-pull",
        "mw-coffee-push",
        "mw-dial-turn",
        "mw-disassemble",
        "mw-door-open",
        "mw-door-close",
        "mw-drawer-close",
        "mw-drawer-open",
        "mw-faucet-open",
        "mw-faucet-close",
        "mw-hammer",
        "mw-handle-press-side",
        "mw-handle-press",
        "mw-handle-pull-side",
        "mw-handle-pull",
        "mw-lever-pull",
        "mw-peg-insert-side",
        "mw-peg-unplug-side",
        "mw-pick-out-of-hole",
        "mw-pick-place",
        "mw-pick-place-wall",
        "mw-plate-slide",
        "mw-plate-slide-side",
        "mw-plate-slide-back",
        "mw-plate-slide-back-side",
        "mw-push-back",
        "mw-push",
        "mw-push-wall",
        "mw-reach",
        "mw-reach-wall",
        "mw-shelf-place",
        "mw-soccer",
        "mw-stick-push",
        "mw-stick-pull",
        "mw-sweep-into",
        "mw-sweep",
        "mw-window-open",
        "mw-window-close",
        "mw-bin-picking",
        "mw-box-close",
        "mw-door-lock",
        "mw-door-unlock",
        "mw-hand-insert",
    ],
}

CONSOLE_FORMAT = [
    ("iteration", "I", "int"),
    ("episode", "E", "int"),
    ("step", "I", "int"),
    ("episode_reward", "R", "float"),
    ("episode_success", "S", "float"),
    ("total_time", "T", "time"),
]

CAT_TO_COLOR = {
    "pretrain": "yellow",
    "train": "blue",
    "eval": "green",
}


def soft_ce(pred, target, vmin, vmax, bin_size, num_bins):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, vmin, vmax, bin_size, num_bins)
    return -(target * pred).sum(-1, keepdim=True)


def safe_log_std(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps, log_std):
    """Compute Gaussian log probability."""
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385175704956
    return log_prob.sum(-1, keepdim=True)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
    return mu, pi, log_pi


def int_to_one_hot(x, num_classes):
    """
    Converts an integer tensor to a one-hot tensor.
    Supports batched inputs.
    """
    one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return one_hot


def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, vmin, vmax, bin_size, num_bins):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), vmin, vmax).squeeze(1)
    bin_idx = torch.floor((x - vmin) / bin_size)
    bin_offset = ((x - vmin) / bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(
        x.shape[0], num_bins, device=x.device, dtype=x.dtype
    )
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(
        1, (bin_idx.unsqueeze(1) + 1) % num_bins, bin_offset
    )
    return soft_two_hot


def two_hot_inv(x, vmin, vmax, num_bins):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symexp(x)
    dreg_bins = torch.linspace(
        vmin, vmax, num_bins, device=x.device, dtype=x.dtype
    )
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return symexp(x)


def gumbel_softmax_sample(p, temperature=1.0, dim=0):
    logits = p.log()
    # Generate Gumbel noise
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    return y_soft.argmax(-1)


class RunningScale(torch.nn.Module):
    """Running trimmed scale estimator."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.value = torch.nn.Buffer(
            torch.ones(1, dtype=torch.float32, device=torch.device("cuda:0"))
        )
        self._percentiles = torch.nn.Buffer(
            torch.tensor(
                [5, 95], dtype=torch.float32, device=torch.device("cuda:0")
            )
        )

    def state_dict(self):
        return dict(value=self.value, percentiles=self._percentiles)

    def load_state_dict(self, state_dict):
        self.value.copy_(state_dict["value"])
        self._percentiles.copy_(state_dict["percentiles"])

    def _positions(self, x_shape):
        positions = self._percentiles * (x_shape - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled = torch.where(ceiled > x_shape - 1, x_shape - 1, ceiled)
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        return (
            floored.long(),
            ceiled.long(),
            weight_floored.unsqueeze(1),
            weight_ceiled.unsqueeze(1),
        )

    def _percentile(self, x):
        x_dtype, x_shape = x.dtype, x.shape
        x = x.flatten(1, x.ndim - 1)
        in_sorted = torch.sort(x, dim=0).values
        floored, ceiled, weight_floored, weight_ceiled = self._positions(
            x.shape[0]
        )
        d0 = in_sorted[floored] * weight_floored
        d1 = in_sorted[ceiled] * weight_ceiled
        return (d0 + d1).reshape(-1, *x_shape[1:]).to(x_dtype)

    def update(self, x):
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.0)
        self.value.data.lerp_(value, self.cfg.tau)

    def forward(self, x, update=False):
        if update:
            self.update(x)
        return x / self.value

    def __repr__(self):
        return f"RunningScale(S: {self.value})"


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


class DefaultSuccessInfoWrapper(gym.Wrapper):
    """
    Gym environment wrapper for filling the info["success"] field with a default value if missing.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, termination, truncation, info = self.env.step(action)
        info = defaultdict(float, info)
        info["success"] = float(info["success"])
        return obs, reward, termination, truncation, info


class NumpyToTorchSpaces(gym.Wrapper):
    """
    Gym environment wrapper for
      (1) sampling the env.action_space
      (2) stepping the environment
      (3) resetting the environment
    with torch float32 tensors as result types.
    """

    def __init__(self, env):
        super().__init__(env)

    def sample_action_space(self):
        return torch.tensor(self.action_space.sample())

    def step(self, action):
        obs, reward, termination, truncation, info = self.env.step(action)
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(reward, dtype=torch.float32), termination, truncation, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return torch.tensor(obs, dtype=torch.float32), info


class Buffer:
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device("cuda:0")
        self._capacity = min(cfg.buffer_size, cfg.steps)
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
            strict_length=True,
            cache_values=False,
        )
        self._batch_size = cfg.batch_size * (cfg.horizon + 1)
        self._num_eps = 0

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=False,
            prefetch=0,
            batch_size=self._batch_size,
        )

    def _init(self, tds):
        """Initialize the replay buffer. Use the first episode to estimate storage requirements."""
        print(f"Buffer capacity: {self._capacity:,}")
        mem_free, _ = torch.cuda.mem_get_info()
        bytes_per_step = sum(
            [
                (
                    v.numel() * v.element_size()
                    if not isinstance(v, TensorDict)
                    else sum([x.numel() * x.element_size() for x in v.values()])
                )
                for v in tds.values()
            ]
        ) / len(tds)
        total_bytes = bytes_per_step * self._capacity
        print(f"Storage required: {total_bytes/1e9:.2f} GB")
        # Heuristic: decide whether to use CUDA or CPU memory
        storage_device = "cuda:0" if 2.5 * total_bytes < mem_free else "cpu"
        print(f"Using {storage_device.upper()} memory for storage.")
        self._storage_device = torch.device(storage_device)
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=self._storage_device)
        )

    def load(self, td):
        """
        Load a batch of episodes into the buffer. This is useful for loading data from disk,
        and is more efficient than adding episodes one by one.
        """
        num_new_eps = len(td)
        episode_idx = torch.arange(
            self._num_eps, self._num_eps + num_new_eps, dtype=torch.int64
        )
        td["episode"] = episode_idx.unsqueeze(-1).expand(
            -1, td["reward"].shape[1]
        )
        if self._num_eps == 0:
            self._buffer = self._init(td[0])
        td = td.reshape(td.shape[0] * td.shape[1])
        self._buffer.extend(td)
        self._num_eps += num_new_eps
        return self._num_eps

    def add(self, td):
        """Add an episode to the buffer."""
        td["episode"] = torch.full_like(
            td["reward"], self._num_eps, dtype=torch.int64
        )
        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.extend(td)
        self._num_eps += 1
        return self._num_eps

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        td = td.select("obs", "action", "reward", "task", strict=False).to(
            self._device, non_blocking=True
        )
        obs = td.get("obs").contiguous()
        action = td.get("action")[1:].contiguous()
        reward = td.get("reward")[1:].unsqueeze(-1).contiguous()
        task = td.get("task", None)
        if task is not None:
            task = task[0].contiguous()
        return obs, action, reward, task

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td = self._buffer.sample().view(-1, self.cfg.horizon + 1).permute(1, 0)
        return self._prepare_batch(td)


class OnlineTrainer:
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, cfg, env, agent, buffer, logger : LoggerBase | None = None, timer: Timer = Timer()):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.timer = timer
        print("Architecture:", self.agent.model)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time.time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time.time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, _ = self.env.reset()
            done, ep_reward, t = False, 0, 0
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, termination, truncation, info = self.env.step(
                    action
                )
                done = termination or truncation
                ep_reward += reward
                t += 1
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )

    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.sample_action_space(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        td = TensorDict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
            batch_size=(1,),
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        done, eval_next = True, False
        steps_in_episode = 0
        self.timer.start("training")
        self.timer.start("seed_acquisition")
        for self._step in trange(self._step, self.cfg.steps, disable=not self.cfg.progress_bar):
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = False # FIXME: originally True

            # Reset environment
            if done:
                if eval_next:
                    self.timer.start("eval")
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    # TODO: log? evaluate at all?
                    eval_next = False
                    self.timer.stop("eval")

                if self._step > 0:
                    episode_reward = torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum()
                    episode_success = info["success"]
                    if self.logger is not None:
                        self.logger.record_stat("return", value=episode_reward)
                        self.logger.record_stat("success", value=episode_success)
                        self.logger.stop_episode(steps_in_episode)

                    steps_in_episode = 0

                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                if self.logger is not None:
                    self.logger.start_new_episode()

                obs, _ = self.env.reset()
                self._tds = [self.to_td(obs)]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                self.timer.start("agent_act")
                action = self.agent.act(obs, t0=len(self._tds) == 1)
                self.timer.stop("agent_act")
            else:
                self.timer.start("env_sample_action_space")
                action = self.env.sample_action_space()
                self.timer.stop("env_sample_action_space")
            self.timer.start("env_step")
            obs, reward, termination, truncation, info = self.env.step(action)
            self.timer.stop("env_step")
            done = termination or truncation
            self._tds.append(self.to_td(obs, action, reward))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    self.timer.stop("seed_acquisition")
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = 1
                for i in range(num_updates):
                    self.timer.start("agent_update")
                    metrics = self.agent.update(self.buffer)
                    self.timer.stop("agent_update")
                    if i == num_updates - 1:
                        for k, v in metrics.items():
                            self.logger.record_stat(k, v)

            steps_in_episode += 1

        # End last (potentially partial) episode # TODO: necessary?
        if self.logger is not None:
            self.timer.stop("training")
            self.timer.log(self.logger)
            self.logger.stop_episode(steps_in_episode)


class TDMPC2(torch.nn.Module):
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        self.model = WorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam(
            [
                {
                    "params": self.model._encoder.parameters(),
                    "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                },
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                {"params": self.model._Qs.parameters()},
                {"params": []},
            ],
            lr=self.cfg.lr,
            capturable=True,
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(),
            lr=self.cfg.lr,
            eps=1e-5,
            capturable=True,
        )
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.discount = self._get_discount(cfg.episode_length)
        self._prev_mean = torch.nn.Buffer(
            torch.zeros(
                self.cfg.horizon, self.cfg.action_dim, device=self.device
            )
        )
        if cfg.compile:
            print("Compiling update function with torch.compile...")
            self._update = torch.compile(self._update, mode="reduce-overhead")

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        if self.cfg.compile:
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
                episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
                float: Discount factor for the task.
        """
        frac = episode_length / self.cfg.discount_denom
        return min(
            max((frac - 1) / (frac), self.cfg.discount_min),
            self.cfg.discount_max,
        )

    def save(self, fp):
        """
        Save state dict of the agent to filepath.

        Args:
                fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
                fp (str or dict): Filepath or state dict to load.
        """
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(
                fp, map_location=torch.get_default_device(), weights_only=False
            )
        state_dict = (
            state_dict["model"] if "model" in state_dict else state_dict
        )
        state_dict = api_model_conversion(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict)
        return

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.

        Args:
                obs (torch.Tensor): Observation from the environment.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (int): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        if self.cfg.mpc:
            return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
        z = self.model.encode(obs, task)
        action, info = self.model.pi(z, task)
        if eval_mode:
            action = info["mean"]
        return action[0].cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = two_hot_inv(
                self.model.reward(z, actions[t], task),
                self.cfg.vmin,
                self.cfg.vmax,
                self.cfg.num_bins,
            )
            z = self.model.next(z, actions[t], task)
            G = G + discount * reward
            discount_update = self.discount
            discount = discount * discount_update
        action, _ = self.model.pi(z, task)
        return G + discount * self.model.Q(z, action, task, return_type="avg")

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.

        Args:
                z (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        # Sample policy trajectories
        z = self.model.encode(obs, task)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.cfg.horizon,
                self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=self.device,
            )
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[t], _ = self.model.pi(_z, task)
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1], _ = self.model.pi(_z, task)

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(
            self.cfg.horizon, self.cfg.action_dim, device=self.device
        )
        std = torch.full(
            (self.cfg.horizon, self.cfg.action_dim),
            self.cfg.max_std,
            dtype=torch.float,
            device=self.device,
        )
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(
            self.cfg.horizon,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device,
        )
        if self.cfg.num_pi_trajs > 0:
            actions[:, : self.cfg.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            r = torch.randn(
                self.cfg.horizon,
                self.cfg.num_samples - self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=std.device,
            )
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs :] = actions_sample

            # Compute elite actions
            value = self._estimate_value(z, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(
                value.squeeze(1), self.cfg.num_elites, dim=0
            ).indices
            elite_value, elite_actions = (
                value[elite_idxs],
                actions[:, elite_idxs],
            )

            # Update parameters
            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (
                score.sum(0) + 1e-9
            )
            std = (
                (
                    score.unsqueeze(0)
                    * (elite_actions - mean.unsqueeze(1)) ** 2
                ).sum(dim=1)
                / (score.sum(0) + 1e-9)
            ).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)

        # Select action
        rand_idx = gumbel_softmax_sample(
            score.squeeze(1)
        )  # gumbel_softmax_sample is compatible with cuda graphs
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = actions[0], std[0]
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)

    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.

        Args:
                zs (torch.Tensor): Sequence of latent states.
                task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                float: Loss of the policy update.
        """
        action, info = self.model.pi(zs, task)
        qs = self.model.Q(zs, action, task, return_type="avg", detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = (
            -(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(
                dim=(1, 2)
            )
            * rho
        ).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm
        )
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)

        info = TensorDict(
            {
                "policy loss": pi_loss,
                "policy grad norm": pi_grad_norm,
                "policy entropy": info["entropy"],
                "policy scaled entropy": info["scaled_entropy"],
                "policy scale": self.scale.value,
            }
        )
        return info

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
                next_z (torch.Tensor): Latent state at the following time step.
                reward (torch.Tensor): Reward at the current time step.
                task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: TD-target.
        """
        action, _ = self.model.pi(next_z, task)
        return reward + self.discount * self.model.Q(
            next_z, action, task, return_type="min", target=True
        )

    def _update(self, obs, action, reward, task=None):
        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)

        # Prepare for update
        self.model.train()

        # Latent rollout
        zs = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
            device=self.device,
        )
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(
            zip(action.unbind(0), next_z.unbind(0), strict=False)
        ):
            z = self.model.next(z, _action, task)
            consistency_loss = (
                consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
            )
            zs[t + 1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type="all")
        reward_preds = self.model.reward(_zs, action, task)

        # Compute losses
        reward_loss, value_loss = 0, 0
        for t, (
            rew_pred_unbind,
            rew_unbind,
            td_targets_unbind,
            qs_unbind,
        ) in enumerate(
            zip(
                reward_preds.unbind(0),
                reward.unbind(0),
                td_targets.unbind(0),
                qs.unbind(1),
                strict=False,
            )
        ):
            reward_loss = (
                reward_loss
                + soft_ce(
                    rew_pred_unbind,
                    rew_unbind,
                    self.cfg.vmin,
                    self.cfg.vmax,
                    self.cfg.bin_size,
                    self.cfg.num_bins,
                ).mean()
                * self.cfg.rho**t
            )
            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = (
                    value_loss
                    + soft_ce(
                        qs_unbind_unbind,
                        td_targets_unbind,
                        self.cfg.vmin,
                        self.cfg.vmax,
                        self.cfg.bin_size,
                        self.cfg.num_bins,
                    ).mean()
                    * self.cfg.rho**t
                )

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Update policy
        pi_info = self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        info = TensorDict(
            {
                "consistency loss": consistency_loss,
                "reward loss": reward_loss,
                "q loss": value_loss,
                "total loss": total_loss,
                "grad norm": grad_norm,
            }
        )
        info.update(pi_info)
        return info.detach().mean()

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
                buffer (common.buffer.Buffer): Replay buffer.

        Returns:
                dict: Dictionary of training statistics.
        """
        obs, action, reward, task = buffer.sample()
        kwargs = {}
        if task is not None:
            kwargs["task"] = task
        torch.compiler.cudagraph_mark_step_begin()
        return self._update(obs, action, reward, **kwargs)


class WorldModel(nn.Module):
    """TD-MPC2 implicit world model architecture.

    The world model consists of

    * an encoder that maps observations to latent states
    * a dynamics model that predicts the next latent state given the current
      latent state and action
    * a reward model that predicts the reward given the current latent state
    * a policy prior that predicts the action given the current latent state
    * a Q-function ensemble that predicts the value of a given action
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = enc(cfg)
        self._dynamics = mlp(
            cfg.latent_dim + cfg.action_dim,
            2 * [cfg.mlp_dim],
            cfg.latent_dim,
            act=SimNorm(cfg),
        )
        self._reward = mlp(
            cfg.latent_dim + cfg.action_dim,
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1),
        )
        self._pi = mlp(
            cfg.latent_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim
        )
        self._Qs = Ensemble(
            [
                mlp(
                    cfg.latent_dim + cfg.action_dim,
                    2 * [cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_q)
            ]
        )
        self.apply(weight_init)
        zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer(
            "log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min
        )
        self.init()

    def init(self):
        # Create params
        self._detach_Qs_params = TensorDictParams(
            self._Qs.params.data, no_convert=True
        )
        self._target_Qs_params = TensorDictParams(
            self._Qs.params.data.clone(), no_convert=True
        )

        # Create modules
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)

        # Assign params to modules
        # We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params

    def __repr__(self):
        repr = "TD-MPC2 World Model\n"
        modules = [
            "Encoder",
            "Dynamics",
            "Reward",
            "Policy prior",
            "Q-functions",
        ]
        for i, m in enumerate(
            [self._encoder, self._dynamics, self._reward, self._pi, self._Qs]
        ):
            repr += f"{modules[i]}: {m}\n"
        repr += f"Learnable parameters: {self.total_params:,}"
        return repr

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        # Gaussian policy prior
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = safe_log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        log_prob = gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1]
        scaled_log_prob = log_prob * size

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict(
            {
                "mean": mean,
                "log_std": log_std,
                "action_prob": 1.0,
                "entropy": -log_prob,
                "scaled_entropy": -log_prob * entropy_scale,
            }
        )
        return action, info

    def Q(self, z, a, task, return_type="min", target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
                - `min`: return the minimum of two randomly subsampled Q-values.
                - `avg`: return the average of two randomly subsampled Q-values.
                - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {"min", "avg", "all"}

        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == "all":
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = two_hot_inv(
            out[qidx], self.cfg.vmin, self.cfg.vmax, self.cfg.num_bins
        )
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        # combine_state_for_ensemble causes graph breaks
        self.params = from_modules(*modules, as_module=True)
        with self.params[0].data.to("meta").to_module(modules[0]):
            self.module = deepcopy(modules[0])
        self._repr = str(modules[0])
        self._n = len(modules)

    def __len__(self):
        return self._n

    def _call(self, params, *args, **kwargs):
        with params.to_module(self.module):
            return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return torch.vmap(self._call, (0, None), randomness="different")(
            self.params, *args, **kwargs
        )

    def __repr__(self):
        return f"Vectorized {len(self)}x " + self._repr


class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad
        self.padding = tuple([self.pad] * 4)

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        x = F.pad(x, self.padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=x.dtype,
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0,
            2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=x.dtype,
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0).sub(0.5)


class SimNorm(nn.Module):
    """Simplicial normalization.

    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0.0, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        if act is None:
            act = nn.Mish(inplace=False)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(
            NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0))
        )
    mlp.append(
        NormedLinear(dims[-2], dims[-1], act=act)
        if act
        else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
    """
    Basic convolutional encoder for TD-MPC2 with raw image observations.
    4 layers of convolution with ReLU activations, followed by a linear layer.
    """
    assert in_shape[-1] == 64  # assumes rgb observations to be 64x64
    layers = [
        ShiftAug(),
        PixelPreprocess(),
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2),
        nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 5, stride=2),
        nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 3, stride=2),
        nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 3, stride=1),
        nn.Flatten(),
    ]
    if act:
        layers.append(act)
    return nn.Sequential(*layers)


def enc(cfg, out={}):
    """
    Returns a dictionary of encoders for each observation in the dict.
    """
    for k in cfg.obs_shape.keys():
        if k == "state":
            out[k] = mlp(
                cfg.obs_shape[k][0],
                max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                cfg.latent_dim,
                act=SimNorm(cfg),
            )
        elif k == "rgb":
            out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
        else:
            raise NotImplementedError(
                f"Encoder for observation type {k} not implemented."
            )
    return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
    """
    Converts a checkpoint from our old API to the new torch.compile compatible API.
    """
    # check whether checkpoint is already in the new format
    if "_detach_Qs_params.0.weight" in source_state_dict:
        return source_state_dict

    name_map = ["weight", "bias", "ln.weight", "ln.bias"]
    new_state_dict = dict()

    # rename keys
    for key, val in list(source_state_dict.items()):
        if key.startswith("_Qs."):
            num = key[len("_Qs.params.") :]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            new_total_key = "_Qs.params." + new_key
            del source_state_dict[key]
            new_state_dict[new_total_key] = val
            new_total_key = "_detach_Qs_params." + new_key
            new_state_dict[new_total_key] = val
        elif key.startswith("_target_Qs."):
            num = key[len("_target_Qs.params.") :]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            new_total_key = "_target_Qs_params." + new_key
            del source_state_dict[key]
            new_state_dict[new_total_key] = val

    # add batch_size and device from target_state_dict to new_state_dict
    for prefix in ("_Qs.", "_detach_Qs_", "_target_Qs_"):
        for key in ("__batch_size", "__device"):
            new_key = prefix + "params." + key
            new_state_dict[new_key] = target_state_dict[new_key]

    # check that every key in new_state_dict is in target_state_dict
    for key in new_state_dict:
        assert key in target_state_dict, f"key {key} not in target_state_dict"
    # check that all Qs keys in target_state_dict are in new_state_dict
    for key in target_state_dict.keys():
        if "Qs" in key:
            assert key in new_state_dict, f"key {key} not in new_state_dict"
    # check that source_state_dict contains no Qs keys
    for key in source_state_dict.keys():
        assert "Qs" not in key, f"key {key} contains 'Qs'"

    # copy log_std_min and log_std_max from target_state_dict to new_state_dict
    new_state_dict["log_std_min"] = target_state_dict["log_std_min"]
    new_state_dict["log_std_dif"] = target_state_dict["log_std_dif"]
    new_state_dict["_action_masks"] = target_state_dict["_action_masks"]

    # copy new_state_dict to source_state_dict
    source_state_dict.update(new_state_dict)

    return source_state_dict


def weight_init(m):
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.02, 0.02)
    elif isinstance(m, nn.ParameterList):
        for i, p in enumerate(m):
            if p.dim() == 3:  # Linear
                nn.init.trunc_normal_(p, std=0.02)  # Weight
                nn.init.constant_(m[i + 1], 0)  # Bias


def zero_(params):
    """Initialize parameters to zero."""
    for p in params:
        p.data.fill_(0)


Config = namedtuple(
    "Config",
    [
        "task",
        "obs",
        "eval_episodes",
        "eval_freq",
        "steps",
        "batch_size",
        "reward_coef",
        "value_coef",
        "consistency_coef",
        "rho",
        "lr",
        "enc_lr_scale",
        "grad_clip_norm",
        "tau",
        "discount_denom",
        "discount_min",
        "discount_max",
        "buffer_size",
        "exp_name",
        "mpc",
        "iterations",
        "num_samples",
        "num_elites",
        "num_pi_trajs",
        "horizon",
        "min_std",
        "max_std",
        "temperature",
        "log_std_min",
        "log_std_max",
        "entropy_coef",
        "num_bins",
        "vmin",
        "vmax",
        "model_size",
        "num_enc_layers",
        "enc_dim",
        "num_channels",
        "mlp_dim",
        "latent_dim",
        "num_q",
        "dropout",
        "simnorm_dim",
        "compile",
        # internal configuration
        "obs_shape",
        "tasks",
        "work_dir",
        "task_title",
        "bin_size",
        "action_dim",
        "episode_length",
        "seed_steps",
        "progress_bar",
    ],
)


def train_tdmpc2(
    env,
    task,
    obs="state",
    # eval
    eval_episodes=10,
    eval_freq=50_000,
    steps=10_000_000,
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
    model_size=5,  # 1, 5, 19, 48, 317
    num_enc_layers=2,
    enc_dim=256,
    num_channels=32,
    mlp_dim=512,
    latent_dim=512,
    num_q=5,
    dropout=0.01,
    simnorm_dim=8,
    # misc
    seed=1,
    # speedups
    compile=False,
    progress_bar=True,
    logger: LoggerBase | None = None,
    timer: Timer = Timer(),
) -> TDMPC2:
    """TD-MPC2.

    Parameters
    ----------
    env : gymnasium.Env
        Training environment.
    task : str
        Task name.
    obs : str in ["state", "rgb"]
        Observation type.
    eval_episodes : int
        Number of evaluation episodes.
    eval_freq : int
        Evaluate every eval_freq steps.
    steps : int
        Number of training / environment steps
    batch_size : int
        Batch size.
    reward_coef
        TODO
    value_coef
        TODO
    consistency_coef
        TODO
    rho
        TODO
    lr
        Learning rate.
    enc_lr_scale
        Scaling for encoder learning rate.
    grad_clip_norm
        Clip the gradients during backpropagation.
    tau
        For Polyak averaging. Determines the interpolation factor between
        the current parameters of the target network and the parameters of the
        main network.
    discount_denom
        TODO
    discount_min
        TODO
    discount_max
        TODO
    buffer_size : int
        Size of the replay buffer.
    exp_name : str
        Name of the experiment.
    mpc : bool
        TODO
    iterations : int
        Number of iterations to optimize plan. We add 2 iterations for large
        action spaces (>= 20 dimensions).
    num_samples : int
        Number of samples for MPC planning.
    num_elites : int
        Number of samples to use for update of search distribution in planning.
    num_pi_trajs : int
        Number of samples generated with policy.
    horizon : int
        Planning horizon.
    min_std : float
        TODO
    max_std : float
        TODO
    temperature : float
        Temperature for planning with MPPI.
    log_std_min : float
        Minimum of log std for actor.
    log_std_max : float
        Maximum of log std for actor.
    entropy_coef : float
        Entropy coefficient for policy update.
    num_bins : float
        TODO
    vmin : float
        TODO
    vmax : float
        TODO
    model_size : int
        Model size, must be one of [1, 5, 19, 48, 317]. If none, use values for
        num_enc_layers, enc_dim, num_channels, and mlp_dim, latent_dim, and
        num_q to define model architecture.
    num_enc_layers : int
        Number of layers in encoder.
    enc_dim : int
        Number of nodes in encoder layers.
    num_channels : int
        Number of channels for convolutional encoder with raw image
        observations.
    mlp_dim : int
        Number of hidden nodes in dynamics model, reward model, policy, and Q
        network.
    latent_dim : int
        Dimensions of latent space to which the encoder projects.
    num_q : int
        Number of networks in the ensemble of Q-functions.
    dropout : float
        Dropout probability for Q-functions.
    simnorm_dim : int
        Number of dimensions for simplicial normalization in encoder.
    seed : int
        Random seed
    compile : bool
        Compile graphs for faster training.
    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.
    logger : LoggerBase, optional
        Experiment logger.
    """
    assert torch.cuda.is_available()

    # Check parameters, compute defaults, and create configuration object
    assert steps > 0, "Must train for at least 1 step."

    try:  # Dict
        obs_shape = {
            k: v.shape for k, v in env.observation_space.spaces.items()
        }
    except:  # Box
        obs_shape = {obs: env.observation_space.shape}
    tasks = TASK_SET.get(task, [task])
    work_dir = Path(".") / "logs" / task / str(seed) / exp_name
    task_title = task.replace("-", " ").title()
    # Bin size for discrete regression
    bin_size = (vmax - vmin) / (num_bins - 1)
    # Model size
    if model_size is not None:
        if model_size not in MODEL_SIZE:
            raise ValueError(f"Invalid model size {model_size}. "
                             f"Must be one of {list(MODEL_SIZE.keys())}")
        for k, v in MODEL_SIZE[model_size].items():
            enc_dim = MODEL_SIZE[model_size]["enc_dim"]
            mlp_dim = MODEL_SIZE[model_size]["mlp_dim"]
            latent_dim = MODEL_SIZE[model_size]["latent_dim"]
            num_enc_layers = MODEL_SIZE[model_size]["num_enc_layers"]
            num_q = MODEL_SIZE[model_size]["num_q"]
    action_dim = env.action_space.shape[0]
    episode_length = env.spec.max_episode_steps
    seed_steps = max(1000, 5 * episode_length)
    # Heuristic for large action spaces
    iterations += 2 * int(action_dim >= 20)

    cfg = Config(
        task=task,
        obs=obs,
        eval_episodes=eval_episodes,
        eval_freq=eval_freq,
        steps=steps,
        batch_size=batch_size,
        reward_coef=reward_coef,
        value_coef=value_coef,
        consistency_coef=consistency_coef,
        rho=rho,
        lr=lr,
        enc_lr_scale=enc_lr_scale,
        grad_clip_norm=grad_clip_norm,
        tau=tau,
        discount_denom=discount_denom,
        discount_min=discount_min,
        discount_max=discount_max,
        buffer_size=buffer_size,
        exp_name=exp_name,
        mpc=mpc,
        iterations=iterations,
        num_samples=num_samples,
        num_elites=num_elites,
        num_pi_trajs=num_pi_trajs,
        horizon=horizon,
        min_std=min_std,
        max_std=max_std,
        temperature=temperature,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        entropy_coef=entropy_coef,
        num_bins=num_bins,
        vmin=vmin,
        vmax=vmax,
        model_size=model_size,
        num_enc_layers=num_enc_layers,
        enc_dim=enc_dim,
        num_channels=num_channels,
        mlp_dim=mlp_dim,
        latent_dim=latent_dim,
        num_q=num_q,
        dropout=dropout,
        simnorm_dim=simnorm_dim,
        compile=compile,
        obs_shape=obs_shape,
        tasks=tasks,
        work_dir=work_dir,
        task_title=task_title,
        bin_size=bin_size,
        action_dim=action_dim,
        episode_length=episode_length,
        seed_steps=seed_steps,
        progress_bar=progress_bar,
    )

    set_seed(seed)

    gym.logger.min_level = 40
    env = DefaultSuccessInfoWrapper(env)
    env = NumpyToTorch(env)
    env = NumpyToTorchSpaces(env)

    print(colored("Work dir:", "yellow", attrs=["bold"]), work_dir)

    trainer = OnlineTrainer(
        cfg=cfg,
        env=env,
        agent=TDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=logger,
        timer=timer,
    )
    trainer.train()
    print("\nTraining completed successfully")
    return trainer.agent


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
