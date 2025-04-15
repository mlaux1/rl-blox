# Dependencies:
# pip install stable-baselines3 tensorflow tf-keras tensorflow-probability[jax]

# The MIT License
#
# CrossQ modifications copyright (c) 2023 Aditya Bhatt and Daniel Palenicek
# Copyright (c) 2022 Antonin Raffin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import time
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from typing import Callable, Sequence
from typing import ClassVar
from typing import NamedTuple
from typing import no_type_check

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from flax.linen.module import Module, compact, merge_param  # pylint: disable=g-multiple-import
from flax.linen.normalization import _compute_stats, _normalize, _canonicalize_axes
from flax.training.train_state import TrainState
from gymnasium import spaces
from jax.nn import initializers
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import is_vectorized_observation

tfd = tfp.distributions

PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]


################################################################################
# common.off_policy_algorithm.py
################################################################################


class OffPolicyAlgorithmJax(OffPolicyAlgorithm):
    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        qf_learning_rate: Optional[float] = None,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: str = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        stats_window_size: int = 100,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            action_noise=action_noise,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            sde_support=sde_support,
            supported_action_spaces=supported_action_spaces,
            support_multi_env=support_multi_env,
            stats_window_size=stats_window_size,
        )
        # Will be updated later
        self.key = jax.random.PRNGKey(0)
        # Note: we do not allow schedule for it
        self.qf_learning_rate = qf_learning_rate

    def _get_torch_save_params(self):
        return [], []

    def _excluded_save_params(self) -> List[str]:
        excluded = super()._excluded_save_params()
        excluded.remove("policy")
        return excluded

    def set_random_seed(self, seed: Optional[int]) -> None:  # type: ignore[override]
        super().set_random_seed(seed)
        if seed is None:
            # Sample random seed
            seed = np.random.randint(2**14)
        self.key = jax.random.PRNGKey(seed)

    def _setup_model(self) -> None:
        if self.replay_buffer_class is None:  # type: ignore[has-type]
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        self._setup_lr_schedule()
        # By default qf_learning_rate = pi_learning_rate
        self.qf_learning_rate = self.qf_learning_rate or self.lr_schedule(1)
        self.set_random_seed(self.seed)
        # Make a local copy as we should not pickle
        # the environment when using HerReplayBuffer
        replay_buffer_kwargs = deepcopy(self.replay_buffer_kwargs)
        if issubclass(self.replay_buffer_class, HerReplayBuffer):  # type: ignore[arg-type]
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
            replay_buffer_kwargs["env"] = self.env

        self.replay_buffer = self.replay_buffer_class(  # type: ignore[misc]
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device="cpu",  # force cpu device to easy torch -> numpy conversion
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **replay_buffer_kwargs,
        )
        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()


################################################################################
# common.type_aliases.py
################################################################################

class ActorTrainState(TrainState):
    batch_stats: flax.core.FrozenDict

class RLTrainState(TrainState):  # type: ignore[misc]
    target_params: flax.core.FrozenDict  # type: ignore[misc]
    batch_stats: flax.core.FrozenDict
    target_batch_stats: flax.core.FrozenDict


class ReplayBufferSamplesNp(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray


################################################################################
# common.policies.py
################################################################################


class BaseJaxPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["return_logprob"])
    def sample_action(actor_state, obervations, key, return_logprob=False):
        if hasattr(actor_state, "batch_stats"):
            dist = actor_state.apply_fn({"params": actor_state.params, "batch_stats": actor_state.batch_stats},
                                        obervations, train=False)
        else:
            dist = actor_state.apply_fn(actor_state.params, obervations)
        action = dist.sample(seed=key)
        if not return_logprob:
            return action
        else:
            return action, dist.log_prob(action)

    @staticmethod
    @partial(jax.jit, static_argnames=["return_logprob"])
    def select_action(actor_state, obervations, return_logprob=False):
        if hasattr(actor_state, "batch_stats"):
            dist = actor_state.apply_fn({"params": actor_state.params, "batch_stats": actor_state.batch_stats},
                                        obervations, train=False)
        else:
            dist = actor_state.apply_fn(actor_state.params, obervations)
        action = dist.mode()

        if not return_logprob:
            return action
        else:
            return action, dist.log_prob(action)

    @no_type_check
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # self.set_training_mode(False)

        observation, vectorized_env = self.prepare_obs(observation)

        actions = self._predict(observation, deterministic=deterministic)

        # Convert to numpy, and reshape to the original action shape
        actions = np.array(actions).reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Clip due to numerical instability
                actions = np.clip(actions, -1, 1)
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)  # type: ignore[call-overload]

        return actions, state

    def prepare_obs(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(self.observation_space, spaces.Dict)
            # Minimal dict support: flatten
            keys = list(self.observation_space.keys())
            vectorized_env = is_vectorized_observation(observation[keys[0]], self.observation_space[keys[0]])

            # Add batch dim and concatenate
            observation = np.concatenate(
                [observation[key].reshape(-1, *self.observation_space[key].shape) for key in keys],
                axis=1,
            )
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            # observation = copy.deepcopy(observation)
            # for key, obs in observation.items():
            #     obs_space = self.observation_space.spaces[key]
            #     if is_image_space(obs_space):
            #         obs_ = maybe_transpose(obs, obs_space)
            #     else:
            #         obs_ = np.array(obs)
            #     vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
            #     # Add batch dimension if needed
            #     observation[key] = obs_.reshape((-1, *self.observation_space[key].shape))

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(self.observation_space, spaces.Dict):
            assert isinstance(observation, np.ndarray)
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))  # type: ignore[misc]

        assert isinstance(observation, np.ndarray)
        return observation, vectorized_env

    def set_training_mode(self, mode: bool) -> None:
        # self.actor.set_training_mode(mode)
        # self.critic.set_training_mode(mode)
        self.training = mode


################################################################################
# common.distributions.py
################################################################################


class TanhTransformedDistribution(tfd.TransformedDistribution):  # type: ignore[name-defined]
    """
    From https://github.com/ikostrikov/walk_in_the_park
    otherwise mode is not defined for Squashed Gaussian
    """

    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):  # type: ignore[name-defined]
        super().__init__(distribution=distribution, bijector=tfp.bijectors.Tanh(), validate_args=validate_args)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


################################################################################
# policies.py
################################################################################

class BatchRenorm(Module):
    """BatchRenorm Module, implemented based on the Batch Renormalization paper (https://arxiv.org/abs/1702.03275).
    and adapted from Flax's BatchNorm implementation:
    https://github.com/google/flax/blob/ce8a3c74d8d1f4a7d8f14b9fb84b2cc76d7f8dbf/flax/linen/normalization.py#L228


    Attributes:
      use_running_average: if True, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of the batch
        statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  if True, bias (beta) is added.
      use_scale: if True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
        examples on the first two and last two devices. See `jax.lax.psum` for
        more details.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.999
    epsilon: float = 0.001
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """
        Args:
          x: the input to be normalized.
          use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.

        Returns:
          Normalized inputs (the same shape as inputs).
        """

        use_running_average = merge_param(
            'use_running_average', self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            'batch_stats',
            'mean',
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            'batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        r_max = self.variable(
            'batch_stats',
            'r_max',
            lambda s: s,
            3,
        )
        d_max = self.variable(
            'batch_stats',
            'd_max',
            lambda s: s,
            5,
        )
        steps = self.variable(
            'batch_stats',
            'steps',
            lambda s: s,
            0,
        )

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
            custom_mean = mean
            custom_var = var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var
            if not self.is_initializing():
                # The code below is implemented following the Batch Renormalization paper
                r = 1
                d = 0
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r ** 2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                # Warm up batch renorm for 100_000 steps to build up proper running statistics
                warmed_up = jnp.greater_equal(steps.value, 100_000).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + (1. - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + (1. - warmed_up) * custom_mean

                ra_mean.value = (
                        self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


class Critic(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    batch_norm_momentum: float
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    use_batch_norm: bool = False
    bn_mode: str = "bn"

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, train) -> jnp.ndarray:
        if 'bn' in self.bn_mode:
            BN = nn.BatchNorm
        elif 'brn' in self.bn_mode:
            BN = BatchRenorm
        else:
            raise NotImplementedError

        x = jnp.concatenate([x, action], -1)

        if self.use_batch_norm:
            x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        else:
            # Hack to make flax return state_updates. Is only necessary such that the downstream
            # functions have the same function signature.
            x_dummy = BN(use_running_average=not train)(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)

            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

            if self.use_layer_norm:
                x = nn.LayerNorm()(x)

            x = self.activation_fn()(x)

            if self.use_batch_norm:
                x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
            else:
                x_dummy = BN(use_running_average=not train)(x)
        x = nn.Dense(1)(x)
        return x


class VectorCritic(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    batch_norm_momentum: float
    use_batch_norm: bool = False
    batch_norm_mode: str = "bn"
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool = True):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "dropout": True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            use_layer_norm=self.use_layer_norm,
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            bn_mode=self.batch_norm_mode,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )(obs, action, train)
        return q_values


class Actor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    batch_norm_momentum: float
    log_std_min: float = -20
    log_std_max: float = 2
    use_batch_norm: bool = False
    bn_mode: str = "bn"

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    # type: ignore[name-defined]
    def __call__(self, x: jnp.ndarray, train) -> tfd.Distribution:

        if 'brn_actor' in self.bn_mode:
            BN = BatchRenorm
        elif 'bn' in self.bn_mode or 'brn' in self.bn_mode:
            BN = nn.BatchNorm
        else:
            raise NotImplementedError

        if self.use_batch_norm and not 'noactor' in self.bn_mode:
            x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        else:
            # Hack to make flax return state_updates. Is only necessary such that the downstream
            # functions have the same function signature.
            x_dummy = BN(use_running_average=not train)(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)
            if self.use_batch_norm and not 'noactor' in self.bn_mode:
                x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
            else:
                x_dummy = BN(use_running_average=not train)(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )
        return dist


class SACPolicy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            lr_schedule: Schedule,
            activation_fn: Type[nn.Module],
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            dropout_rate: float = 0.0,
            layer_norm: bool = False,
            batch_norm: bool = False,
            batch_norm_momentum: float = 0.9,
            batch_norm_mode: str = "bn",
            use_sde: bool = False,
            # Note: most gSDE parameters are not used
            # this is to keep API consistent with SB3
            log_std_init: float = -3,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class=None,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Callable[...,
            optax.GradientTransformation] = optax.adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = False,
            td3_mode: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_mode = batch_norm_mode
        self.activation_fn = activation_fn
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_qf = net_arch
            else:
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_qf = net_arch["qf"]
        else:
            self.net_arch_pi = self.net_arch_qf = [256, 256]
        self.n_critics = n_critics
        self.use_sde = use_sde

        self.key = self.noise_key = jax.random.PRNGKey(0)

        if td3_mode:
            self._predict = self._predict_deterministic

    def build(self, key: jnp.ndarray, lr_schedule: Schedule, qf_learning_rate: float) -> jnp.ndarray:
        key, actor_key, qf_key, dropout_key, bn_key = jax.random.split(key, 5)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array(
                [spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        self.actor = Actor(
            action_dim=int(np.prod(self.action_space.shape)),
            net_arch=self.net_arch_pi,
            use_batch_norm=self.batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            bn_mode=self.batch_norm_mode,
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        # params=self.actor.init(actor_key, obs)
        actor_init_variables = self.actor.init(
            {"params": actor_key, "batch_stats": bn_key},
            obs,
            train=False
        )
        self.actor_state = ActorTrainState.create(
            apply_fn=self.actor.apply,
            params=actor_init_variables["params"],
            batch_stats=actor_init_variables["batch_stats"],
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.qf = VectorCritic(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            use_batch_norm=self.batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_mode=self.batch_norm_mode,
            net_arch=self.net_arch_qf,
            activation_fn=self.activation_fn,
            n_critics=self.n_critics,
        )

        qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        target_qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=qf_init_variables["params"],
            batch_stats=qf_init_variables["batch_stats"],
            target_params=target_qf_init_variables["params"],
            target_batch_stats=target_qf_init_variables["batch_stats"],
            tx=self.optimizer_class(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.actor.apply = jax.jit(  # type: ignore[method-assign]
            self.actor.apply,
            static_argnames=("use_batch_norm", "batch_norm_momentum", "bn_mode")
        )
        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm",
                             "use_batch_norm", "batch_norm_momentum", "bn_mode"),
        )

        return key

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    # type: ignore[override]
    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return BaseJaxPolicy.select_action(self.actor_state, observation)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        return BaseJaxPolicy.sample_action(self.actor_state, observation, self.noise_key)

    def _predict_deterministic(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        return BaseJaxPolicy.select_action(self.actor_state, observation)

    def predict_action_with_logprobs(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return BaseJaxPolicy.select_action(self.actor_state, observation, True)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()

        return BaseJaxPolicy.sample_action(self.actor_state, observation, self.noise_key, True)

    def predict_critic(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:

        if not self.use_sde:
            self.reset_noise()

        def Q(params, batch_stats, o, a, dropout_key):
            return self.qf_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                o, a,
                rngs={"dropout": dropout_key},
                train=False
            )

        return jax.jit(Q)(
            self.qf_state.params,
            self.qf_state.batch_stats,
            observation,
            action,
            self.noise_key,
        )


################################################################################
# utils.py
################################################################################


def is_slurm_job():
    """Checks whether the script is run within slurm"""
    return bool(len({k: v for k, v in os.environ.items() if 'SLURM' in k}))


class ReLU(nn.Module):
    def __call__(self, x):
        return nn.relu(x)


class ReLU6(nn.Module):
    def __call__(self, x):
        return nn.relu6(x)


class Tanh(nn.Module):
    def __call__(self, x):
        return nn.tanh(x)


class Sin(nn.Module):
    def __call__(self, x):
        return jnp.sin(x)


class Elu(nn.Module):
    def __call__(self, x):
        return nn.elu(x)


class GLU(nn.Module):
    def __call__(self, x):
        return nn.glu(x)


class LayerNormedReLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.LayerNorm()(nn.relu(x))


class ReLUOverMax(nn.Module):
    def __call__(self, x):
        act = nn.relu(x)
        return act / (jnp.max(act) + 1e-6)


activation_fn = {
    # unbounded
    "relu": ReLU,
    "elu": Elu,
    "glu": GLU,
    # bounded
    "tanh": Tanh,
    "sin": Sin,
    "relu6": ReLU6,
    # unbounded with normalizer
    "layernormed_relu": LayerNormedReLU,  # with normalizer
    "relu_over_max": ReLUOverMax,  # with normalizer
}


################################################################################
# sac.py
################################################################################


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


class ConstantEntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> float:
        # Hack to not optimize the entropy coefficient while not having to use if/else for the jit
        self.param("dummy_param", init_fn=lambda key: jnp.full((), self.ent_coef_init))
        return self.ent_coef_init


class SAC(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[Dict[str, Type[SACPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": SACPolicy,
        # Minimal dict support using flatten()
        "MultiInputPolicy": SACPolicy,
    }

    policy: SACPolicy
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
            self,
            policy,
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            qf_learning_rate: Optional[float] = None,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            crossq_style: bool = False,
            td3_mode: bool = False,
            use_bnstats_from_live_net: bool = False,
            policy_q_reduce_fn=jnp.min,
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            policy_delay: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            ent_coef: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: str = "auto",
            _init_setup_model: bool = True,
            stats_window_size: int = 100,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            qf_learning_rate=qf_learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            stats_window_size=stats_window_size,
        )

        self.policy_delay = policy_delay
        self.ent_coef_init = ent_coef
        self.crossq_style = crossq_style
        self.td3_mode = td3_mode
        self.use_bnstats_from_live_net = use_bnstats_from_live_net
        self.policy_q_reduce_fn = policy_q_reduce_fn

        if td3_mode:
            self.action_noise = NormalActionNoise(mean=jnp.zeros(1), sigma=jnp.ones(1) * 0.1)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            # pytype: disable=not-instantiable
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                td3_mode=self.td3_mode,
                **self.policy_kwargs,
            )
            # pytype: enable=not-instantiable

            assert isinstance(self.qf_learning_rate, float)

            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)

            self.key, ent_key = jax.random.split(self.key, 2)

            self.actor = self.policy.actor  # type: ignore[assignment]
            self.qf = self.policy.qf  # type: ignore[assignment]

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if isinstance(self.ent_coef_init, str) and self.ent_coef_init.startswith("auto"):
                # Default initial value of ent_coef when learned
                ent_coef_init = 1.0
                if "_" in self.ent_coef_init:
                    ent_coef_init = float(self.ent_coef_init.split("_")[1])
                    assert ent_coef_init > 0.0, "The initial value of ent_coef must be greater than 0"

                # Note: we optimize the log of the entropy coeff which is slightly different from the paper
                # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                self.ent_coef = EntropyCoef(ent_coef_init)
            else:
                # This will throw an error if a malformed string (different from 'auto') is passed
                assert isinstance(
                    self.ent_coef_init, float
                ), f"Entropy coef must be float when not equal to 'auto', actual: {self.ent_coef_init}"
                self.ent_coef = ConstantEntropyCoef(self.ent_coef_init)  # type: ignore[assignment]

            self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=self.ent_coef.init(ent_key)["params"],
                tx=optax.adam(
                    learning_rate=self.learning_rate,
                ),
            )

        # automatically set target entropy if needed
        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "SAC",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def train(self, batch_size, gradient_steps):
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)
        # Pre-compute the indices where we need to update the actor
        # This is a hack in order to jit the train loop
        # It will compile once per value of policy_delay_indices
        policy_delay_indices = {i: True for i in range(gradient_steps) if
                                ((self._n_updates + i + 1) % self.policy_delay) == 0}
        policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        # Convert to numpy
        data = ReplayBufferSamplesNp(
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
            log_metrics,
        ) = self._train(
            self.crossq_style,
            self.td3_mode,
            self.use_bnstats_from_live_net,
            self.gamma,
            self.tau,
            self.target_entropy,
            gradient_steps,
            data,
            policy_delay_indices,
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
            self.policy_q_reduce_fn,
        )
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for k, v in log_metrics.items():
            self.logger.record(f"train/{k}", v.item())

    @staticmethod
    @partial(jax.jit, static_argnames=["crossq_style", "td3_mode", "use_bnstats_from_live_net"])
    def update_critic(
            crossq_style: bool,
            td3_mode: bool,
            use_bnstats_from_live_net: bool,
            gamma: float,
            actor_state: ActorTrainState,
            qf_state: RLTrainState,
            ent_coef_state: TrainState,
            observations: np.ndarray,
            actions: np.ndarray,
            next_observations: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray,
            key: jnp.ndarray,
    ):
        key, noise_key, dropout_key_target, dropout_key_current, redq_key = jax.random.split(key, 5)
        # sample action from the actor
        dist = actor_state.apply_fn(
            {"params": actor_state.params, "batch_stats": actor_state.batch_stats},
            next_observations, train=False
        )

        if td3_mode:
            ent_coef_value = 0.0
            target_policy_noise = 0.2
            target_noise_clip = 0.5

            next_state_actions = dist.mode()
            noise = jax.random.normal(noise_key, next_state_actions.shape) * target_policy_noise
            noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
            next_state_actions = jnp.clip(next_state_actions + noise, -1.0, 1.0)
            next_log_prob = jnp.zeros(next_state_actions.shape[0])
        else:
            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

            next_state_actions = dist.sample(seed=noise_key)
            next_log_prob = dist.log_prob(next_state_actions)

        def mse_loss(params, batch_stats, dropout_key):
            if not crossq_style:
                next_q_values = qf_state.apply_fn(
                    {
                        "params": qf_state.target_params,
                        "batch_stats": qf_state.target_batch_stats if not use_bnstats_from_live_net else batch_stats
                    },
                    next_observations, next_state_actions,
                    rngs={"dropout": dropout_key_target},
                    train=False
                )

                # shape is (n_critics, batch_size, 1)
                current_q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    observations, actions,
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )

            else:
                # ----- CrossQ's One Weird Trick™ -----
                # concatenate current and next observations to double the batch size
                # new shape of input is (n_critics, 2*batch_size, obs_dim + act_dim)
                # apply critic to this bigger batch
                catted_q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    jnp.concatenate([observations, next_observations], axis=0),
                    jnp.concatenate([actions, next_state_actions], axis=0),
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )
                current_q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)

            if next_q_values.shape[0] > 2:  # only for REDQ
                # REDQ style subsampling of critics.
                m_critics = 2
                next_q_values = jax.random.choice(redq_key, next_q_values, (m_critics,), replace=False, axis=0)

            next_q_values = jnp.min(next_q_values, axis=0)
            next_q_values = next_q_values - ent_coef_value * next_log_prob.reshape(-1, 1)
            target_q_values = rewards.reshape(-1, 1) + (
                        1 - dones.reshape(-1, 1)) * gamma * next_q_values  # shape is (batch_size, 1)

            loss = 0.5 * ((jax.lax.stop_gradient(target_q_values) - current_q_values) ** 2).mean(axis=1).sum()

            return loss, (state_updates, current_q_values, next_q_values)

        (qf_loss_value, (state_updates, current_q_values, next_q_values)), grads = \
            jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params, qf_state.batch_stats, dropout_key_current)

        qf_state = qf_state.apply_gradients(grads=grads)
        qf_state = qf_state.replace(batch_stats=state_updates["batch_stats"])

        metrics = {
            'critic_loss': qf_loss_value,
            'ent_coef': ent_coef_value,
            'current_q_values': current_q_values.mean(),
            'next_q_values': next_q_values.mean(),
        }

        return (qf_state, metrics, key)

    @staticmethod
    @partial(jax.jit, static_argnames=["q_reduce_fn", "td3_mode"])
    def update_actor(
            actor_state: ActorTrainState,
            qf_state: RLTrainState,
            ent_coef_state: TrainState,
            observations: np.ndarray,
            key: jnp.ndarray,
            q_reduce_fn=jnp.min,  # Changes for redq and droq
            td3_mode=False,
    ):
        key, dropout_key, noise_key, redq_key = jax.random.split(key, 4)

        def actor_loss(params, batch_stats):
            dist, state_updates = actor_state.apply_fn({
                "params": params, "batch_stats": batch_stats},
                observations,
                mutable=["batch_stats"],
                train=True
            )

            if td3_mode:
                actor_actions = dist.mode()
                ent_coef_value = 0.0
                log_prob = jnp.zeros(actor_actions.shape[0])
            else:
                actor_actions = dist.sample(seed=noise_key)
                ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
                log_prob = dist.log_prob(actor_actions).reshape(-1, 1)

            qf_pi = qf_state.apply_fn(
                {
                    "params": qf_state.params,
                    "batch_stats": qf_state.batch_stats
                },
                observations,
                actor_actions,
                rngs={"dropout": dropout_key}, train=False
            )

            min_qf_pi = q_reduce_fn(qf_pi, axis=0)

            actor_loss = (ent_coef_value * log_prob - min_qf_pi).mean()
            return actor_loss, (-log_prob.mean(), state_updates)

        (actor_loss_value, (entropy, state_updates)), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_state.params, actor_state.batch_stats)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(batch_stats=state_updates["batch_stats"])

        return actor_state, qf_state, actor_loss_value, key, entropy

    @staticmethod
    @jax.jit
    def soft_update(tau: float, qf_state: RLTrainState):
        qf_state = qf_state.replace(
            target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau))
        qf_state = qf_state.replace(
            target_batch_stats=optax.incremental_update(qf_state.batch_stats, qf_state.target_batch_stats, tau))
        return qf_state

    @staticmethod
    @jax.jit
    def update_temperature(target_entropy: np.ndarray, ent_coef_state: TrainState, entropy: float):
        def temperature_loss(temp_params):
            ent_coef_value = ent_coef_state.apply_fn({"params": temp_params})
            ent_coef_loss = ent_coef_value * (entropy - target_entropy).mean()
            return ent_coef_loss

        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

    @classmethod
    @partial(jax.jit, static_argnames=["cls", "crossq_style", "td3_mode", "use_bnstats_from_live_net", "gradient_steps",
                                       "q_reduce_fn"])
    def _train(
            cls,
            crossq_style: bool,
            td3_mode: bool,
            use_bnstats_from_live_net: bool,
            gamma: float,
            tau: float,
            target_entropy: np.ndarray,
            gradient_steps: int,
            data: ReplayBufferSamplesNp,
            policy_delay_indices: flax.core.FrozenDict,
            qf_state: RLTrainState,
            actor_state: ActorTrainState,
            ent_coef_state: TrainState,
            key,
            q_reduce_fn,
    ):
        actor_loss_value = jnp.array(0)

        for i in range(gradient_steps):

            def slice(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * step: batch_size * (step + 1)]

            (
                qf_state,
                log_metrics_critic,
                key,
            ) = SAC.update_critic(
                crossq_style,
                td3_mode,
                use_bnstats_from_live_net,
                gamma,
                actor_state,
                qf_state,
                ent_coef_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
                slice(data.rewards),
                slice(data.dones),
                key,
            )
            qf_state = SAC.soft_update(tau, qf_state)

            # hack to be able to jit (n_updates % policy_delay == 0)
            if i in policy_delay_indices:
                (actor_state, qf_state, actor_loss_value, key, entropy) = cls.update_actor(
                    actor_state,
                    qf_state,
                    ent_coef_state,
                    slice(data.observations),
                    key,
                    q_reduce_fn,
                    td3_mode,
                )
                ent_coef_state, _ = SAC.update_temperature(target_entropy, ent_coef_state, entropy)

        log_metrics = {
            'actor_loss': actor_loss_value,
            **log_metrics_critic
        }

        return (
            qf_state,
            actor_state,
            ent_coef_state,
            key,
            log_metrics,
        )

    def predict_critic(self, observation, action):
        return self.policy.predict_critic(observation, action)

    def current_entropy_coeff(self):
        return self.ent_coef_state.apply_fn({"params": self.ent_coef_state.params})


def train_crossq(
    env: gym.Env,
    algo: str = "sac",
    seed: int = 1,
    log_freq: int = 300,
    adam_b1: float = 0.5,
    bn: bool = True,
    bn_momentum: float = 0.99,
    bn_mode: str = "brn_actor",
    critic_activation: str = "relu",
    crossq_style: bool = True,
    dropout: int = 1,  # TODO bool?
    ln: float = 0.0,  # TODO bool?
    lr: float = 1e-3,
    n_critics: int = 2,
    n_neurons: int = 256,
    policy_delay: int = 1,
    tau: float = 0.005,
    utd: int = 1,
    total_timesteps: int = 5_000_000,
    bnstats_live_net: int = 0,  # TODO bool?
) -> SAC:
    """CrossQ.

    Parameters
    ----------
    env: gym.Env
        The environment to train on.
    """
    experiment_time = time.time()

    algo = algo.lower()
    tau = tau if not crossq_style else 1.0
    bn_momentum = bn_momentum if bn else 0.0
    dropout_rate, layer_norm = None, False
    policy_q_reduce_fn = jax.numpy.min
    net_arch = {'pi': [256, 256], 'qf': [n_neurons, n_neurons]}
    eval_freq = max(5_000_000 // log_freq, 1)
    td3_mode = False

    if algo == 'droq':
        dropout_rate = 0.01
        layer_norm = True
        policy_q_reduce_fn = jax.numpy.mean
        n_critics = 2
        # adam_b1 = 0.9  # adam default
        adam_b2 = 0.999  # adam default
        policy_delay = 20
        utd = 20
        group = f'DroQ_{env}_bn({bn})_ln{(ln)}_xqstyle({crossq_style}/{tau})_utd({utd}/{policy_delay})_Adam({adam_b1})_Q({net_arch["qf"][0]})'
    elif algo == 'redq':
        policy_q_reduce_fn = jax.numpy.mean
        n_critics = 10
        # adam_b1 = 0.9  # adam default
        adam_b2 = 0.999  # adam default
        policy_delay = 20
        utd = 20
        group = f'REDQ_{env}_bn({bn})_ln{(ln)}_xqstyle({crossq_style}/{tau})_utd({utd}/{policy_delay})_Adam({adam_b1})_Q({net_arch["qf"][0]})'
    elif algo == 'td3':
        # With the right hyperparameters, this here can run all the above algorithms
        # and ablations.
        td3_mode = True
        layer_norm = ln
        if dropout:
            dropout_rate = 0.01
        group = f'TD3_{env}_bn({bn}/{bn_momentum}/{bn_mode})_ln{(ln)}_xq({crossq_style}/{tau})_utd({utd}/{policy_delay})_A{adam_b1}_Q({net_arch["qf"][0]})_l{lr}'
    elif algo == 'sac':
        # With the right hyperparameters, this here can run all the above algorithms
        # and ablations.
        layer_norm = ln
        if dropout:
            dropout_rate = 0.01
        group = f'SAC_{env}_bn({bn}/{bn_momentum}/{bn_mode})_ln{(ln)}_xq({crossq_style}/{tau})_utd({utd}/{policy_delay})_A{adam_b1}_Q({net_arch["qf"][0]})_l{lr}'
    elif algo == 'crossq':
        adam_b1 = 0.5
        policy_delay = 3
        n_critics = 2
        utd = 1  # nice
        net_arch["qf"] = [2048, 2048]  # wider critics
        bn = True  # use batch norm
        bn_momentum = 0.99
        crossq_style = True  # with a joint forward pass
        tau = 1.0  # without target networks
        group = f'CrossQ_{env}'
    else:
        raise NotImplemented

    model = SAC(
        "MultiInputPolicy" if isinstance(
            env.observation_space, gym.spaces.Dict
        ) else "MlpPolicy",
        env,
        policy_kwargs=dict({
            'activation_fn': activation_fn[critic_activation],
            'layer_norm': layer_norm,
            'batch_norm': bool(bn),
            'batch_norm_momentum': float(bn_momentum),
            'batch_norm_mode': bn_mode,
            'dropout_rate': dropout_rate,
            'n_critics': n_critics,
            'net_arch': net_arch,
            'optimizer_class': optax.adam,
            'optimizer_kwargs': dict({
                'b1': adam_b1,
                'b2': 0.999  # default
            })
        }),
        gradient_steps=utd,
        policy_delay=policy_delay,
        crossq_style=bool(crossq_style),
        td3_mode=td3_mode,
        use_bnstats_from_live_net=bool(bnstats_live_net),
        policy_q_reduce_fn=policy_q_reduce_fn,
        learning_starts=5000,
        learning_rate=lr,
        qf_learning_rate=lr,
        tau=tau,
        gamma=0.99,
        verbose=0,
        buffer_size=1_000_000,
        seed=seed,
        stats_window_size=1,  # don't smooth the episode return stats over time
        tensorboard_log=f"logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/",
    )

    # Create log dir where evaluation results will be saved
    eval_log_dir = f"./eval_logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/eval/"
    qbias_log_dir = f"./eval_logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/qbias/"
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(qbias_log_dir, exist_ok=True)

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    return model

