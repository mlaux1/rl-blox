from collections import OrderedDict, namedtuple
from collections.abc import Callable
from functools import partial

import chex
import gymnasium as gym
import jax.numpy as jnp
import jax.random
import numpy as np
import numpy.typing as npt
import optax
import tqdm
from flax import nnx

from ..blox.double_qnet import ContinuousClippedDoubleQNet
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy
from ..blox.preprocessing import make_two_hot_bins, two_hot_cross_entropy_loss
from ..blox.target_net import hard_target_net_update
from ..logging.logger import LoggerBase
from .ddpg import make_sample_actions
from .td3 import make_sample_target_actions


class EpisodicReplayBuffer:
    """Episodic replay buffer for the MR.Q algorithm.

    Parameters
    ----------
    buffer_size : int
        Maximum size of the buffer.

    horizon : int, optional
        Maximum length of the horizon.

    keys : list[str], optional
        Names of the quantities that should be stored in the replay buffer.
        Defaults to ['observation', 'action', 'reward', 'next_observation',
        'terminated', 'truncated']. These names have to be used as key word
        arguments when adding a sample. When sampling a batch, the arrays will
        be returned in this order. Must contain at least 'observation',
        'next_observation', 'terminated', and 'truncated'.

    dtypes : list[dtype], optional
        dtype used for each buffer. Defaults to float for everything except
        'terminated' and 'truncated'.

    discrete_actions : bool, optional
        Changes the default dtype for actions to int.
    """

    def __init__(
        self,
        buffer_size: int,
        horizon: int = 1,
        keys: list[str] | None = None,
        dtypes: list[npt.DTypeLike] | None = None,
        discrete_actions: bool = False,
    ):
        chex.assert_scalar_positive(buffer_size)
        chex.assert_scalar_positive(horizon)

        if keys is None:
            keys = [
                "observation",
                "action",
                "reward",
                "next_observation",
                "terminated",
                "truncated",
            ]
        if dtypes is None:
            dtypes = [
                float,
                int if discrete_actions else float,
                float,
                float,
                int,
                int,
            ]
        for key in [
            "observation",
            "next_observation",
            "terminated",
            "truncated",
        ]:
            if key not in keys:
                raise ValueError(f"'{key}' must be in keys")

        self.buffer = OrderedDict()
        for k, t in zip(keys, dtypes, strict=True):
            self.buffer[k] = np.empty(0, dtype=t)
        self.Batch = namedtuple("Batch", self.buffer)
        self.buffer_size = buffer_size
        self.current_len = 0
        self.insert_idx = 0

        self.episode_timesteps = 0
        # track if there are any terminal transitions in the buffer
        self.environment_terminates = True
        self.horizon = horizon
        self.mask = np.zeros(self.buffer_size, dtype=int)

        # TODO prioritized experience replay

    def add_sample(self, **sample):
        """Add transition sample to the replay buffer.

        Note that the individual arguments have to be passed as keyword
        arguments with keys matching the ones passed to the constructor or
        the default keys respectively.
        """
        if self.current_len == 0:
            for k, v in sample.items():
                assert k in self.buffer, f"{k} not in {self.buffer.keys()}"
                self.buffer[k] = np.empty(
                    (self.buffer_size,) + np.asarray(v).shape,
                    dtype=self.buffer[k].dtype,
                )
        for k, v in sample.items():
            self.buffer[k][self.insert_idx] = v

        self.current_len = min(self.current_len + 1, self.buffer_size)
        self.episode_timesteps += 1
        if sample["terminated"]:
            self.environment_terminates = True

        self.mask[self.insert_idx] = 0
        if self.episode_timesteps > self.horizon:
            self.mask[(self.insert_idx - self.horizon) % self.buffer_size] = 1

        self.insert_idx = (self.insert_idx + 1) % self.buffer_size

        if sample["terminated"] or sample["truncated"]:
            # TODO what about action, next observation, and reward?
            self.buffer["observation"][self.insert_idx] = sample[
                "next_observation"
            ]

            self.mask[self.insert_idx % self.buffer_size] = 0
            past_idx = (
                self.insert_idx
                - np.arange(min(self.episode_timesteps, self.horizon))
                - 1
            ) % self.buffer_size
            self.mask[past_idx] = (
                0 if sample["truncated"] else 1
            )  # mask out truncated subtrajectories

            self.insert_idx = (self.insert_idx + 1) % self.buffer_size
            self.current_len = min(self.current_len + 1, self.buffer_size)

            self.episode_timesteps = 0

    def sample_batch(
        self,
        batch_size: int,
        horizon: int,
        include_intermediate: bool,
        rng: np.random.Generator,
    ) -> tuple[jnp.ndarray]:
        """Sample a batch of transitions from the replay buffer.

        Parameters
        ----------
        batch_size : int
            Number of samples to be returned.

        horizon : int
            Horizon for the sampled transitions.

        include_intermediate : bool
            Whether to include intermediate states in the sampled transitions.

        rng : np.random.Generator
            Random number generator for sampling.

        Returns
        -------
        batch : tuple[jnp.ndarray]
            A tuple containing the sampled observations, actions, rewards,
            next observations, and terminations.
        """
        chex.assert_scalar_positive(batch_size)
        chex.assert_scalar_positive(horizon)

        indices = self._sample_idx(batch_size, rng)
        # TODO % self.current_len or % self.buffer_size? - maybe self.buffer_size is possible because of the mask?
        indices = (
            indices[:, np.newaxis] + np.arange(self.horizon)[np.newaxis]
        ) % self.current_len

        if include_intermediate:
            # sample subtrajectories (with horizon dimension) for unrolling
            # dynamics
            chex.assert_shape(indices, (batch_size, horizon))

            batch = self.Batch(
                **{k: jnp.asarray(self.buffer[k][indices]) for k in self.buffer}
            )
        else:
            # sample at specific horizon (used for multistep rewards)
            indices = indices[:, 0]
            chex.assert_shape(indices, (batch_size,))

            batch = {}
            for k in self.buffer:
                if k in ["observation", "action"]:
                    indices_without_intermediate = indices[:, 0]
                elif k == "next_observation":
                    indices_without_intermediate = indices[:, -1]
                else:
                    indices_without_intermediate = indices
                batch[k] = jnp.asarray(
                    self.buffer[k][indices_without_intermediate]
                )
            batch = self.Batch(**batch)

        return batch

    def _sample_idx(
        self, batch_size: int, rng: np.random.Generator
    ) -> npt.NDArray[int]:
        # TODO prioritized experience replay
        nz = np.nonzero(self.mask)[0]
        indices = rng.integers(0, len(nz), size=batch_size)
        self.sampled_indices = nz[indices]
        return self.sampled_indices

    # TODO save and load with pickle


mrq_kernel_init = jax.nn.initializers.variance_scaling(
    scale=2, mode="fan_avg", distribution="uniform"
)


class LayerNormMLP(nnx.Module):
    """Multilayer Perceptron.

    Parameters
    ----------
    n_features : int
        Number of features.

    n_outputs : int
        Number of output components.

    hidden_nodes : list
        Numbers of hidden nodes of the MLP.

    activation : str
        Activation function. Has to be the name of a function defined in the
        flax.nnx module.

    rngs : nnx.Rngs
        Random number generator.
    """

    n_outputs: int
    """Number of output components."""

    activation: Callable[[jnp.ndarray], jnp.ndarray]
    """Activation function."""

    hidden_layers: list[nnx.Linear]
    """Hidden layers."""

    output_layer: nnx.Linear
    """Output layer."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        activation: str,
        rngs: nnx.Rngs,
    ):
        chex.assert_scalar_positive(n_features)
        chex.assert_scalar_positive(n_outputs)

        self.n_outputs = n_outputs
        self.activation = getattr(nnx, activation)

        self.hidden_layers = []
        self.layer_norms = []
        n_in = n_features
        for n_out in hidden_nodes:
            self.hidden_layers.append(
                nnx.Linear(n_in, n_out, rngs=rngs, kernel_init=mrq_kernel_init)
            )
            self.layer_norms.append(
                nnx.LayerNorm(num_features=n_out, rngs=rngs)
            )
            n_in = n_out

        self.output_layer = nnx.Linear(
            n_in,
            n_outputs,
            rngs=rngs,
            kernel_init=mrq_kernel_init,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer, norm in zip(
            self.hidden_layers, self.layer_norms, strict=True
        ):
            x = self.activation(norm(layer(x)))
        return self.output_layer(x)


class Encoder(nnx.Module):
    """Encoder for the MR.Q algorithm.

    Parameters
    ----------
    n_bins : int
        Number of bins for the two-hot encoding.

    n_state_features : int
        Number of state components.

    n_action_features : int
        Number of action components.

    zs_dim : int
        Dimension of the latent state representation.

    za_dim : int
        Dimension of the latent action representation.

    zsa_dim : int
        Dimension of the latent state-action representation.

    hidden_nodes : list
        Numbers of hidden nodes of the MLP.

    activation : str
        Activation function. Has to be the name of a function defined in the
        flax.nnx module.

    rngs : nnx.Rngs
        Random number generator.
    """

    zs: LayerNormMLP
    """Maps observations to latent state representations (nonlinear)."""

    za: nnx.Linear
    """Maps actions to latent action representations (linear)."""

    zsa: LayerNormMLP
    """Maps zs and za to latent state-action representations (nonlinear)."""

    model: nnx.Linear
    """Maps zsa to done flag, next latent state (zs), and reward (linear)."""

    zs_dim: int
    """Dimension of the latent state representation."""

    activation: Callable[[jnp.ndarray], jnp.ndarray]
    """Activation function."""

    zs_layer_norm: nnx.LayerNorm
    """Layer normalization for the latent state representation."""

    def __init__(
        self,
        n_state_features: int,
        n_action_features: int,
        n_bins: int,
        zs_dim: int,
        za_dim: int,
        zsa_dim: int,
        hidden_nodes: list[int],
        activation: str,
        rngs: nnx.Rngs,
    ):
        self.zs = LayerNormMLP(
            n_state_features,
            zs_dim,
            hidden_nodes,
            activation,
            rngs=rngs,
        )
        self.za = nnx.Linear(
            n_action_features, za_dim, rngs=rngs, kernel_init=mrq_kernel_init
        )
        self.zsa = LayerNormMLP(
            zs_dim + za_dim,
            zsa_dim,
            hidden_nodes,
            activation,
            rngs=rngs,
        )
        self.model = nnx.Linear(
            zsa_dim, n_bins + zs_dim + 1, rngs=rngs, kernel_init=mrq_kernel_init
        )
        self.zs_dim = zs_dim
        self.activation = getattr(nnx, activation)
        self.zs_layer_norm = nnx.LayerNorm(num_features=zs_dim, rngs=rngs)

    def encode_zsa(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Encodes the state and action into latent representation.

        Parameters
        ----------
        zs : array, shape (n_samples, n_state_features)
            State representation.

        action : array, shape (n_samples, n_action_features)
            Action representation.

        Returns
        -------
        zsa : array, shape (n_samples, zsa_dim)
            Latent state-action representation.
        """
        za = self.activation(self.za(action))
        return self.zsa(jnp.concatenate((zs, za), axis=-1))

    def encode_zs(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Encodes the observation into a latent state representation.

        Parameters
        ----------
        observation : array, shape (n_samples, n_state_features)
            Observation representation.

        Returns
        -------
        zs : array, shape (n_samples, zs_dim)
            Latent state representation.
        """
        return self.activation(self.zs_layer_norm(self.zs(observation)))

    def model_head(
        self, zs: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Predicts the full model.

        Parameters
        ----------
        zs : array, shape (n_samples, zs_dim)
            Latent state representation.

        action : array, shape (n_samples, n_action_features)
            Action.

        Returns
        -------
        done : array, shape (n_samples,)
            Flag indicating whether the episode is done.
        next_zs : array, shape (n_samples, zs_dim)
            Predicted next state representation.
        reward : array, shape (n_samples, n_bins)
            Two-hot encoded reward.
        """
        zsa = self.encode_zsa(zs, action)
        dzr = self.model(zsa)
        done = dzr[:, 0]
        next_zs = dzr[:, 1 : 1 + self.zs_dim]
        reward = dzr[:, 1 + self.zs_dim :]
        return done, next_zs, reward


class DeterministicPolicyWithEncoder(nnx.Module):
    """Combines encoder and deterministic policy."""

    def __init__(self, encoder: Encoder, policy: DeterministicTanhPolicy):
        self.encoder = encoder
        self.policy = policy

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        return self.policy(self.encoder.encode_zs(observation))


def masked_mse_loss(
    predictions: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """Masked mean squared error loss.

    Parameters
    ----------
    predictions : jnp.ndarray
        Predicted values.

    targets : jnp.ndarray
        Target values.

    mask : jnp.ndarray
        Mask indicating which values to include in the loss calculation.

    Returns
    -------
    loss : jnp.ndarray
        Masked mean squared error loss.
    """
    return jnp.mean(
        optax.squared_error(predictions=predictions, targets=targets) * mask
    )


@partial(
    nnx.jit,
    static_argnames=(
        "encoder_horizon",
        "dynamics_weight",
        "reward_weight",
        "done_weight",
    ),
)
def update_encoder(
    encoder: Encoder,
    encoder_target: Encoder,
    encoder_optimizer: nnx.Optimizer,
    the_bins: jnp.ndarray,
    batch: tuple[jnp.ndarray],
    encoder_horizon: int,
    dynamics_weight: float,
    reward_weight: float,
    done_weight: float,
):
    (loss, losses), grads = nnx.value_and_grad(
        encoder_loss, argnums=0, has_aux=True
    )(
        encoder,
        encoder_target,
        the_bins,
        batch,
        encoder_horizon,
        dynamics_weight,
        reward_weight,
        done_weight,
    )
    encoder_optimizer.update(grads)
    return loss, losses


def encoder_loss(
    encoder: Encoder,
    encoder_target: Encoder,
    the_bins: jnp.ndarray,
    batch: tuple[jnp.ndarray],
    encoder_horizon: int,
    dynamics_weight: float,
    reward_weight: float,
    done_weight: float,
) -> tuple[float, tuple[float, float, float]]:
    flat_next_observation = batch.next_observation.reshape(
        -1, *batch.next_observation.shape[2:]
    )
    flat_next_zs = jax.lax.stop_gradient(
        encoder_target.zs(flat_next_observation)
    )
    next_zs = flat_next_zs.reshape(
        list(batch.next_observation.shape[:2]) + [-1]
    )
    pred_zs_t = encoder.encode_zs(batch.observation[:, 0])
    not_done = 1 - batch.terminated
    # in subtrajectories with termination mask, mask out losses
    # after termination
    prev_not_done = 1

    total_dynamics_loss = 0.0
    total_reward_loss = 0.0
    total_done_loss = 0.0

    for t in range(encoder_horizon):
        pred_done_t, pred_zs_t, pred_reward_t = encoder.model_head(
            pred_zs_t, batch.action[:, t]
        )

        target_zs_t = next_zs[:, t]
        target_reward_t = batch.reward[:, t]
        target_done_t = batch.terminated[:, t]
        total_dynamics_loss += masked_mse_loss(
            pred_zs_t, target_zs_t, prev_not_done
        )
        total_reward_loss += jnp.mean(
            two_hot_cross_entropy_loss(the_bins, pred_reward_t, target_reward_t)
        )
        total_done_loss += masked_mse_loss(
            pred_done_t, target_done_t, prev_not_done
        )

        # Update termination mask
        prev_not_done = not_done[:, t].reshape(-1, 1) * prev_not_done

    loss = (
        dynamics_weight * total_dynamics_loss
        + reward_weight * total_reward_loss
        + done_weight * total_done_loss
    )
    return loss, (total_dynamics_loss, total_reward_loss, total_done_loss)


def create_mrq_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_activation: str = "relu",
    policy_learning_rate: float = 3e-4,
    policy_weight_decay: float = 1e-4,
    q_hidden_nodes: list[int] | tuple[int] = (512, 512, 512),
    q_activation: str = "elu",
    q_learning_rate: float = 3e-4,
    q_weight_decay: float = 1e-4,
    encoder_n_bins: int = 65,
    encoder_zs_dim: int = 512,
    encoder_za_dim: int = 256,
    encoder_zsa_dim: int = 512,
    encoder_hidden_nodes: list[int] | tuple[int] = (512, 512),
    encoder_activation: str = "elu",
    encoder_learning_rate: float = 1e-4,
    encoder_weight_decay: float = 1e-4,
    seed: int = 0,
):
    env.action_space.seed(seed)

    rngs = nnx.Rngs(seed)
    encoder = Encoder(
        n_state_features=env.observation_space.shape[0],
        n_action_features=env.action_space.shape[0],
        n_bins=encoder_n_bins,
        zs_dim=encoder_zs_dim,
        za_dim=encoder_za_dim,
        zsa_dim=encoder_zsa_dim,
        hidden_nodes=encoder_hidden_nodes,
        activation=encoder_activation,
        rngs=rngs,
    )
    encoder_optimizer = nnx.Optimizer(
        encoder,
        optax.adamw(
            learning_rate=encoder_learning_rate,
            weight_decay=encoder_weight_decay,
        ),
    )

    q1 = LayerNormMLP(
        encoder_zsa_dim,
        1,
        q_hidden_nodes,
        q_activation,
        rngs=rngs,
    )
    q2 = LayerNormMLP(
        encoder_zsa_dim,
        1,
        q_hidden_nodes,
        q_activation,
        rngs=rngs,
    )
    q = ContinuousClippedDoubleQNet(q1, q2)
    q_optimizer = nnx.Optimizer(
        q,
        optax.adamw(
            learning_rate=q_learning_rate,
            weight_decay=q_weight_decay,
        ),
    )

    policy_net = LayerNormMLP(
        encoder_zs_dim,
        env.action_space.shape[0],
        policy_hidden_nodes,
        policy_activation,
        rngs=rngs,
    )
    policy = DeterministicTanhPolicy(policy_net, env.action_space)
    policy_optimizer = nnx.Optimizer(
        policy,
        optax.adamw(
            learning_rate=policy_learning_rate,
            weight_decay=policy_weight_decay,
        ),
    )

    the_bins = make_two_hot_bins(n_bin_edges=encoder_n_bins)

    return namedtuple(
        "MRQState",
        [
            "encoder",
            "encoder_optimizer",
            "q",
            "q_optimizer",
            "policy",
            "policy_optimizer",
            "the_bins",
        ],
    )(
        encoder,
        encoder_optimizer,
        q,
        q_optimizer,
        policy,
        policy_optimizer,
        the_bins,
    )


def train_mrq(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    encoder: Encoder,
    encoder_optimizer: nnx.Optimizer,
    policy: DeterministicTanhPolicy,
    policy_optimizer: nnx.Optimizer,
    q: ContinuousClippedDoubleQNet,
    q_optimizer: nnx.Optimizer,
    the_bins: jnp.ndarray,
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    buffer_size: int = 1_000_000,
    gamma: float = 0.99,
    target_delay: int = 250,
    batch_size: int = 256,
    exploration_noise: float = 0.1,
    target_policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    learning_starts: int = 10_000,
    encoder_horizon: int = 5,
    q_horizon: int = 3,
    dynamics_weight: float = 1.0,
    reward_weight: float = 0.1,
    done_weight: float = 0.1,
    logger: LoggerBase | None = None,
) -> tuple[
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    EpisodicReplayBuffer,
]:
    r"""Model-based Representation for Q-learning (MR.Q).

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium environment.

    encoder : Encoder
        Encoder for the MR.Q algorithm.

    encoder_optimizer : nnx.Optimizer
        Optimizer for the encoder.

    policy : DeterministicTanhPolicy
        Policy for the MR.Q algorithm. Maps the latent state representation
        to actions in the environment.

    policy_optimizer : nnx.Optimizer
        Optimizer for the policy.

    q : ContinuousClippedDoubleQNet
        Action-value function approximator for the MR.Q algorithm. Maps the
        latent state-action representation to the expected value of the
        state-action pair.

    q_optimizer : nnx.Optimizer
        Optimizer for the action-value function approximator.

    the_bins : jnp.ndarray
        Bin edges for the two-hot encoding of the reward predicted by the model.

    seed : int, optional
        Seed for random number generators in Jax and NumPy.

    total_timesteps : int, optional
        Number of steps to execute in the environment.

    buffer_size : int, optional
        Size of the replay buffer.

    gamma : float, optional
        Discount factor.

    target_delay : int, optional
        Delayed target net updates. The target nets are updated every
        ``target_delay`` steps.

    batch_size : int, optional
        Size of a batch during gradient computation.

    exploration_noise : float, optional
        Exploration noise in action space. Will be scaled by half of the range
        of the action space.

    target_policy_noise : float, optional
        Exploration noise in action space for target policy smoothing.

    noise_clip : float, optional
        Maximum absolute value of the exploration noise for sampling target
        actions for the critic update. Will be scaled by half of the range
        of the action space.

    learning_starts : int, optional
        Learning starts after this number of random steps was taken in the
        environment.

    encoder_horizon : int, optional
        Horizon for encoder training.

    q_horizon : int, optional
        Horizon for Q training.

    dynamics_weight : float, optional
        Weight for the dynamics loss in the encoder training.

    reward_weight : float, optional
        Weight for the reward loss in the encoder training.

    done_weight : float, optional
        Weight for the done loss in the encoder training.

    logger : LoggerBase, optional
        Experiment logger.

    Returns
    -------
    encoder : Encoder
        Encoder for the MR.Q algorithm.

    encoder_target : Encoder
        Target encoder for the MR.Q algorithm.

    encoder_optimizer : nnx.Optimizer
        Optimizer for the encoder.

    policy : DeterministicTanhPolicy
        Policy for the MR.Q algorithm.

    policy_target : DeterministicTanhPolicy
        Target policy for the MR.Q algorithm.

    policy_optimizer : nnx.Optimizer
        Optimizer for the policy.

    q : ContinuousClippedDoubleQNet
        Action-value function approximator for the MR.Q algorithm.

    q_target : ContinuousClippedDoubleQNet
        Target action-value function approximator for the MR.Q algorithm.

    q_optimizer : nnx.Optimizer
        Optimizer for the action-value function approximator.

    replay_buffer : EpisodicReplayBuffer
        Episodic replay buffer for the MR.Q algorithm.

    References
    ----------
    .. [1] Fujimoto, S., D'Oro, P., Zhang, A., Tian, Y., Rabbat, M. (2025).
       Towards General-Purpose Model-Free Reinforcement Learning. In
       International Conference on Learning Representations (ICLR).
       https://openreview.net/forum?id=R1hIXdST22
    """
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    replay_buffer = EpisodicReplayBuffer(
        buffer_size,
        horizon=max(encoder_horizon, q_horizon),
    )

    _sample_actions = make_sample_actions(env.action_space, exploration_noise)
    _sample_target_actions = make_sample_target_actions(
        env.action_space, target_policy_noise, noise_clip
    )

    epoch = 0

    encoder_target = nnx.clone(encoder)
    policy_target = nnx.clone(policy)
    q_target = nnx.clone(q)

    policy_with_encoder = DeterministicPolicyWithEncoder(encoder, policy)

    # TODO track reward scale and target reward scale

    if logger is not None:
        logger.start_new_episode()
    obs, _ = env.reset(seed=seed)
    steps_per_episode = 0
    accumulated_reward = 0.0

    for global_step in tqdm.trange(total_timesteps):
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            key, action_key = jax.random.split(key, 2)
            action = np.asarray(
                _sample_actions(
                    policy_with_encoder, jnp.asarray(obs), action_key
                )
            )

        next_obs, reward, terminated, truncated, info = env.step(action)
        steps_per_episode += 1
        accumulated_reward += reward

        replay_buffer.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            terminated=terminated,
            truncated=truncated,
        )

        if global_step >= learning_starts:
            epoch += 1
            if epoch % target_delay == 0:
                hard_target_net_update(policy, policy_target)
                hard_target_net_update(q, q_target)
                hard_target_net_update(encoder, encoder_target)

                for delayed_train_step_idx in range(1, target_delay + 1):
                    batch = replay_buffer.sample_batch(
                        batch_size, encoder_horizon, True, rng
                    )

                    encoder_loss, (dynamics_loss, reward_loss, done_loss) = (
                        update_encoder(
                            encoder,
                            encoder_target,
                            encoder_optimizer,
                            the_bins,
                            batch,
                            encoder_horizon,
                            dynamics_weight,
                            reward_weight,
                            done_weight,
                        )
                    )
                    if logger is not None:
                        stats = {
                            "encoder_loss": encoder_loss,
                            "dynamics_loss": dynamics_loss,
                            "reward_loss": reward_loss,
                            "done_loss": done_loss,
                        }
                        log_step = (
                            global_step
                            + 1
                            - target_delay
                            + delayed_train_step_idx
                        )
                        for k, v in stats.items():
                            logger.record_stat(k, v, step=log_step)

            # TODO update policy and q networks
            # replay_buffer.sample_batch(batch_size, q_horizon, False, rng)

        if terminated or truncated:
            if logger is not None:
                logger.record_stat(
                    "return", accumulated_reward, step=global_step + 1
                )
                logger.stop_episode(steps_per_episode)
                logger.start_new_episode()

            obs, _ = env.reset()

            steps_per_episode = 0
            accumulated_reward = 0.0
        else:
            obs = next_obs

    return namedtuple(
        "MRQResult",
        [
            "encoder",
            "encoder_target",
            "encoder_optimizer",
            "policy",
            "policy_target",
            "policy_optimizer",
            "q",
            "q_target",
            "q_optimizer",
            "replay_buffer",
        ],
    )(
        encoder,
        encoder_target,
        encoder_optimizer,
        policy,
        policy_target,
        policy_optimizer,
        q,
        q_target,
        q_optimizer,
        replay_buffer,
    )
