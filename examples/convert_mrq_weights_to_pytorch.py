from collections.abc import Callable

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_blox.blox.embedding.model_based_encoder import create_model_based_encoder_and_policy
from rl_blox.util.torch import transfer_parameters_flax_to_torch

# pytorch module definitions equivalent to flax modules created by rl-blox

class LayerNormMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        activation: str,
    ):
        super().__init__()

        self.n_outputs = n_outputs
        self.activation = getattr(F, activation)

        hidden_layers = []
        layer_norms = []
        n_in = n_features
        for n_out in hidden_nodes:
            hidden_layers.append(nn.Linear(n_in, n_out))
            layer_norms.append(nn.LayerNorm(normalized_shape=n_out))
            n_in = n_out
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.layer_norms = nn.ModuleList(layer_norms)

        self.output_layer = nn.Linear(n_in, n_outputs)

    def __call__(self, x):
        for layer, norm in zip(
            self.hidden_layers, self.layer_norms, strict=True
        ):
            x = self.activation(norm(layer(x)))
        return self.output_layer(x)


class DeterministicTanhPolicy(nn.Module):
    def __init__(self, policy_net: nn.Module, action_space: gym.spaces.Box):
        super().__init__()

        self.policy_net = policy_net

        self.register_buffer(
            "action_scale",
            torch.Tensor((action_space.high - action_space.low) / 2.0),
        )
        self.register_buffer(
            "action_bias",
            torch.Tensor((action_space.high + action_space.low) / 2.0),
        )

    def __call__(self, observation):
        y = self.policy_net(observation)
        return self.scale_output(y)

    def scale_output(self, y):
        return (
            F.tanh(y) * self.action_scale[None, :] + self.action_bias[None, :]
        )


class ModelBasedEncoder(nn.Module):
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
    ):
        super().__init__()

        self.zs = LayerNormMLP(
            n_state_features,
            zs_dim,
            hidden_nodes,
            activation,
        )
        self.za = nn.Linear(n_action_features, za_dim)
        self.zsa = LayerNormMLP(
            zs_dim + za_dim,
            zsa_dim,
            hidden_nodes,
            activation,
        )
        self.model = nn.Linear(zsa_dim, n_bins + zs_dim + 1)
        self.zs_dim = zs_dim
        self.activation = getattr(F, activation)
        self.zs_layer_norm = nn.LayerNorm(normalized_shape=zs_dim)

    def encode_zsa(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        za = self.activation(self.za(action))
        return self.zsa(jnp.concatenate((zs, za), axis=-1))

    def encode_zs(self, observation: jnp.ndarray) -> jnp.ndarray:
        return self.activation(self.zs_layer_norm(self.zs(observation)))

    def model_head(
        self, zs: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        zsa = self.encode_zsa(zs, action)
        dzr = self.model(zsa)
        done = dzr[:, 0]
        next_zs = dzr[:, 1 : 1 + self.zs_dim]
        reward = dzr[:, 1 + self.zs_dim :]
        return done, next_zs, reward


class DeterministicPolicyWithEncoder(nn.Module):
    def __init__(
        self, encoder: ModelBasedEncoder, policy: DeterministicTanhPolicy
    ):
        super().__init__()

        self.encoder = encoder
        self.policy = policy

    def __call__(self, observation):
        return self.policy(self.encoder.encode_zs(observation))


env_name = "Hopper-v5"
env = gym.make(env_name)

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_state_features = env.observation_space.shape[0]
n_action_features = env.action_space.shape[0]
action_space: gym.spaces.Box = env.action_space
policy_hidden_nodes = [512, 512]
policy_activation = "relu"
encoder_n_bins = 65
encoder_zs_dim = 512
encoder_za_dim = 256
encoder_zsa_dim = 512
encoder_hidden_nodes = [512, 512]
encoder_activation = "elu"

policy_with_encoder_flax = create_model_based_encoder_and_policy(
    n_state_features=n_state_features,
    n_action_features=n_action_features,
    action_space=action_space,
    encoder_n_bins=encoder_n_bins,
    encoder_zs_dim=encoder_zs_dim,
    encoder_za_dim=encoder_za_dim,
    encoder_zsa_dim=encoder_zsa_dim,
    encoder_hidden_nodes=encoder_hidden_nodes,
    encoder_activation=encoder_activation,
)

encoder = ModelBasedEncoder(
    n_state_features=n_state_features,
    n_action_features=n_action_features,
    n_bins=encoder_n_bins,
    zs_dim=encoder_zs_dim,
    za_dim=encoder_za_dim,
    zsa_dim=encoder_zsa_dim,
    hidden_nodes=encoder_hidden_nodes,
    activation=encoder_activation,
)
policy_net = LayerNormMLP(
    encoder_zs_dim,
    n_action_features,
    policy_hidden_nodes,
    policy_activation,
)
policy = DeterministicTanhPolicy(policy_net, action_space)
policy_with_encoder = DeterministicPolicyWithEncoder(encoder, policy).to(torch_device)

# Transfer weights from flax modules to pytorch modules
transfer_parameters_flax_to_torch(policy_with_encoder_flax, policy_with_encoder)

rng = np.random.default_rng(42)
for i in range(10):
    observation = rng.normal(size=(1, env.observation_space.shape[0])).astype(
        dtype=np.float32
    )
    observation_jax = jnp.array(observation)
    action_flax = policy_with_encoder_flax(observation_jax)
    observation_torch = torch.Tensor(observation).to(torch_device)
    action_torch = policy_with_encoder(observation_torch)

    print(f"Test {i}:")
    print(f"Flax output: {np.asarray(action_flax)}")
    print(f"Torch output: {action_torch.cpu().detach().numpy()}")
    print()
