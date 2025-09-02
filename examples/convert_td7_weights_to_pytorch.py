from collections import OrderedDict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flax import nnx

from rl_blox.algorithm.td7 import create_td7_state


def translate_to_torch(flax_module: nnx.Module, torch_module: nn.Module):
    torch_state = torch_module.state_dict()  # flat state dict
    flax_state = nnx.state(flax_module)  # nested state dict

    output_state = OrderedDict()

    # Converted nested flax state dict to flat torch state dict.
    for torch_param_key in torch_state.keys():
        attributes = torch_param_key.split(".")
        obj = flax_state
        for attr in attributes:
            try:
                attr = int(attr)  # for list indices
            except ValueError:
                pass  # not int
            if attr == "weight":
                attr = "kernel"
            obj = obj[attr]

        # flax kernels are transposed in comparison to torch weights
        params = obj.value.T

        # We cannot directly transfer jax arrays to torch tensors, so we
        # convert to numpy first. In addition, we make a copy, because torch
        # complains about the parameters not being writeable otherwise.
        params = np.copy(params)

        output_state[torch_param_key] = torch.Tensor(params)

    torch_module.load_state_dict(output_state)


# pytorch module definitions equivalent to flax modules created by rl-blox


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class MLP(nn.Module):
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
        n_in = n_features
        for n_out in hidden_nodes:
            hidden_layers.append(nn.Linear(n_in, n_out))
            n_in = n_out
        self.hidden_layers = nn.ModuleList(hidden_layers)

        self.output_layer = nn.Linear(n_in, n_outputs)

    def __call__(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


class DeterministicTanhPolicy(nn.Module):
    def __init__(self, policy_net: nn.Module, action_space: gym.spaces.Box):
        super().__init__()

        self.policy_net = policy_net
        self.action_scale = torch.Tensor(
            (action_space.high - action_space.low) / 2.0
        )
        self.action_bias = torch.Tensor(
            (action_space.high + action_space.low) / 2.0
        )

    def __call__(self, observation):
        y = self.policy_net(observation)
        return self.scale_output(y)

    def scale_output(self, y):
        return (
            F.tanh(y) * self.action_scale[None, :] + self.action_bias[None, :]
        )


class SALE(nn.Module):
    def __init__(
        self, state_embedding: nn.Module, state_action_embedding: nn.Module
    ):
        super().__init__()

        self.state_embedding = state_embedding
        self.state_action_embedding = state_action_embedding

    def __call__(self, state, action):
        zs = self.state_embedding(state)
        zs_action = torch.cat([zs, action], 1)
        zsa = self.state_action_embedding(zs_action)
        return zsa, zs

    def state_embedding(self, state):
        return avg_l1_norm(self._state_embedding(state))


class ActorSALE(nn.Module):
    def __init__(
        self,
        policy_net: nn.Module,
        n_state_features: int,
        hidden_nodes: int,
    ):
        super().__init__()

        self.policy_net = policy_net
        self.l0 = nn.Linear(n_state_features, hidden_nodes)

    def __call__(self, state, zs):
        h = avg_l1_norm(self.l0(state))
        he = torch.cat([h, zs], 1)
        return self.policy_net(he)


env_name = "Hopper-v5"
env = gym.make(env_name)

# Hyperparameters for encoder and actor
n_embedding_dimensions = 256
policy_sa_encoding_nodes = 256
policy_activation = "relu"
policy_hidden_nodes = [256, 256]
state_embedding_hidden_nodes = [256]
state_action_embedding_hidden_nodes = [256]
embedding_activation = "elu"

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_embedding = MLP(
    env.observation_space.shape[0],
    n_embedding_dimensions,
    state_embedding_hidden_nodes,
    embedding_activation,
)
state_action_embedding = MLP(
    n_embedding_dimensions + env.action_space.shape[0],
    n_embedding_dimensions,
    state_action_embedding_hidden_nodes,
    embedding_activation,
)
embedding = SALE(state_embedding, state_action_embedding)

policy_net = MLP(
    policy_sa_encoding_nodes + n_embedding_dimensions,
    env.action_space.shape[0],
    policy_hidden_nodes,
    policy_activation,
)
policy = DeterministicTanhPolicy(policy_net, env.action_space)
actor = ActorSALE(
    policy,
    env.observation_space.shape[0],
    policy_sa_encoding_nodes,
).to(torch_device)

state = create_td7_state(
    env,
    n_embedding_dimensions=n_embedding_dimensions,
    policy_sa_encoding_nodes=policy_sa_encoding_nodes,
    policy_activation=policy_activation,
    policy_hidden_nodes=policy_hidden_nodes,
    embedding_activation=embedding_activation,
    state_embedding_hidden_nodes=state_embedding_hidden_nodes,
    state_action_embedding_hidden_nodes=state_action_embedding_hidden_nodes,
    seed=1,
)

translate_to_torch(state.embedding, embedding)
translate_to_torch(state.actor, actor)
