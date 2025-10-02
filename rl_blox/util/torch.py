import contextlib
from collections import OrderedDict

from flax import nnx
import numpy as np
import torch
import torch.nn as nn


def transfer_parameters_flax_to_torch(
    flax_module: nnx.Module, torch_module: nn.Module
):
    torch_state = torch_module.state_dict()  # flat state dict
    flax_state = nnx.state(flax_module)  # nested state dict

    output_state = OrderedDict()

    # Converted nested flax state dict to flat torch state dict.
    for torch_param_key in torch_state:
        attributes = torch_param_key.split(".")
        obj = flax_state
        for attr in attributes:
            with contextlib.suppress(ValueError):
                attr = int(attr)  # for list indices
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
