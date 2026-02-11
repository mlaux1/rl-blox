import contextlib
from collections import OrderedDict

from flax import nnx
import numpy as np
import torch
import torch.nn as nn


FLAX_WEIGHT_NAMES = [
    "kernel",  # for linear layers
    "scale",  # for layer norm
]


def transfer_parameters_flax_to_torch(
    flax_module: nnx.Module, torch_module: nn.Module, verbose: int = 0
):
    torch_state = torch_module.state_dict()  # flat state dict
    flax_state = nnx.state(flax_module)  # nested state dict

    output_state = OrderedDict()

    # Converted nested flax state dict to flat torch state dict.
    for torch_param_key in torch_state:
        if verbose:
            print(f"Extracting torch parameters '{torch_param_key}' to ", end="")

        attributes = torch_param_key.split(".")
        obj = flax_state
        for attr in attributes:
            with contextlib.suppress(ValueError):
                attr = int(attr)  # for list indices

            if attr == "weight":
                for name in FLAX_WEIGHT_NAMES:
                    if name in obj:
                        attr = name
                        break
                if attr == "weight":
                    raise NotImplementedError(
                        f"No idea how to translate attribute 'weight' to flax "
                        f"state: {obj}"
                    )

            if verbose:
                print(f".{attr}", end="")

            obj = obj[attr]

        # flax kernels are transposed in comparison to torch weights
        params = obj.value.T

        # We cannot directly transfer jax arrays to torch tensors, so we
        # convert to numpy first. In addition, we make a copy, because torch
        # complains about the parameters not being writeable otherwise.
        params = np.copy(params)

        output_state[torch_param_key] = torch.Tensor(params)

        if verbose:
            print(" DONE.")

    torch_module.load_state_dict(output_state)
