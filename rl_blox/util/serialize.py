import pickle

import jax
from flax import nnx


def save_pickle(
    filename: str, net: nnx.Module, move_to_device: None | str = None
):
    graphdef, state = nnx.split(net)

    if move_to_device is not None:
        state = _put_on_device(state, move_to_device)

    with open(filename, "wb") as f:
        pickle.dump(state, f)


def load_pickle(
    filename: str, graphdef: nnx.GraphDef, move_to_device: None | str = None
):
    with open(filename, "rb") as f:
        if move_to_device is not None:
            if move_to_device not in ["cpu", "cuda"]:
                raise ValueError(f"Unknown device: '{move_to_device}'")
            with jax.default_device(jax.devices(move_to_device)[0]):
                state = pickle.load(f)
                net = nnx.merge(graphdef, state)
        else:
            state = pickle.load(f)
            net = nnx.merge(graphdef, state)

    return net


def _put_on_device(state, move_to_device: str):
    if move_to_device not in ["cpu", "cuda"]:
        raise ValueError(f"Unknown device: '{move_to_device}'")
    device = jax.devices(move_to_device)[0]
    return jax.device_put(state, device)
