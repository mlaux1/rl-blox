import os
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx

from rl_blox.algorithm.td7 import create_td7_state
from rl_blox.util.serialize import load_pickle, save_pickle


def test_pickle():
    env_name = "Hopper-v5"
    env = gym.make(env_name)

    td7_state = create_td7_state(env, seed=1)

    filename = "embedding.pkl"

    try:
        save_pickle(filename, td7_state.embedding, move_to_device="cpu")
        graphdef = nnx.graphdef(td7_state.embedding)
        embedding2 = load_pickle(filename, graphdef, "cpu")
    finally:
        os.remove(filename)

    out1 = td7_state.embedding(state=jnp.ones(11), action=jnp.ones(3))
    out2 = embedding2(
        state=jax.device_put(jnp.ones(11), jax.devices("cpu")[0]),
        action=jax.device_put(jnp.ones(3), jax.devices("cpu")[0]),
    )

    assert (
        embedding2.state_action_embedding.hidden_layers[0].kernel.value.device
        == jax.devices("cpu")[0]
    )
    assert out2[0].device == jax.devices("cpu")[0]

    assert jax.tree_util.tree_all(
        jax.tree.map(partial(jnp.allclose, atol=1e-5), out1, out2)
    )
