from functools import partial
from typing import List, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from rl_blox.model.gaussian_mlp_ensemble import EnsembleOfGaussianMlps


def train_ddpg():
    raise NotImplementedError()
