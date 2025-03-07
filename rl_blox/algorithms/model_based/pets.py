import gymnasium as gym
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from rl_blox.model.gaussian_mlp_ensemble import EnsembleOfGaussianMlps


def train_pets(
    env: gym.Env,
    dynamics_model: EnsembleOfGaussianMlps,
    horizon: int,
):
    """Probabilistic Ensemble - Trajectory Sampling (PE-TS).

    Each probabilistic neural network of the probabilistic ensemble (PE)
    dynamics model captures aleatoric uncertainty (inherent variance of the
    observed data). The ensemble captures epistemic uncertainty through
    bootstrap disagreement far from data. The trajectory sampling (TS)
    propagation technique uses this dynamics model to re-sample each particle
    (with associated bootstrap) according to its probabilistic prediction at
    each point in time, up until a given horizon. At each time step, the
    model-predictive control (MPC) algorithm computes an optimal action
    sequence, applies the first action in the sequence, and repeats until the
    task-horizon.

    Parameters
    ----------
    env
        gymnasium environment.
    dynamics_model
        Probabilistic ensemble dynamics model.
    sampling_horizon
        Horizon length for trajectory sampling.

    References
    ----------
    .. [1] Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine.
           2018. Deep reinforcement learning in a handful of trials using
           probabilistic dynamics models. In Proceedings of the 32nd
           International Conference on Neural Information Processing Systems
           (NeurIPS'18). Curran Associates Inc., Red Hook, NY, USA, 4759–4770.
           https://papers.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html
    """
    raise NotImplementedError()
