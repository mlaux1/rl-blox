import chex
import jax.numpy as jnp
import optax
from flax import nnx


def mse_value_loss(
    observations: jnp.ndarray,
    v_target: jnp.ndarray,
    v: nnx.Module,
) -> jnp.ndarray:
    r"""Mean squared error as loss for a value function network.

    For a given value function :math:`v(o)` and target values :math:`R(o)`, the
    loss is defined as

    .. math::

        \mathcal{L}(v) = \frac{1}{2 N} \sum_{i=1}^{N} (v(o_i) - R(o_i))^2.

    :math:`R(o)` could be the Monte Carlo return.

    Parameters
    ----------
    observations : array, shape (n_samples, n_observation_features)
        Observations.

    v_target : array, shape (n_samples,)
        Target values, obtained, e.g., through Monte Carlo sampling.

    v : nnx.Module
        Value function that maps observations to expected returns.

    Returns
    -------
    loss : float
        Value function loss.
    """
    values = v(observations).squeeze()  # squeeze Nx1-D -> N-D
    chex.assert_equal_shape((values, v_target))
    return optax.l2_loss(predictions=values, targets=v_target).mean()
