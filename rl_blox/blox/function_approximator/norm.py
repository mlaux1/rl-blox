import jax.numpy as jnp


def avg_l1_norm(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    r"""AvgL1Norm.

    A normalization layer that divides the input vector by its average absolute
    value in each dimension, thus keeping the relative scale of the embedding
    constant throughout learning.

    Let :math:`x_i` by the i-th dimension of an N-dimensional vector x, then

    .. math::

        \text{AvgL1Norm}(x)
        :=
        \frac{x}{\max(\frac{1}{N} \sum_i |x_i|, \epsilon)},

    with a small constant :math:`\epsilon`.

    AvgL1Norm protects from monotonic growth, but also keeps the scale of the
    downstream input constant without relying on updating statistics.

    Parameters
    ----------
    x : array
        Input vector(s).

    eps : float
        Small constant.

    Returns
    -------
    avg_l1_norm_x : float
        AvgL1Norm(x).
    """
    return x / jnp.maximum(jnp.mean(jnp.abs(x), axis=-1, keepdims=True), eps)
