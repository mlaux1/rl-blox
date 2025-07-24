import chex
import jax
import jax.numpy as jnp


def make_two_hot_bins(
    lower_exponent: float = -10.0,
    upper_exponent: float = 10.0,
    n_bin_edges: int = 101,
):
    r"""Create bins for two-hot encoding of continuous values.

    To handle a wide range of magnitudes without prior knowledge, the locations
    of the two-hot encoding are spaced at increasing non-uniform intervals,
    according to :math:`\text{symexp}(x) = \text{sign}(x) (\exp(x) - 1)` with
    :math:`x \in \left[ x_{lo}, x_{hi} \right]`.

    Parameters
    ----------
    lower_exponent : float, optional
        Lower exponent :math:`x_{lo}` to be transformed by symmetric
        exponential function to define lower limit of the bins.

    upper_exponent : float, optional
        Upper exponent :math:`x_{hi}` to be transformed by symmetric
        exponential function to define upper limit of the bins.

    n_bin_edges : int, optional
        Number of bin edges to create. Must be a positive integer.

    Returns
    -------
    bins : array, shape (n_bin_edges,)
        Bin edges for two-hot encoding of continuous values.
    """
    chex.assert_scalar_positive(n_bin_edges)
    bins = jnp.linspace(lower_exponent, upper_exponent, n_bin_edges)
    return jnp.sign(bins) * (jnp.exp(jnp.abs(bins)) - 1.0)  # symexp


def two_hot_encoding(bins: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Two-hot encoding of the input array.

    Parameters
    ----------
    bins : array, shape (n_bin_edges,)
        Bin edges for two-hot encoding of continuous values.

    x : array, shape (n_samples,)
        Input array of continuous values to be encoded.

    Returns
    -------
    two_hot_encoded : array, shape (n_samples, n_bins)
        Two-hot encoded representation of the input values. Each row
        contains at most two non-zero, positive values that sum up to 1
        and encode the corresponding real value. If the real value is
        exactly at one of the bin edges, only one non-zero value is present.
    """
    diff = x[:, jnp.newaxis] - bins[jnp.newaxis]
    diff = diff - 1e8 * (jnp.sign(diff) - 1)
    ind_lo = jnp.argmin(diff, 1, keepdims=False)
    ind_up = jnp.clip(ind_lo + 1, 0, bins.shape[0] - 1)

    lower = bins[ind_lo]
    upper = bins[ind_up]
    weight = (x - lower) / (upper - lower)

    two_hot = jnp.zeros((x.shape[0], bins.shape[0]))
    two_hot = two_hot.at[jnp.arange(x.shape[0]), ind_lo].set(1.0 - weight)
    two_hot = two_hot.at[jnp.arange(x.shape[0]), ind_up].set(weight)
    return two_hot


def two_hot_decoding(
    bins: jnp.ndarray, two_hot_encoded: jnp.ndarray
) -> jnp.ndarray:
    """Inverse of the two-hot encoding.

    Parameters
    ----------
    bins : array, shape (n_bin_edges,)
        Bin edges for two-hot encoding of continuous values.

    two_hot_encoded : array, shape (n_samples, n_bins)
        Two-hot encoded representation of real values. Each row
        contains at most two non-zero, positive values that sum up to 1
        and encode the corresponding real value. If the real value is
        exactly at one of the bin edges, only one non-zero value is present.

    Returns
    -------
    x : array, shape (n_samples,)
        Original real values recovered from two-hot encoding.
    """
    return jnp.sum(two_hot_encoded * bins, axis=-1)


def two_hot_cross_entropy_loss(
    bins: jnp.ndarray, logits: jnp.ndarray, target: jnp.ndarray
) -> jnp.ndarray:
    """Cross-entropy (CE) loss between two-hot encoded prediction and targets.

    The CE loss of two-hot encoded real values is more effective than
    mean squared error (MSE) loss for sparse (i.e., mostly 0) real values, and
    it is more robust to different magnitudes.

    Parameters
    ----------
    bins : array, shape (n_bin_edges,)
        Bin edges for two-hot encoding of continuous values.

    logits : array, shape (n_samples, n_bins)
        Predicted logits of two-hot encoded representation of real values.
        A softmax function would transform these logits into a two-hot
        representation of the approximated real values.

    target : array, shape (n_samples,)
        Real target values to be encoded and compared against the
        predictions with the cross-entropy loss.

    Returns
    -------
    ce_loss : array, shape (n_samples,)
        Cross-entropy loss values for each sample.
    """
    log_pred = jax.nn.log_softmax(logits, axis=-1)
    target = two_hot_encoding(bins, target)
    return -jnp.sum(target * log_pred, axis=-1)
