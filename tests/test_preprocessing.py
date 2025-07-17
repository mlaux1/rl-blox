import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal

from rl_blox.blox.preprocessing import (
    make_two_hot_bins,
    two_hot_cross_entropy_loss,
    two_hot_decoding,
    two_hot_encoding,
)


def test_two_hot_encoder():
    bins = make_two_hot_bins(
        lower_exponent=-10.0, upper_exponent=10.0, n_bin_edges=65
    )

    x = jnp.array([-100.0, -10.0, 0.1, 10.0, 100.0])

    two_hot_encoded = two_hot_encoding(bins, x)

    assert two_hot_encoded.shape == (5, 65), "Encoded shape mismatch"
    # two nonzero values per encoded value (only if not at the edges though!)
    assert_array_almost_equal(
        jnp.count_nonzero(two_hot_encoded, axis=1), 2 * jnp.ones(5)
    )
    assert_array_almost_equal(jnp.sum(two_hot_encoded, axis=1), jnp.ones(5))
    assert jnp.all(two_hot_encoded >= 0)

    # Test inverse
    decoded = two_hot_decoding(bins, two_hot_encoded)

    assert_array_almost_equal(decoded, x, decimal=2)

    # Test cross-entropy loss
    # TODO not quite sure what the input should be
    loss = two_hot_cross_entropy_loss(bins, two_hot_encoded, x)

    assert loss.shape == (5,), "Loss shape mismatch"
    assert loss.mean() == 0.0
