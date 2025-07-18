import jax.nn
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


def test_two_hot_cross_entropy_loss():
    bins = make_two_hot_bins(
        lower_exponent=-5.0, upper_exponent=5.0, n_bin_edges=5
    )
    logits = -1000.0 * jnp.ones((1, 5))
    logits = logits.at[0, 2].set(10.0)
    logits = logits.at[0, 3].set(10.0)
    two_hot_encoded = jax.nn.softmax(logits, axis=-1)
    target = two_hot_decoding(bins, two_hot_encoded)
    loss = two_hot_cross_entropy_loss(bins, logits, target)

    assert loss.shape == (1,), "Loss shape mismatch"
    assert loss.mean() == 0.6931472, "Regression test"
