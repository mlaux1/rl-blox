import jax
from numpy.testing import assert_array_almost_equal
from rl_blox.algorithms.model_free.sac import NormalizeObservationStreamX, NormalizeObservationSimBa


def test_compare_running_observation_stats():
    key = jax.random.key(0)
    observations = jax.random.normal(key, (1000, 2))
    nosx = NormalizeObservationStreamX()
    nosb = NormalizeObservationSimBa()
    for obs in observations:
        nosb.add_sample(obs)
        nosx.add_sample(obs)
    assert_array_almost_equal(nosx.running_stats.mean, nosb.mean)
    assert_array_almost_equal(nosx.running_stats.var, nosb.var, decimal=2)
