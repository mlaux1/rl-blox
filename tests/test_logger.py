import numpy as np
from numpy.testing import assert_array_almost_equal

from rl_blox.logging.logger import MemoryLogger


def test_memory_logger():
    logger = MemoryLogger()
    logger.define_experiment("TestEnv", "AlgorithmEnv", {})
    logger.start_new_episode()
    for i in range(10):
        logger.record_stat("1 / i", 1.0 / (1 + i), step=i + 1)
    logger.stop_episode(10)
    x, y = logger.get_stat("1 / i", x_key="step")
    assert_array_almost_equal(x, np.arange(1, 11))
    assert_array_almost_equal(y, 1.0 / np.arange(1, 11))
