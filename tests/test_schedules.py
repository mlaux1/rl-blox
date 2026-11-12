import numpy as np
import pytest

from rl_blox.blox.schedules import linear_schedule, linear_schedule_value


@pytest.mark.parametrize("total_timesteps", [5, 10, 15, 1000])
@pytest.mark.parametrize("kwargs", [{}, dict(start=0.4, end=1.0, fraction=1.0)])
def test_linear_schedule_value_matches_linear_schedule(total_timesteps, kwargs):
    values = [
        linear_schedule_value(step, total_timesteps, **kwargs)
        for step in range(total_timesteps)
    ]
    np.testing.assert_allclose(
        values, linear_schedule(total_timesteps, **kwargs), atol=1e-6
    )
