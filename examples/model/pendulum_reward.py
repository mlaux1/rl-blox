import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from rl_blox.algorithms.model_based.pets_reward_models import (
    PENDULUM_MAX_TORQUE,
    pendulum_reward,
)

n_samples = 101
n_actions = 9
angles = jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, n_samples)
speed = jnp.linspace(-8.0, 8.0, n_samples)
angle_grid, speed_grid = jnp.meshgrid(angles, speed)
cos_grid = jnp.cos(angle_grid)
sin_grid = jnp.sin(angle_grid)
obs_grid = jnp.stack((cos_grid, sin_grid, speed_grid), axis=-1)

actions = np.linspace(-PENDULUM_MAX_TORQUE, PENDULUM_MAX_TORQUE, n_actions)[
    :, jnp.newaxis
]

fig, axes = plt.subplots(n_actions // 3, 3)
for act, ax in zip(actions, axes.ravel(), strict=False):
    ax.set_title(f"torque = {act[0]}")
    rews = pendulum_reward(act, obs_grid)
    contour = ax.contourf(angles, speed, rews, vmin=-10, vmax=0)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label("Reward")
plt.xlabel("Angle")
plt.ylabel("Speed (angular velocity)")
plt.tight_layout()
plt.show()
