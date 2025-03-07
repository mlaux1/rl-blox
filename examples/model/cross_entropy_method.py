import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import cm

from rl_blox.model.cross_entropy_method import optimize_cem


def cost_function(x: jnp.ndarray) -> jnp.ndarray:
    x = x - jnp.arange(x.shape[-1])[jnp.newaxis] - 1.0
    return jnp.sum(x * x, axis=-1)


x = jnp.linspace(-4, 4, 101)
y = jnp.linspace(-4, 4, 101)
X, Y = jnp.meshgrid(x, y)
XY = jnp.stack((X, Y), axis=-1)
C = cost_function(XY)

plt.figure(figsize=(8, 4))

ax = plt.subplot(121)
init_mean = jnp.array([-3, -3])
init_var = 9 * jnp.ones(2)
key = jax.random.PRNGKey(42)
solution, path, samples = optimize_cem(
    cost_function,
    init_mean,
    init_var,
    key,
    n_iter=200,
    n_population=50,
    n_elite=20,
    lower_bound=jnp.array([x[0], y[0]]),
    upper_bound=jnp.array([x[-1], y[-1]]),
    epsilon=1e-8,
    alpha=0.95,
    return_history=True,
)

ax.contourf(X, Y, C, cmap=cm.PuBu_r)
ax.scatter(
    samples[:, 0], samples[:, 1], marker="x", c="gray", s=5, label="Samples"
)
ax.plot(path[:, 0], path[:, 1], ls="-.", c="orange", label="Search path")
ax.scatter(
    init_mean[0], init_mean[1], marker="x", c="k", s=20, label="Initial guess"
)
ax.scatter(solution[0], solution[1], marker="x", c="r", s=20, label="Solution")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim((x[0], x[-1]))
ax.set_ylim((y[0], y[-1]))
ax.legend(loc="best")

ax = plt.subplot(122)
init_mean = jnp.array([-1, -1])
init_var = 9 * jnp.ones(2)
key = jax.random.PRNGKey(42)
solution, path, samples = optimize_cem(
    cost_function,
    init_mean,
    init_var,
    key,
    n_iter=200,
    n_population=50,
    n_elite=20,
    lower_bound=jnp.array([x[0], y[0]]),
    upper_bound=jnp.array([x[-1], y[-1]]),
    epsilon=1e-8,
    alpha=0.95,
    return_history=True,
)

ax.contourf(X, Y, C, cmap=cm.PuBu_r)
ax.scatter(
    samples[:, 0], samples[:, 1], marker="x", c="gray", s=5, label="Samples"
)
ax.plot(path[:, 0], path[:, 1], ls="-.", c="orange", label="Search path")
ax.scatter(
    init_mean[0], init_mean[1], marker="x", c="k", s=20, label="Initial guess"
)
ax.scatter(solution[0], solution[1], marker="x", c="r", s=20, label="Solution")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim((x[0], x[-1]))
ax.set_ylim((y[0], y[-1]))
ax.legend(loc="best")

plt.tight_layout()
plt.show()
