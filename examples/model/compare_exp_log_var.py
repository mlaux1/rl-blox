import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def clipped_exp(x, min_x, max_x):
    return jnp.exp(jnp.clip(x, min_x, max_x))


def soft_clipped_exp(x, min_x, max_x):
    upper_clipped_x = max_x - jax.nn.softplus(max_x - x)
    clipped_x = min_x + jax.nn.softplus(upper_clipped_x - min_x)
    return jnp.exp(clipped_x)


x = jnp.linspace(-20, 20, 1001)

min_x = -10
max_x = 2

plt.figure()
plt.plot(x, jnp.exp(x), label="exp")
plt.plot(x, clipped_exp(x, min_x, max_x), label="clipped exp")
plt.plot(x, soft_clipped_exp(x, min_x, max_x), label="soft clipped exp")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim((0, 10))
plt.legend()
plt.show()
