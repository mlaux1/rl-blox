import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx

from rl_blox.model.probabilistic_ensemble import (
    GaussianMlpEnsemble,
    bootstrap,
    train_step,
)


def generate_dataset1(data_key, n_samples):
    x = jnp.linspace(0, 4, n_samples)
    y_true = jnp.exp(x)
    y = y_true + (max(x) - x) ** 2 * jax.random.normal(data_key, x.shape)
    X_train = x[:, jnp.newaxis]
    Y_train = y[:, jnp.newaxis]
    return X_train, Y_train, x[:, jnp.newaxis], y_true[:, jnp.newaxis]


def generate_dataset2(data_key, n_samples):
    x1 = jnp.linspace(-1.0, -0.5, n_samples // 2)
    x2 = jnp.linspace(0.5, 1.0, n_samples // 2)
    x_train = jnp.hstack((x1, x2))
    y_train = x_train**3 + 0.1 * jax.random.normal(data_key, x_train.shape)
    x_test = jnp.linspace(-1.5, 1.5, n_samples)
    y_test = x_test**3
    return (
        x_train[:, jnp.newaxis],
        y_train[:, jnp.newaxis],
        x_test[:, jnp.newaxis],
        y_test[:, jnp.newaxis],
    )


def generate_dataset3(data_key, n_samples):
    x1 = jnp.linspace(-2.0 * jnp.pi, -jnp.pi, n_samples // 2)
    x2 = jnp.linspace(jnp.pi, 2.0 * jnp.pi, n_samples // 2)
    x_train = jnp.hstack((x1, x2))
    y_train = jnp.sin(x_train) + 0.1 * jax.random.normal(
        data_key, x_train.shape
    )
    x_test = jnp.linspace(-3.0 * jnp.pi, 3.0 * jnp.pi, n_samples)
    y_test = jnp.sin(x_test)
    return (
        x_train[:, jnp.newaxis],
        y_train[:, jnp.newaxis],
        x_test[:, jnp.newaxis],
        y_test[:, jnp.newaxis],
    )


seed = 42
learning_rate = 3e-3
n_samples = 200
batch_size = 50
n_epochs = 3_000
plot_base_models = True
train_size = 0.7

key = jax.random.PRNGKey(seed)
key, data_key = jax.random.split(key, 2)
X_train, Y_train, X_test, Y_test = generate_dataset3(data_key, n_samples)

model = GaussianMlpEnsemble(
    n_ensemble=5,
    shared_head=True,
    n_features=1,
    n_outputs=1,
    hidden_nodes=[100, 50],
    rngs=nnx.Rngs(seed),
)
opt = nnx.Optimizer(model, optax.adam(learning_rate=learning_rate))

# TODO mini-batches
key, bootstrap_key = jax.random.split(key, 2)
bootstrap_indices = bootstrap(
    model.n_ensemble, train_size, n_samples, bootstrap_key
)
for t in range(n_epochs):
    key, shuffle_key = jax.random.split(key, 2)
    shuffled_indices = jax.random.permutation(key, bootstrap_indices, axis=1)
    for batch_start in jnp.arange(0, shuffled_indices.shape[1], batch_size):
        batch_indices = shuffled_indices[
            :, batch_start : batch_start + batch_size
        ]
        loss = train_step(model, opt, X_train, Y_train, batch_indices)
    if t % 100 == 0:
        print(f"{t=}: {loss=}")
print(model)

plt.figure()
plt.scatter(X_train[:, 0], Y_train[:, 0], label="Samples")
plt.plot(X_test[:, 0], Y_test[:, 0], label="True function")
if plot_base_models:
    for i in range(model.n_ensemble):
        mean, var = model.base_predict(X_test, i)
        std_196 = 1.96 * jnp.sqrt(var).squeeze()
        mean = mean.squeeze()
        plt.fill_between(
            X_test[:, 0], mean - std_196, mean + std_196, alpha=0.3
        )
        plt.plot(
            X_test[:, 0], mean, ls="--", label=f"Prediction of model {i + 1}"
        )
mean, var = model.aggegrate(X_test)
mean = mean.squeeze()
std = jnp.sqrt(var).squeeze()
plt.plot(X_test[:, 0], mean, label="Ensemble", c="k")
for factor in [1.0, 2.0, 3.0]:
    plt.fill_between(
        X_test[:, 0],
        mean - factor * std,
        mean + factor * std,
        color="k",
        alpha=0.1,
    )
min_y = Y_test.min()
max_y = Y_test.max()
plt.ylim((min_y - 2, max_y + 2))
plt.legend(loc="best")
plt.show()
