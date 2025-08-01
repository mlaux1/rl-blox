import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx

from rl_blox.blox.probabilistic_ensemble import (
    GaussianMLPEnsemble,
    train_ensemble,
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
batch_size = 32
n_epochs = 1_000
train_size = 0.7
plot_base_models = True

key = jax.random.key(seed)
key, data_key = jax.random.split(key, 2)
X_train, Y_train, X_test, Y_test = generate_dataset3(data_key, n_samples)

model = GaussianMLPEnsemble(
    n_ensemble=5,
    shared_head=False,
    n_features=1,
    n_outputs=1,
    hidden_nodes=[100, 50],
    activation="swish",
    rngs=nnx.Rngs(seed),
)
opt = nnx.Optimizer(
    model, optax.adam(learning_rate=learning_rate), wrt=nnx.Param
)

key, train_key = jax.random.split(key, 2)
train_ensemble(
    model,
    opt,
    train_size,
    X_train,
    Y_train,
    n_epochs,
    batch_size,
    train_key,
    verbose=2,
)
print(model)
print(f"{model.min_log_var=}")
print(f"{model.max_log_var=}")

plt.figure(figsize=(10, 5))

plt.subplot(211)
plt.scatter(X_train[:, 0], Y_train[:, 0], label="Training set")
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
mean, var = model.aggregate(X_test)
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
plt.ylim((min_y - 5, max_y + 5))
plt.legend(loc="best")

plt.subplot(212)
model_idx = 0
plt.title(f"Samples from model #{model_idx}")
plt.scatter(X_train[:, 0], Y_train[:, 0], label="Training set")
distribution = model.base_distribution(X_test, model_idx)
key, sampling_key = jax.random.split(key, 2)
samples = distribution.sample(seed=sampling_key, sample_shape=(10,))
x = jnp.broadcast_to(X_test[jnp.newaxis], samples.shape)
plt.scatter(x.flatten(), samples.flatten(), label="Samples", alpha=0.3, s=5)
x_test = jnp.array([[5.0]])
distribution_point = model.base_distribution(x_test, model_idx)
key, sampling_key = jax.random.split(key, 2)
samples = distribution_point.sample(seed=sampling_key, sample_shape=(100,))
plt.scatter(
    jnp.broadcast_to(x_test, samples.shape).flatten(),
    samples.flatten(),
    label="Samples",
    alpha=0.3,
    s=10,
)
plt.ylim((min_y - 5, max_y + 5))
plt.legend(loc="best")

plt.show()
