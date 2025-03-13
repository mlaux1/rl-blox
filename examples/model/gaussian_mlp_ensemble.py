import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from rl_blox.model.gaussian_mlp_ensemble import EnsembleOfGaussianMlps


def generate_dataset1(data_key, n_samples):
    x = jnp.linspace(0, 4, n_samples)
    y_true = jnp.exp(x)
    y = y_true + (max(x) - x) ** 2 * jax.random.normal(data_key, x.shape)
    X_train = x[:, np.newaxis]
    Y_train = y[:, np.newaxis]
    return X_train, Y_train, x[:, np.newaxis], y_true[:, np.newaxis]


def generate_dataset2(data_key, n_samples):
    x1 = jnp.linspace(-1.0, -0.5, n_samples // 2)
    x2 = jnp.linspace(0.5, 1.0, n_samples // 2)
    x_train = jnp.hstack((x1, x2))
    y_train = x_train**3 + 0.1 * jax.random.normal(data_key, x_train.shape)
    x_test = jnp.linspace(-1.5, 1.5, n_samples)
    y_test = x_test**3
    return (
        x_train[:, np.newaxis],
        y_train[:, np.newaxis],
        x_test[:, np.newaxis],
        y_test[:, np.newaxis],
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
        x_train[:, np.newaxis],
        y_train[:, np.newaxis],
        x_test[:, np.newaxis],
        y_test[:, np.newaxis],
    )


seed = 42
learning_rate = 3e-3
n_samples = 200
batch_size = n_samples
n_epochs = 15000
plot_base_models = True

random_state = np.random.RandomState(seed)
key = jax.random.PRNGKey(seed)

key, ensemble_key = jax.random.split(key, 2)
ensemble = EnsembleOfGaussianMlps.create(
    1,
    [100, 50],
    5,
    ensemble_key,
    shared_head=True,
    train_size=0.7,
    warm_start=True,
    learning_rate=learning_rate,
    verbose=1,
)

key, data_key = jax.random.split(key, 2)
X_train, Y_train, X_test, Y_test = generate_dataset3(data_key, n_samples)

ensemble.fit(X_train, Y_train, n_epochs)

plt.figure()
plt.scatter(X_train[:, 0], Y_train[:, 0], label="Samples")
plt.plot(X_test[:, 0], Y_test[:, 0], label="True function")
if plot_base_models:
    for i in range(ensemble.n_base_models):
        mean, log_var = ensemble.base_predict(X_test, i)
        std_196 = 1.96 * jnp.exp(0.5 * log_var).squeeze()
        mean = mean.squeeze()
        plt.fill_between(
            X_test[:, 0], mean - std_196, mean + std_196, alpha=0.3
        )
        plt.plot(
            X_test[:, 0], mean, ls="--", label=f"Prediction of model {i + 1}"
        )
mean, var = ensemble.predict(X_test)
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
