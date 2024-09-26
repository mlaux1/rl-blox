from functools import partial
from typing import List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from benchmarks.benchmark_distance_point_to_triangle import random_state
from flax.training.train_state import TrainState
import optax


class GaussianMlp(nn.Module):
    """Base model of probabilistic ensemble."""

    shared_head: bool
    """All nodes of the last hidden layer are connected to each mean AND log_std output."""
    n_outputs: int
    """Number of output components"""
    hidden_nodes: List[int]
    """Numbers of hidden nodes of the MLP."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:# -> Tuple[jnp.ndarray, jnp.ndarray]:
        for n_nodes in self.hidden_nodes:
            x = nn.Dense(n_nodes)(x)
            x = nn.softplus(x)
        x = nn.Dense(2 * self.n_outputs)(x)
        if self.shared_head:
            mean, log_std = jnp.split(x, (self.n_outputs,), axis=-1)
        else:
            mean = nn.Dense(self.n_outputs)(x)
            log_std = nn.Dense(self.n_outputs)(x)
        return mean, log_std


@jax.jit
def heteroscedastic_aleatoric_uncertainty_loss(mean_pred, log_std_pred, y):
    """Heteroscedastic aleatoric uncertainty loss for Gaussian NN.

    References
    ----------
    .. [1] Kendall, Gal (2017). What Uncertainties Do We Need in Bayesian Deep
       Learning for Computer Vision? In Advances in Neural Information
       Processing Systems (NeurIPS). https://arxiv.org/abs/1703.04977,
       https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf
    """
    var = jnp.exp(log_std_pred) ** 2
    squared_erros = optax.l2_loss(mean_pred, y)  # including factor 0.5
    # Second term should be 0.5 * jnp.mean(jnp.log(var)), but this is the same
    # because 2 * log_std_pred == jnp.log(var), so 2 and 0.5 cancel out.
    return jnp.mean(squared_erros / var) + jnp.mean(log_std_pred)


def update_base_model(model, train_state, X, y, n_iter):
    @jax.jit
    def compute_loss(X, y, params):
        mean_pred, log_std_pred = model.apply(params, X)
        return heteroscedastic_aleatoric_uncertainty_loss(mean_pred, log_std_pred, y)
    loss = partial(compute_loss, X, y)
    for _ in range(n_iter):
        loss_value, grads = jax.value_and_grad(loss)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        print(loss_value)
    return loss_value, train_state


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
    y_train = x_train ** 3 + 0.1 * jax.random.normal(data_key, x_train.shape)
    x_test = jnp.linspace(-1.5, 1.5, n_samples)
    y_test = x_test ** 3
    return (x_train[:, np.newaxis], y_train[:, np.newaxis],
            x_test[:, np.newaxis], y_test[:, np.newaxis])


def generate_dataset3(data_key, n_samples):
    x1 = jnp.linspace(-2.0 * jnp.pi, -jnp.pi, n_samples // 2)
    x2 = jnp.linspace(jnp.pi, 2.0 * jnp.pi, n_samples // 2)
    x_train = jnp.hstack((x1, x2))
    y_train = jnp.sin(x_train) + 0.1 * jax.random.normal(data_key, x_train.shape)
    x_test = jnp.linspace(-4.0 * jnp.pi, 4.0 * jnp.pi, n_samples)
    y_test = jnp.sin(x_test)
    return (x_train[:, np.newaxis], y_train[:, np.newaxis],
            x_test[:, np.newaxis], y_test[:, np.newaxis])


seed = 42
learning_rate = 3e-3
n_samples = 200
batch_size = n_samples
n_epochs = 5000
net = GaussianMlp(shared_head=True, n_outputs=1, hidden_nodes=[50, 30])

random_state = np.random.RandomState(seed)
key = jax.random.PRNGKey(seed)

key, data_key = jax.random.split(key, 2)
X_train, Y_train, X_test, Y_test = generate_dataset3(data_key, n_samples)

key, net_key = jax.random.split(key, 2)
train_state = TrainState.create(
    apply_fn=net.apply,
    params=net.init(net_key, random_state.randn(1)),
    tx=optax.adam(learning_rate=learning_rate),
)

#indices = jnp.arange(len(X))
#for _ in range(n_epochs):
#    for i in range(n_samples // batch_size):
#        batch_indices = random_state.choice(indices, batch_size, False)
#        batch_X = X[batch_indices]
#        batch_Y = Y[batch_indices]
loss_value, train_state = update_base_model(net, train_state, X_train, Y_train, n_epochs)


import matplotlib.pyplot as plt


plt.figure()
plt.scatter(X_train[:, 0], Y_train[:, 0], label="Samples")
plt.plot(X_test[:, 0], Y_test[:, 0], label="True function")
mean, log_std = net.apply(train_state.params, X_test)
std_196 = 1.96 * jnp.exp(log_std).squeeze()
mean = mean.squeeze()
plt.fill_between(X_test[:, 0], mean - std_196, mean + std_196, alpha=0.3)
plt.plot(X_test[:, 0], mean, label="Prediction")
plt.show()
