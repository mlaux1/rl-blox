from functools import partial
from typing import List
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax


class BaseModel(nn.Module):
    """Base model of probabilistic ensemble."""

    n_outputs: int
    """Number of output components"""
    hidden_nodes: List[int]
    """Numbers of hidden nodes of the MLP."""

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        for n_nodes in self.hidden_nodes:
            x = nn.Dense(n_nodes)(x)
            x = nn.relu(x)
        #x = nn.Dense(2 * self.n_outputs)(x)
        #mean, log_std = jnp.split(x, (self.n_outputs,), axis=-1)
        mean = nn.Dense(self.n_outputs)(x)
        log_std = nn.Dense(self.n_outputs)(x)
        return mean, log_std


def heteroscedastic_aleatoric_uncertainty_loss(model, X, y, params):
    mean, log_std = model.apply(params, X)
    var = jnp.exp(log_std) ** 2
    squared_erros = (y - mean) ** 2
    return jnp.mean(squared_erros / var) + jnp.mean(log_std)


def update_base_model(base_model, train_state, X, y):
    loss = partial(heteroscedastic_aleatoric_uncertainty_loss, base_model, X, y)
    loss_value, grads = jax.value_and_grad(loss)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return loss_value, train_state


seed = 42
learning_rate = 1e-3
batch_size = 16
n_epochs = 100
net = BaseModel(n_outputs=1, hidden_nodes=[300, 300, 10])

key = jax.random.PRNGKey(seed)

key, data_key = jax.random.split(key, 2)
n_samples = 200
x = jnp.linspace(0, 4, n_samples)
y_true = jnp.exp(x)
y = y_true + (max(x) - x) ** 2 * jax.random.normal(data_key, x.shape)
X = x[:, np.newaxis]

key, net_key = jax.random.split(key, 2)
train_state = TrainState.create(
    apply_fn=net.apply,
    params=net.init(net_key, jnp.array([0.0])),
    tx=optax.adam(learning_rate=learning_rate),
)
indices = jnp.arange(len(X))
random_state = np.random.RandomState(seed)
for _ in range(n_epochs):
    for i in range(n_samples // batch_size):
        batch_indices = random_state.choice(indices, batch_size, False)
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        loss_value, train_state = update_base_model(net, train_state, batch_X, batch_y)
        print(loss_value)


import matplotlib.pyplot as plt


plt.figure()
plt.scatter(x, y, label="Samples")
plt.plot(x, y_true, label="True function")
mean, log_std = net.apply(train_state.params, X)
mean = mean.squeeze()
std = jnp.exp(log_std).squeeze()
plt.fill_between(x, mean - std, mean + std, alpha=0.3)
plt.plot(x, mean, label="Prediction")
plt.show()
