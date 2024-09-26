from codecs import replace_errors
from functools import partial
from typing import List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax


class EnsembleOfGaussianMlps:
    """Ensemble of Gaussian MLPs.

    References
    ----------
    .. [1] TODO PETS paper
    """
    def __init__(self, base_model, n_base_models, train_size, warm_start, learning_rate, key, verbose=0):
        self.base_model = base_model
        self.n_base_models = n_base_models
        self.train_size = train_size
        self.warm_start = warm_start
        self.learning_rate = learning_rate
        self.key = key
        self.verbose = verbose

    def fit(self, X, Y, n_epochs):
        n_samples = len(X)
        n_bootstrapped = int(self.train_size * n_samples)

        if not hasattr(self, "train_states_") or not self.warm_start:
            self.key, init_key = jax.random.split(self.key, 2)
            model_keys = jax.random.split(init_key, self.n_base_models)
            self.train_states_ = [
                TrainState.create(
                    apply_fn=self.base_model.apply,
                    params=self.base_model.init(key, X[0]),
                    tx=optax.adam(learning_rate=self.learning_rate),
                ) for key in model_keys]

        self.key, bootstrapping_key = jax.random.split(self.key, 2)
        bootstrapped_indices = jax.random.choice(
            bootstrapping_key, n_samples,
            shape=(self.n_base_models, n_bootstrapped), replace=True
        )

        for i in range(self.n_base_models):
            # TODO parallelize (vmap?)
            # TODO mini-batches?
            X_train = X[bootstrapped_indices[i]]
            Y_train = Y[bootstrapped_indices[i]]

            @jax.jit
            def update_base_model(train_state, X, y):
                def compute_loss(X, y, params):
                    mean_pred, log_std_pred = self.base_model.apply(params, X)
                    return heteroscedastic_aleatoric_uncertainty_loss(
                        mean_pred, log_std_pred, y)

                loss = partial(compute_loss, X, y)
                loss_value, grads = jax.value_and_grad(loss)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                return loss_value, train_state

            for _ in range(n_epochs):
                loss_value, self.train_states_[i] = update_base_model(
                    self.train_states_[i], X_train, Y_train)

            if self.verbose:
                print(f"base model {i + 1}, loss {loss_value:.4f}")

    def predict(self, X):
        means = []
        log_stds = []
        for i, train_state in enumerate(self.train_states_):
            mean_i, log_std_i = self.base_model.apply(train_state.params, X_test)
            means.append(mean_i)
            log_stds.append(log_std_i)
        return gaussian_ensemble_prediction(means, log_stds)


@jax.jit
def gaussian_ensemble_prediction(means: List[jnp.ndarray], log_stds: List[jnp.ndarray]):
    n_base_models = len(means)
    aleatoric_vars = [jnp.exp(log_std_i) ** 2 for log_std_i in log_stds]
    mean = jnp.mean(jnp.stack(means, axis=0), axis=0)
    aleatoric_var = jnp.mean(jnp.stack(aleatoric_vars, axis=0), axis=0)
    diffs = [(means[i] - mean) ** 2 for i in range(n_base_models)]
    epistemic_var = jnp.sum(jnp.stack(diffs, axis=0), axis=0) / (n_base_models + 1)
    return mean, aleatoric_var + epistemic_var


class GaussianMlp(nn.Module):
    """Probabilistic neural network that predicts a Gaussian distribution."""

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
def heteroscedastic_aleatoric_uncertainty_loss(mean_pred, log_std_pred, Y):
    """Heteroscedastic aleatoric uncertainty loss for Gaussian NN.

    .. math::

        -\log p_{\theta}(Y|X) = \frac{1}{N}\sum_n
        \frac{1}{2} \frac{(y_n - \mu_{\theta}(x_n))^2}{\sigma_{\theta}^2(x)}
        + \frac{1}{2} \log \sigma_{\theta}^2(x_n) + \text{constant}

    This is the negative log-likelihood of Gaussian distributions predicted
    by a neural network, i.e., the neural network predicted a mean vector and
    a vector of component-wise log standard deviations per sample.

    :param mean_pred: Means of the predicted Gaussian distributions.
    :param log_std_pred: Logarithm of standard deviations of predicted Gaussian
                         distributions.
    :param Y: Actual outputs.
    :returns: Negative log-likelihood.

    References
    ----------
    .. [1] Nix, Weigand (1994). Estimating the mean and variance of the target
       probability distribution. in International Conference on Neural Networks
       (ICNN). https://doi.org/10.1109/ICNN.1994.374138

    .. [2] Kendall, Gal (2017). What Uncertainties Do We Need in Bayesian Deep
       Learning for Computer Vision? In Advances in Neural Information
       Processing Systems (NeurIPS). https://arxiv.org/abs/1703.04977,
       https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf

    .. [3] Lakshminarayanan, Pritzel, Blundell (2017): Simple and Scalable
       Predictive Uncertainty Estimation using Deep Ensembles. In Advances in
       Neural Information Processing Systems (NeurIPS).
       https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf
    """
    var = jnp.exp(log_std_pred) ** 2
    # TODO what if var == 0?
    squared_erros = optax.l2_loss(mean_pred, Y)  # including factor 0.5
    # Second term should be 0.5 * jnp.mean(jnp.log(var)), but this is the same
    # because 2 * log_std_pred == jnp.log(var), so 2 and 0.5 cancel out.
    return jnp.mean(squared_erros / var) + jnp.mean(log_std_pred)


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
    x_test = jnp.linspace(-3.0 * jnp.pi, 3.0 * jnp.pi, n_samples)
    y_test = jnp.sin(x_test)
    return (x_train[:, np.newaxis], y_train[:, np.newaxis],
            x_test[:, np.newaxis], y_test[:, np.newaxis])


seed = 42
learning_rate = 3e-3
n_samples = 200
batch_size = n_samples
n_epochs = 5000
plot_base_models = False

random_state = np.random.RandomState(seed)
key = jax.random.PRNGKey(seed)

net = GaussianMlp(shared_head=True, n_outputs=1, hidden_nodes=[50, 30])
net.apply = jax.jit(net.apply)
key, ensemble_key = jax.random.split(key, 2)
ensemble = EnsembleOfGaussianMlps(net, 5, 0.5, True, learning_rate, ensemble_key, verbose=1)

key, data_key = jax.random.split(key, 2)
X_train, Y_train, X_test, Y_test = generate_dataset3(data_key, n_samples)

#loss_value, train_state = update_base_model(net, train_state, X_train, Y_train, n_epochs, verbose=1)
ensemble.fit(X_train, Y_train, n_epochs)


import matplotlib.pyplot as plt


plt.figure()
plt.scatter(X_train[:, 0], Y_train[:, 0], label="Samples")
plt.plot(X_test[:, 0], Y_test[:, 0], label="True function")
if plot_base_models:
    for idx, train_state in enumerate(ensemble.train_states_):
        mean, log_std = net.apply(train_state.params, X_test)
        std_196 = 1.96 * jnp.exp(log_std).squeeze()
        mean = mean.squeeze()
        plt.fill_between(X_test[:, 0], mean - std_196, mean + std_196, alpha=0.3)
        plt.plot(X_test[:, 0], mean, ls="--", label=f"Prediction of model {idx + 1}")
mean, var = ensemble.predict(X_test)
mean = mean.squeeze()
std = jnp.sqrt(var).squeeze()
plt.plot(X_test[:, 0], mean, label="Ensemble", c="k")
for factor in [1.0, 2.0, 3.0]:
    plt.fill_between(
        X_test[:, 0], mean - factor * std, mean + factor * std, color="k",
        alpha=0.1)
plt.legend(loc="best")
plt.show()
