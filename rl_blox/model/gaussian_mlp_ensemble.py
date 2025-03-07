from functools import partial
from typing import List

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


class EnsembleOfGaussianMlps:
    """Ensemble of Gaussian MLPs.

    References
    ----------
    .. [1] Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine.
           2018. Deep reinforcement learning in a handful of trials using
           probabilistic dynamics models. In Proceedings of the 32nd
           International Conference on Neural Information Processing Systems
           (NeurIPS'18). Curran Associates Inc., Red Hook, NY, USA, 4759–4770.
           https://papers.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html
    """

    def __init__(
        self,
        base_model,
        n_base_models,
        train_size,
        warm_start,
        learning_rate,
        key,
        verbose=0,
    ):
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

        self.key, bootstrapping_key = jax.random.split(self.key, 2)
        bootstrapped_indices = jax.random.choice(
            bootstrapping_key,
            n_samples,
            shape=(self.n_base_models, n_bootstrapped),
            replace=True,
        )

        X_train = X[bootstrapped_indices]
        Y_train = Y[bootstrapped_indices]

        if not hasattr(self, "train_states_") or not self.warm_start:
            self.key, init_key = jax.random.split(self.key, 2)
            model_keys = jax.random.split(init_key, self.n_base_models)

            def create_train_state(key, X_train):
                return TrainState.create(
                    apply_fn=self.base_model.apply,
                    params=self.base_model.init(key, X_train[0]),
                    tx=optax.adam(learning_rate=self.learning_rate),
                )

            create_train_states = jax.vmap(create_train_state, in_axes=(0, 0))
            self.train_states_ = create_train_states(model_keys, X_train)

        def update_base_model(train_state, X, y):
            def compute_loss(X, y, params):
                mean_pred, log_std_pred = self.base_model.apply(params, X)
                return heteroscedastic_aleatoric_uncertainty_loss(
                    mean_pred, log_std_pred, y
                )

            loss = partial(compute_loss, X, y)
            loss_value, grads = jax.value_and_grad(loss)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return loss_value, train_state

        update_base_models = jax.jit(
            jax.vmap(update_base_model, in_axes=(0, 0, 0))
        )

        for _ in range(n_epochs):
            loss_value, self.train_states_ = update_base_models(
                self.train_states_, X_train, Y_train
            )

        if self.verbose:
            print(f"loss {loss_value}")

    def predict(self, X):
        def base_model_predict(train_state, X):
            return self.base_model.apply(train_state.params, X)

        ensemble_predict = jax.jit(
            jax.vmap(base_model_predict, in_axes=(0, None))
        )
        means, log_stds = ensemble_predict(self.train_states_, X)
        return gaussian_ensemble_prediction(means, log_stds)


@jax.jit
def gaussian_ensemble_prediction(
    means: List[jnp.ndarray], log_stds: List[jnp.ndarray]
):
    n_base_models = len(means)
    mean = jnp.mean(means, axis=0)
    epistemic_var = jnp.sum((means - mean) ** 2, axis=0) / (n_base_models + 1)
    aleatoric_var = jnp.mean(jnp.exp(log_stds) ** 2, axis=0)
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
    def __call__(
        self, x: jnp.ndarray
    ) -> jnp.ndarray:  # -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    # var = jnp.where(var < 1e-6, 1.0, var)  # TODO do we need this?
    squared_erros = optax.l2_loss(mean_pred, Y)  # including factor 0.5
    # Second term should be 0.5 * jnp.mean(jnp.log(var)), but this is the same
    # because 2 * log_std_pred == jnp.log(var), so 2 and 0.5 cancel out.
    return jnp.mean(squared_erros / var) + jnp.mean(log_std_pred)
