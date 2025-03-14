from typing import Callable

import chex
import jax.numpy as jnp
import optax
from flax import nnx


class GaussianMlp(nnx.Module):
    """Probabilistic neural network that predicts a Gaussian distribution.

    Parameters
    ----------
    shared_head
        All nodes of the last hidden layer are connected to mean AND log_std.

    n_inputs
        Number of features.

    n_outputs
        Number of output components.

    hidden_nodes
        Numbers of hidden nodes of the MLP.

    rngs
        Random number generator.
    """

    shared_head: bool
    n_outputs: int
    hidden_layers: list[nnx.Linear]
    output_layers: list[nnx.Linear]

    def __init__(
        self,
        shared_head: bool,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        rngs: nnx.Rngs,
    ):
        chex.assert_scalar_positive(n_features)
        chex.assert_scalar_positive(n_outputs)

        self.shared_head = shared_head
        self.n_outputs = n_outputs

        self.hidden_layers = []
        n_in = n_features
        for n_out in hidden_nodes:
            self.hidden_layers.append(nnx.Linear(n_in, n_out, rngs=rngs))
            n_in = n_out

        self.output_layers = []
        if shared_head:
            self.output_layers.append(
                nnx.Linear(n_in, 2 * n_outputs, rngs=rngs)
            )
        else:
            self.output_layers.append(nnx.Linear(n_in, n_outputs, rngs=rngs))
            self.output_layers.append(nnx.Linear(n_in, n_outputs, rngs=rngs))

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        for layer in self.hidden_layers:
            x = nnx.swish(layer(x))

        if self.shared_head:
            y = self.output_layers[0](x)
            mean, log_var = jnp.split(y, (self.n_outputs,), axis=-1)
        else:
            mean = self.output_layers[0](x)
            log_var = self.output_layers[1](x)

        return mean, log_var


class GaussianMlpEnsemble(nnx.Module):
    ensemble: GaussianMlp
    n_ensemble: int
    n_outputs: int

    def __init__(
        self,
        n_ensemble: int,
        shared_head: bool,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        rngs: nnx.Rngs,
    ):
        self.n_ensemble = n_ensemble
        self.n_outputs = n_outputs

        @nnx.split_rngs(splits=self.n_ensemble)
        @nnx.vmap
        def make_model(rngs: nnx.Rngs) -> GaussianMlp:
            return GaussianMlp(
                shared_head=shared_head,
                n_features=n_features,
                n_outputs=n_outputs,
                hidden_nodes=hidden_nodes,
                rngs=rngs,
            )

        self.ensemble = make_model(rngs)

        def forward(
            model: GaussianMlp, x: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            return model(x)

        self._forward_ensemble = nnx.vmap(forward, in_axes=(0, None))
        self._forward_individual = nnx.vmap(forward, in_axes=(0, 0))

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if x.ndim == 2:
            means, log_vars = self._forward_ensemble(self.ensemble, x)
        elif x.ndim == 3:
            means, log_vars = self._forward_individual(self.ensemble, x)
        else:
            raise ValueError(f"{x.shape=}")

        # TODO replace by numerically safe method
        # logvar = max_logvar - jax.nn.softplus(max_logvar - logvar)
        # logvar = min_logvar + jax.nn.softplus(logvar - min_logvar)

        return means, log_vars

    def aggegrate(self, x):
        means, log_vars = self._forward_ensemble(self.ensemble, x)

        # TODO replace by numerically safe method
        # logvar = max_logvar - jax.nn.softplus(max_logvar - logvar)
        # logvar = min_logvar + jax.nn.softplus(logvar - min_logvar)

        mean = jnp.mean(means, axis=0)
        aleatoric_var = jnp.mean(jnp.exp(log_vars), axis=0)
        epistemic_var = jnp.var(means, axis=0)
        return mean, aleatoric_var + epistemic_var


def gaussian_nll(
    mean_pred: jnp.ndarray, log_var_pred: jnp.ndarray, Y: jnp.ndarray
) -> jnp.ndarray:
    r"""Heteroscedastic aleatoric uncertainty loss for Gaussian NN.

    This is the negative log-likelihood of Gaussian distributions
    :math:`p_{\theta}(Y|X)` predicted by a neural network, i.e., the neural
    network predicted a mean vector :math:`\mu_{\theta}(x_n)` and a vector of
    component-wise log variances :math:`\sigma_{\theta}^2(x_n)` per sample:

    .. math::

        -\log p_{\theta}(Y|X) = \frac{1}{N}\sum_n
        \frac{1}{2} \frac{(y_n - \mu_{\theta}(x_n))^2}{\sigma_{\theta}^2(x)}
        + \frac{1}{2} \log \sigma_{\theta}^2(x_n) + \text{constant}

    The loss was originally proposed by Nix and Weigand [1]_.

    Parameters
    ----------
    mean_pred : array, shape (n_samples, n_outputs)
        Means of the predicted Gaussian distributions.

    log_var_pred : array, shape (n_samples, n_outputs)
        Logarithm of variances of predicted Gaussian distributions.

    Y : array, shape (n_samples, n_outputs)
        Actual outputs.

    Returns
    -------
    nll
        Negative log-likelihood.

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
    chex.assert_equal_shape((mean_pred, Y))
    chex.assert_equal_shape((log_var_pred, Y))

    inv_var = jnp.exp(-log_var_pred)  # exp(-log_var) == 1.0 / exp(log_var)
    squared_errors = optax.l2_loss(mean_pred, Y)  # including factor 0.5
    return jnp.mean(squared_errors * inv_var) + 0.5 * jnp.mean(log_var_pred)


@nnx.jit
def train_step(model, optimizer, X, Y):
    gaussian_ensemble_nll = nnx.vmap(gaussian_nll, in_axes=(0, 0, None))

    def loss(model):
        mean, log_var = model(X)
        return gaussian_ensemble_nll(mean, log_var, Y).sum()

    loss, grads = nnx.value_and_grad(loss)(model)
    optimizer.update(grads)

    return loss
