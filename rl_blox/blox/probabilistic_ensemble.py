from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax.distributions as dist
from flax import nnx
from jax.typing import ArrayLike

from .function_approximator.gaussian_mlp import GaussianMLP


def constrained_param(
    x: jnp.ndarray, min_val: ArrayLike, max_val: ArrayLike
) -> jnp.ndarray:
    """Compute sigmoid-constrained parameter."""
    return min_val + (max_val - min_val) * jax.nn.sigmoid(x)


class GaussianMLPEnsemble(nnx.Module):
    """Ensemble of Gaussian MLPs.

    Parameters
    ----------
    n_ensemble
        Number of individual Gaussian MLPs.

    shared_head
        All nodes of the last hidden layer are connected to mean AND log_var.

    n_features
        Number of features.

    n_outputs
        Number of output components.

    hidden_nodes
        Numbers of hidden nodes of the MLP.

    activation : str
        Activation function. Has to be the name of a function defined in the
        flax.nnx module.

    rngs
        Random number generator.

    References
    ----------
    .. [1] Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine.
           2018. Deep reinforcement learning in a handful of trials using
           probabilistic dynamics models. In Proceedings of the 32nd
           International Conference on Neural Information Processing Systems
           (NeurIPS'18). Curran Associates Inc., Red Hook, NY, USA, 4759–4770.
           https://papers.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html
    """

    ensemble: GaussianMLP
    n_ensemble: int
    n_outputs: int

    def __init__(
        self,
        n_ensemble: int,
        shared_head: bool,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        activation: str,
        rngs: nnx.Rngs,
    ) -> None:
        self.n_ensemble = n_ensemble
        self.n_outputs = n_outputs

        @nnx.split_rngs(splits=self.n_ensemble)
        @nnx.vmap
        def make_model(rngs: nnx.Rngs) -> GaussianMLP:
            return GaussianMLP(
                shared_head=shared_head,
                n_features=n_features,
                n_outputs=n_outputs,
                hidden_nodes=hidden_nodes,
                activation=activation,
                rngs=rngs,
            )

        self.ensemble = make_model(rngs)

        # TODO move safe_log_var to nnx.Module
        def safe_log_var(log_var, min_log_var, max_log_var):
            log_var = max_log_var - nnx.softplus(max_log_var - log_var)
            log_var = min_log_var + nnx.softplus(log_var - min_log_var)
            return log_var

        self._safe_log_var_i = nnx.vmap(safe_log_var, in_axes=(0, None, None))
        self._safe_log_var = nnx.vmap(
            self._safe_log_var_i,
            in_axes=(0, None, None),
        )

        self.raw_min_log_var = nnx.Param(jnp.zeros(self.n_outputs))
        self.raw_max_log_var = nnx.Param(jnp.zeros(self.n_outputs))

        def forward(
            model: GaussianMLP, x: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            return model(x)

        self._forward_ensemble = nnx.vmap(forward, in_axes=(0, None))
        self._forward_individual = nnx.vmap(forward, in_axes=(0, 0))

    @property
    def min_log_var(self):
        return constrained_param(self.raw_min_log_var.value, -20.0, 0.0)

    @property
    def max_log_var(self):
        return constrained_param(self.raw_max_log_var.value, -4.0, 5.0)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if x.ndim == 2:
            means, log_vars = self._forward_ensemble(self.ensemble, x)
        elif x.ndim == 3:
            means, log_vars = self._forward_individual(self.ensemble, x)
        else:
            raise ValueError(f"{x.shape=}")

        log_vars = self._safe_log_var(
            log_vars, self.min_log_var, self.max_log_var
        )

        return means, log_vars

    def aggregate(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Make predictions with ensemble and aggregate results.

        Parameters
        ----------
        x : array, shape (n_samples, n_features)
            Inputs.

        Returns
        -------
        mean : array, shape (n_samples, n_outputs)
            Prediction mean of the ensemble.

        var : array, shape (n_samples, n_outputs)
            Sum of aleatoric and epistemic variances. The aleatoric variance is
            the mean of the individual variances of the ensemble, and the
            epistemic variance is the variance of the individual means of the
            ensemble.
        """
        means, log_vars = self._forward_ensemble(self.ensemble, x)

        log_vars = self._safe_log_var(
            log_vars, self.min_log_var, self.max_log_var
        )

        mean = jnp.mean(means, axis=0)
        aleatoric_var = jnp.mean(jnp.exp(log_vars), axis=0)
        epistemic_var = jnp.var(means, axis=0)
        return mean, aleatoric_var + epistemic_var

    def base_predict(self, x, i):
        """Make prediction with individual model.

        Parameters
        ----------
        x : array, shape (n_samples, n_features)
            Inputs.

        i : int
            Model index.

        Returns
        -------
        mean : array, shape (n_samples, n_outputs)
            Predicted mean.

        var : array, shape (n_samples, n_outputs)
            Predicted variance.
        """
        graphdef, state = nnx.split(self.ensemble)
        state_i = jax.tree.map(lambda x: x[i], state)
        base_model = nnx.merge(graphdef, state_i)
        mean_i, log_var_i = base_model(x)
        log_var_i = self._safe_log_var(
            log_var_i, self.min_log_var, self.max_log_var
        )
        return mean_i, jnp.exp(log_var_i)

    def base_distribution(
        self, x: jnp.ndarray, i: int
    ) -> dist.MultivariateNormalDiag:
        """Sample from individual model of the ensemble.

        Parameters
        ----------
        x : array, shape (n_samples, n_features)
            Inputs.

        i : int
            Model index.

        Returns
        -------
        distribution : distrax.MultivariateNormalDiag
            Predicted distribution.
        """
        graphdef, state = nnx.split(self.ensemble)
        state_i = jax.tree.map(lambda x: x[i], state)
        base_model = nnx.merge(graphdef, state_i)
        mean_i, log_var_i = base_model(x)
        log_var_i = self._safe_log_var_i(
            log_var_i, self.min_log_var, self.max_log_var
        )
        std_i = jnp.exp(0.5 * log_var_i)
        return dist.MultivariateNormalDiag(loc=mean_i, scale_diag=std_i)


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


def bootstrap(
    n_ensemble: int, train_size: float, n_samples: int, key: jnp.ndarray
) -> jnp.ndarray:
    """Bootstrap training sets for ensemble.

    Parameters
    ----------
    n_ensemble
        Size of ensemble.

    train_size
        Fraction of training set size to be sampled for each model.

    n_samples
        Training set size.

    key
        Random sampling key.

    Returns
    -------
    indices : array, shape (n_ensemble, n_bootstrapped)
        Indices of
    """
    n_bootstrapped = int(train_size * n_samples)
    return jax.random.choice(
        key,
        n_samples,
        shape=(n_ensemble, n_bootstrapped),
        replace=True,
    )


def gaussian_ensemble_loss(
    model: GaussianMLPEnsemble,
    X: jnp.ndarray,
    Y: jnp.ndarray,
) -> jnp.ndarray:
    """Sum of Gaussian NLL and penalty for log_var boundaries."""
    mean, log_var = model(X)
    boundary_loss = model.max_log_var.sum() - model.min_log_var.sum()
    return gaussian_nll(mean, log_var, Y).sum() + 0.01 * boundary_loss


@nnx.jit
def train_epoch(
    model: GaussianMLPEnsemble,
    optimizer: nnx.Optimizer,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    indices: jnp.ndarray,
) -> jnp.ndarray:
    """Train ensemble for one epoch.

    Parameters
    ----------
    model
        Probabilistic ensemble.

    optimizer
        Optimizer of probabilistic ensemble.

    X : array, shape (n_samples, n_features)
        Feature matrix.

    Y : array, shape (n_samples, n_outputs)
        Target values.

    indices : array, shape (n_batches, n_ensemble, batch_size)
        Data indices for each batch and individual model.

    Returns
    -------
    loss : float
        Mean loss of batches during the epoch.
    """
    chex.assert_equal_shape_prefix((X, Y), prefix_len=1)
    chex.assert_axis_dimension(indices, axis=1, expected=model.n_ensemble)

    @nnx.scan(in_axes=(nnx.Carry, None, None, 0), out_axes=(nnx.Carry, 0))
    def batch_update(mod_opt, X, Y, batch):
        model, optimizer = mod_opt
        loss, grads = nnx.value_and_grad(gaussian_ensemble_loss, argnums=0)(
            model, X[batch], Y[batch]
        )
        optimizer.update(model, grads)
        return (model, optimizer), loss

    (model, optimizer), loss = batch_update((model, optimizer), X, Y, indices)

    return jnp.asarray(loss).mean()


class EnsembleTrainState(NamedTuple):
    model: GaussianMLPEnsemble
    optimizer: nnx.Optimizer
    train_size: float
    batch_size: int


def train_ensemble(
    model: GaussianMLPEnsemble,
    optimizer: nnx.Optimizer,
    train_size: float,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    n_epochs: int,
    batch_size: int,
    key: jnp.ndarray,
    verbose: int = 0,
) -> jnp.ndarray:
    """Train ensemble.

    Parameters
    ----------
    model
        Probabilistic ensemble.
    optimizer
        Optimization algorithm.
    X
        Feature matrix.
    Y
        Target values.
    n_epochs
        Number of epochs to train.
    batch_size
        Batch size.
    key
        For random number generation in bootstrapping to generate individual
        training set of each model and shuffling in each episode.
    verbose
        Verbosity level.

    Returns
    -------
    loss
        Measured in last epoch.
    """
    assert batch_size > 0
    assert n_epochs > 0

    key, bootstrap_key = jax.random.split(key, 2)
    n_samples = len(X)
    bootstrap_indices = bootstrap(
        model.n_ensemble, train_size, n_samples, bootstrap_key
    )

    loss = jnp.inf
    for t in range(1, n_epochs + 1):
        key, shuffle_key = jax.random.split(key, 2)
        shuffled_indices = jax.random.permutation(
            key, bootstrap_indices, axis=1
        )
        remaining = -(bootstrap_indices.shape[1] % batch_size)
        if remaining:
            shuffled_indices = shuffled_indices[:, :remaining]
        batched_indices = shuffled_indices.reshape(
            model.n_ensemble, batch_size, -1
        ).transpose([2, 0, 1])
        loss = train_epoch(
            model,
            optimizer,
            X,
            Y,
            batched_indices,
        )
        if verbose >= 1 and t % 100 == 0 or verbose >= 2 and t % 10 == 0:
            print(f"[train_ensemble] {t=}: {loss=}")

    return loss


def restore_checkpoint(path: str, model: nnx.Module) -> nnx.Module:
    """Restore checkpoint with orbax.

    Parameters
    ----------
    path : str
        Absolute path to directory in which the checkpoint should be stored.
    model : nnx.Module
        Model with graphdef that should be restored.

    Returns
    -------
    model : nnx.Module
        Restored model.
    """
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(path)
    graphdef, _ = nnx.split(model)
    return nnx.merge(graphdef, state)
