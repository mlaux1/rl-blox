from collections.abc import Callable
from functools import partial

import chex
import distrax
import flax.core
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax.typing import ArrayLike


class GaussianMlp(nn.Module):
    """Probabilistic neural network that predicts a Gaussian distribution."""

    shared_head: bool
    """All nodes of the last hidden layer are connected to mean AND log_std."""
    n_outputs: int
    """Number of output components"""
    hidden_nodes: list[int]
    """Numbers of hidden nodes of the MLP."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        for n_nodes in self.hidden_nodes:
            x = nn.Dense(n_nodes)(x)
            x = nn.swish(x)
        x = nn.Dense(2 * self.n_outputs)(x)
        if self.shared_head:
            mean, log_var = jnp.split(x, (self.n_outputs,), axis=-1)
        else:
            mean = nn.Dense(self.n_outputs)(x)
            log_var = nn.Dense(self.n_outputs)(x)
        return mean, log_var


class EnsembleOfGaussianMlps:
    """Ensemble of Gaussian MLPs.

    Parameters
    ----------
    base_model
        Base learner.
    n_base_models
        Number of individual Gaussian MLPs.
    train_size, optional
        Fraction of the original training set to train each individual
        Gaussian MLP.
    warm_start, optional
        Reuse old weights when training.
    learning_rate, optional
        Learning rate for ADAM optimizer.
    key
        jax random key.
    verbose, optional
        Verbosity level.

    References
    ----------
    .. [1] Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine.
           2018. Deep reinforcement learning in a handful of trials using
           probabilistic dynamics models. In Proceedings of the 32nd
           International Conference on Neural Information Processing Systems
           (NeurIPS'18). Curran Associates Inc., Red Hook, NY, USA, 4759–4770.
           https://papers.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html
    """

    base_model: GaussianMlp
    n_base_models: int
    train_size: float
    warm_start: bool
    learning_rate: float
    key: jnp.ndarray
    verbose: int
    _base_model_predict: Callable
    _ensemble_predict: Callable
    train_states_: TrainState

    def __init__(
        self,
        base_model: GaussianMlp,
        n_base_models: int,
        train_size: float,
        warm_start: bool,
        learning_rate: float,
        key: jnp.ndarray,
        verbose: int = 0,
    ):
        self.base_model = base_model
        self.n_base_models = n_base_models
        self.train_size = train_size
        self.warm_start = warm_start
        self.learning_rate = learning_rate
        self.key = key
        self.verbose = verbose

        def base_model_predict(train_state, X):
            return self.base_model.apply(train_state.params, X)

        self._base_model_predict = jax.jit(base_model_predict)
        self._ensemble_predict = jax.jit(
            jax.vmap(base_model_predict, in_axes=(0, None))
        )
        self.train_states_ = None

    @classmethod
    def create(
        cls,
        n_outputs: int,
        hidden_nodes: list[int],
        n_base_models: int,
        key: jnp.ndarray,
        shared_head: bool = True,
        train_size: float = 0.7,
        warm_start: bool = True,
        learning_rate: float = 3e-3,
        verbose: int = 0,
    ) -> "EnsembleOfGaussianMlps":
        """Create ensemble of Gaussian MLPs.

        Parameters
        ----------
        n_outputs
            Number of outputs of one individual Gaussian MLP.
        hidden_nodes
            Number of hidden nodes in each individual Gaussian MLP.
        n_base_models
            Number of individual Gaussian MLPs.
        key
            jax random key.
        shared_head, optional
            All nodes of the last hidden layer are connected to mean AND
            log_var.
        train_size, optional
            Fraction of the original training set to train each individual
            Gaussian MLP.
        warm_start, optional
            Reuse old weights when training.
        learning_rate, optional
            Learning rate for ADAM optimizer.
        verbose, optional
            Verbosity level.

        Returns
        -------
        ensemble
            Ensemble of Gaussian MLPs.
        """
        return cls(
            base_model=GaussianMlp(
                shared_head=shared_head,
                n_outputs=n_outputs,
                hidden_nodes=hidden_nodes,
            ),
            n_base_models=n_base_models,
            train_size=train_size,
            warm_start=warm_start,
            learning_rate=learning_rate,
            key=key,
            verbose=verbose,
        )

    def fit(
        self, X: ArrayLike, Y: ArrayLike, n_epochs: int
    ) -> "EnsembleOfGaussianMlps":
        """Train model on dataset.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Each row contains a feature vector.
        Y : array-like, shape (n_samples, n_outputs)
            Each row contains a desired output.
        n_epochs : int
            Training epochs, i.e., sweeps over the whole dataset.

        Returns
        -------
        self
            For chaining.
        """
        X = jnp.asarray(X)
        Y = jnp.asarray(Y)

        chex.assert_equal_shape_prefix((X, Y), prefix_len=1)
        chex.assert_axis_dimension(
            Y, axis=1, expected=self.base_model.n_outputs
        )

        n_samples = X.shape[0]
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

        if self.train_states_ is None or not self.warm_start:
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

        def update_base_model(
            train_state: TrainState, X: jnp.ndarray, y: jnp.ndarray
        ) -> tuple[jnp.ndarray, TrainState]:
            def compute_loss(
                X: jnp.ndarray, Y: jnp.ndarray, params: flax.core.FrozenDict
            ) -> jnp.ndarray:
                mean_pred, log_var_pred = self.base_model.apply(params, X)
                return heteroscedastic_aleatoric_uncertainty_loss(
                    mean_pred, log_var_pred, Y
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

        return self

    def predict(self, X: ArrayLike) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Predict outputs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Each row contains a feature vector.

        Returns
        -------
        Y : array, shape (n_samples, n_outputs)
            Each row contains a prediction.

        var : array, shape (n_samples, n_outputs)
            Each row contains the sum of aleatoric and epistemic variance.
        """
        assert self.train_states_ is not None
        X = jnp.asarray(X)
        means, log_vars = self._ensemble_predict(self.train_states_, X)
        return gaussian_ensemble_prediction(means, log_vars)

    def base_predict(
        self, X: ArrayLike, i: int
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Predict with one of the base models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Each row contains a feature vector.

        i
            Index of base model.

        Returns
        -------
        Y : array, shape (n_samples, n_outputs)
            Each row contains a prediction.

        var : array, shape (n_samples, n_outputs)
            Each row contains the aleatoric variance.
        """
        assert self.train_states_ is not None
        X = jnp.asarray(X)
        train_state = self._get_train_state(i)
        return self._base_model_predict(train_state, X)

    def base_sample(self, x, i, key):
        assert self.train_states_ is not None
        train_state = self._get_train_state(i)
        mean, log_var = self._base_model_predict(train_state, x[jnp.newaxis])
        y = distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.exp(0.5 * log_var)
        ).sample(seed=key)
        return y[0]

    def _get_train_state(self, i: ArrayLike) -> TrainState:
        """Get train state of individual Gaussian MLP(s)."""
        return jax.tree.map(lambda x: x[i], self.train_states_)


def exp_log_var(
    log_var: jnp.ndarray, min_log_var: jnp.ndarray, max_log_var: jnp.ndarray
) -> jnp.ndarray:
    """Transform logarithm of the variance to variance numerically stable.

    The approach is described by Chua et al. [1]_ in Appendix A.1:

    An under-appreciated detail of probabilistic networks is how the variance
    output is implemented with automatic differentiation. Often the real-valued
    output is treated as a log variance (or similar), and transformed through
    an exponential function (or similar) to produce a nonnegative-valued
    output, necessary to be interpreted as a variance. However, whilst this
    variance output is well-behaved at points within the training distribution,
    its value is undefined outside the trained distribution. In fact, during
    the training, there is no explicit loss term that regulate the behavior of
    the variance outside of the training points. Thus, when this model is then
    evaluated at previously unseen states, as is often the case during the MBRL
    learning process, the outputted variance can assume any arbitrary value,
    and in practice we noticed how it occasionally collapse to zero, or explode
    toward infinity.
    This behavior is in contrast with other models, such as GPs, where the
    variance is more well behaving, being bounded and Lipschitz-smooth. As a
    remedy, we found that in our model lower bounding and upper bounding the
    output variance such that they could not be lower or higher than the lowest
    and highest values in the training data significantly helped. To bound the
    variance output for a probabilistic network to be between the upper and
    lower bounds found during training the network on the training data,
    we used the following code with automatic differentiation:
    `[read the code]`
    with a small regularization penalty on term on max_logvar so that it does
    not grow beyond the training distribution’s maximum output variance, and on
    the negative of min_logvar so that it does not drop below the training
    distribution’s minimum output variance.

    The implementation is available at
    https://github.com/kchua/handful-of-trials/blob/master/dmbrl/modeling/models/BNN.py#L392

    Parameters
    ----------
    log_var : array, shape (n_samples, n_outputs)
        Logarithm of variance, predicted by neural network.

    min_log_var : array, shape (n_outputs,)
        Lower bound.

    max_log_var : array, shape (n_outputs,)
        Upper bound.

    Returns
    -------
    var
        Variance.

    References
    ----------
    .. [1] Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine.
           2018. Deep reinforcement learning in a handful of trials using
           probabilistic dynamics models. In Proceedings of the 32nd
           International Conference on Neural Information Processing Systems
           (NeurIPS'18). Curran Associates Inc., Red Hook, NY, USA, 4759–4770.
           https://papers.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html
    """
    log_var = max_log_var[jnp.newaxis] - jax.nn.softplus(
        max_log_var[jnp.newaxis], log_var
    )
    log_var = min_log_var[jnp.newaxis] + jax.nn.softplus(
        log_var - min_log_var[jnp.newaxis]
    )
    return jnp.exp(log_var)


@jax.jit
def gaussian_ensemble_prediction(
    means: jnp.ndarray, log_vars: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute output of ensemble from outputs of individual Gaussian MLPs.

    Parameters
    ----------
    means : array, shape (n_base_models, n_samples, n_outputs)
        Predicted means of each base model, sample, and output dimension.

    log_vars : array, shape (n_base_models, n_samples, n_outputs)
        Predicted logarithm of variance of each base model, sample,
        and output dimension.

    Returns
    -------
    Y : array, shape (n_samples, n_outputs)
        Each row contains a prediction.

    var : array, shape (n_samples, n_outputs)
        Each row contains the sum of aleatoric and epistemic variance.
    """
    n_base_models = len(means)
    mean = jnp.mean(means, axis=0)
    epistemic_var = jnp.sum((means - mean) ** 2, axis=0) / (n_base_models + 1)
    aleatoric_var = jnp.mean(jnp.exp(log_vars), axis=0)
    return mean, aleatoric_var + epistemic_var


@jax.jit
def heteroscedastic_aleatoric_uncertainty_loss(
    mean_pred: jnp.ndarray, log_var_pred: jnp.ndarray, Y: jnp.ndarray
) -> jnp.ndarray:
    """Heteroscedastic aleatoric uncertainty loss for Gaussian NN.

    .. math::

        -\log p_{\theta}(Y|X) = \frac{1}{N}\sum_n
        \frac{1}{2} \frac{(y_n - \mu_{\theta}(x_n))^2}{\sigma_{\theta}^2(x)}
        + \frac{1}{2} \log \sigma_{\theta}^2(x_n) + \text{constant}

    This is the negative log-likelihood of Gaussian distributions predicted
    by a neural network, i.e., the neural network predicted a mean vector and
    a vector of component-wise log variance per sample.

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

    var = jnp.exp(log_var_pred)
    # var = jnp.where(var < 1e-6, 1.0, var)  # TODO do we need this?
    squared_erros = optax.l2_loss(mean_pred, Y)  # including factor 0.5
    return jnp.mean(squared_erros / var) + 0.5 * jnp.mean(log_var_pred)
