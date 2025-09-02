import jax.numpy as jnp
import jax.random
import optax
from flax import nnx

from ..function_approximator.norm import avg_l1_norm


class SALE(nnx.Module):
    r"""SALE: state-action learned embedding.

    SALE was introduced with TD7 [1]_.

    The objective of SALE is to learn embeddings :math:`z^{sa}, z^s` that
    capture relevant structure in the observation space, as well as the
    transition dynamics of the environment. SALE defines a pair of encoders
    :math:`(f, g)`:

    .. math::

        z^s := f(s), \quad z^{sa} := g(z^s, a).

    The embeddings are split into state and state-action components so that the
    encoders can be trained with a dynamics prediction loss
    (see :func:`state_action_embedding_loss`) that solely relies on the next
    state :math:`s'`, independent of the next action or current policy.

    .. warning::

        Although the embeddings are learned by considering the dynamics of the
        environment, their only purpose is to improve the input to the value
        function :class:`CriticSALE` and policy :class:`ActorSALE`, and not to
        serve as a world model for predicting state transitions.

    Parameters
    ----------
    state_embedding : nnx.Module
        State embedding network **without** AvgL1Norm. AvgL1Norm will be added
        in this module to form :math:`z^s = f(s)`. This networks maps state to
        **unnormalized** zs.

    state_action_embedding : nnx.Module
        :math:`z^{sa} = g(z^s, a)`. State action embedding. Maps zs and action
        to zsa, which is trained to be the same as the normalized zs of the
        next state.

    See Also
    --------
    CriticSALE
        Action-value function that uses SALE as input.
    ActorSALE
        Policy that uses SALE as input.
    state_action_embedding_loss
        Loss to train SALE.

    References
    ----------
    .. [1] Fujimoto, S., Chang, W.D., Smith, E., Gu, S., Precup, D., Meger, D.
       (2023). For SALE: State-Action Representation Learning for Deep
       Reinforcement Learning. In Advances in Neural Information Processing
       Systems 36, pp. 61573-61624. Available from
       https://proceedings.neurips.cc/paper_files/paper/2023/hash/c20ac0df6c213db6d3a930fe9c7296c8-Abstract-Conference.html
    """

    _state_embedding: nnx.Module
    state_action_embedding: nnx.Module
    """:math:`z^{sa} = g(z^s, a)`."""

    def __init__(
        self, state_embedding: nnx.Module, state_action_embedding: nnx.Module
    ):
        self._state_embedding = state_embedding
        self.state_action_embedding = state_action_embedding

    def __call__(
        self, state: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        zs = self.state_embedding(state)
        zs_action = jnp.concatenate((zs, action), axis=-1)
        zsa = self.state_action_embedding(zs_action)
        return zsa, zs

    def state_embedding(self, state: jnp.ndarray) -> jnp.ndarray:
        r""":math:`z^s = f(s)`."""
        return avg_l1_norm(self._state_embedding(state))


class ActorSALE(nnx.Module):
    r"""Deterministic policy with SALE.

    The actor maps a state vector through a linear leayer and applied AvgL1Norm,
    it concatenates the result with the embedded vector zs, and then maps it
    through a deterministic policy net to the action.

    This module implements the function :math:`\pi(s, z^s_t)`.

    Parameters
    ----------
    policy_net : nnx.Module
        Deterministic policy network.

    n_state_features : int
        Number of state components.

    hidden_nodes : int
        Number of nodes in the first layer that encodes state and applies
        :func:`~.blox.function_approximator.norm.avg_l1_norm`.

    rngs : nnx.Rngs
        Random number generator.
    """

    policy_net: nnx.Module
    l0: nnx.Linear

    def __init__(
        self,
        policy_net: nnx.Module,
        n_state_features: int,
        hidden_nodes: int,
        rngs: nnx.Rngs,
    ):
        self.policy_net = policy_net
        self.l0 = nnx.Linear(n_state_features, hidden_nodes, rngs=rngs)

    def __call__(self, state: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        """pi(state, zs)."""
        h = avg_l1_norm(self.l0(state))
        he = jnp.concatenate((h, zs), axis=-1)
        return self.policy_net(he)


class DeterministicSALEPolicy(nnx.Module):
    """Combines SALE encoder and ActorSALE to form a deterministic policy."""

    def __init__(self, embedding: SALE, actor: ActorSALE):
        self.embedding = embedding
        self.actor = actor

    def __call__(self, observation: jnp.ndarray):
        return self.actor(
            observation, self.embedding.state_embedding(observation)
        )


class CriticSALE(nnx.Module):
    r"""Action-value function Q with SALE.

    The critic maps a concatenated state-action vector through a linear layer
    and applies AvgL1Norm, it concatenates the result with the embedded vectors
    zs and zsa of the state-action pair, and then maps it through the q_net
    to the expected value of the state-action pair.

    This module implements the function :math:`Q(s, a, z^{sa}_t z^s_t)`.

    Parameters
    ----------
    q_net : nnx.Module
        Action-value function.

    n_state_features : int
        Number of state components.

    n_action_features : int
        Number of action components.

    hidden_nodes : int
        Number of nodes in the first layer that encodes state and action and
        applies :func:`~.blox.function_approximator.norm.avg_l1_norm`.

    rngs : nnx.Rngs
        Random number generator.
    """

    q_net: nnx.Module
    q0: nnx.Linear

    def __init__(
        self,
        q_net: nnx.Module,
        n_state_features: int,
        n_action_features: int,
        hidden_nodes: int,
        rngs: nnx.Rngs,
    ):
        self.q_net = q_net
        self.q0 = nnx.Linear(
            n_state_features + n_action_features, hidden_nodes, rngs=rngs
        )

    def __call__(
        self, sa: jnp.ndarray, zsa: jnp.ndarray, zs: jnp.ndarray
    ) -> jnp.ndarray:
        """Q(s, a, zsa, zs)."""
        h = avg_l1_norm(self.q0(sa))
        embeddings = jnp.concatenate((zsa, zs), axis=-1)
        he = jnp.concatenate((h, embeddings), axis=-1)
        return self.q_net(he)


def state_action_embedding_loss(
    embedding: SALE,
    observation: jnp.ndarray,
    action: jnp.ndarray,
    next_observation: jnp.ndarray,
) -> float:
    r"""Loss of state-action embedding.

    The encoders are jointly trained using the mean squared error (MSE) between
    the state-action embedding :math:`z^{sa}` and the embedding of the next
    state :math:`z^{s'}`:

    .. math::

        \mathcal{L} = \frac{1}{N} \sum_i \left(
        z^{s_i,a_i} - \texttt{sg}(z^{s_i'}) \right)^2,

    where :math:`\texttt{sg}(\cdot)` denotes the stop-gradient operation.

    The embeddings are designed to model the underlying structure of the
    environment. However, they may not encompass all relevant information
    needed by the value function and policy, such as features related to the
    reward, current policy, or task horizon.

    Parameters
    ----------
    embedding : SALE
        State-action learned embedding.

    observation : array, shape (batch_size,) + observation_space.shape
        Observations.

    action : array, shape (batch_size,) + action.shape
        Action.

    next_observation : array, shape (batch_size,) + observation_space.shape
        Next observations.

    Returns
    -------
    loss : float
        Loss value.
    """
    zsa, _ = embedding(observation, action)
    zsp = jax.lax.stop_gradient(embedding.state_embedding(next_observation))
    return optax.squared_error(predictions=zsa, targets=zsp).mean()


@nnx.jit
def update_sale(
    embedding: SALE,
    embedding_optimizer: nnx.Optimizer,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
) -> float:
    """Update SALE.

    Parameters
    ----------
    embedding : SALE
        State-action learned embedding.

    embedding_optimizer : nnx.Optimizer
        Optimizer for embedding.

    observations : array
        Batch of observations.

    actions : array
        Batch of actions.

    next_observations : array
        Batch of next observations.

    Returns
    -------
    embedding_loss_value : float
        Loss value.
    """
    embedding_loss_value, grads = nnx.value_and_grad(
        state_action_embedding_loss, argnums=0
    )(embedding, observations, actions, next_observations)
    embedding_optimizer.update(embedding, grads)
    return embedding_loss_value
