import jax.numpy as jnp
from flax import nnx


class ContinuousClippedDoubleQNet(nnx.Module):
    """Clipped Double Q network for continuous action spaces.

    Thin wrapper around two action-value networks. To avoid overestimation
    bias, we take the minimum of the prediction of the two networks.
    This is called clipped double Q-learning [1]_, which is based on double
    Q-learning [2]_. Note that clipped double Q-learning is not the same as
    deep double Q-learning [3]_, which is also based on the idea of double
    Q-learning.

    Parameters
    ----------
    q1 : nnx.Module
        Action-value network that maps a pair of state and action to the
        estimated expected return.

    q2 : nnx.Module
        Action-value network that maps a pair of state and action to the
        estimated expected return.

    References
    ----------
    .. [1] Fujimoto, S., Hoof, H., Meger, D. (2018). Addressing Function
       Approximation Error in Actor-Critic Methods. Proceedings of the 35th
       International Conference on Machine Learning, in Proceedings of Machine
       Learning Research 80:1587-1596 Available from
       https://proceedings.mlr.press/v80/fujimoto18a.html.

    .. [2] Hasselt, H. (2010). Double Q-learning. In Advances in Neural
       Information Processing Systems 23.
       https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html

    .. [3] Hasselt, H., Guez, A., Silver, D. (2016). Deep reinforcement
       learning with double Q-Learning. In Proceedings of the Thirtieth AAAI
       Conference on Artificial Intelligence (AAAI'16). AAAI Press, 2094–2100.
       https://arxiv.org/abs/1509.06461
    """

    q1: nnx.Module
    q2: nnx.Module

    def __init__(self, q1: nnx.Module, q2: nnx.Module):
        self.q1 = q1
        self.q2 = q2

    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        return jnp.minimum(self.q1(*args, **kwargs), self.q2(*args, **kwargs))

    def mean(self, *args, **kwargs) -> jnp.ndarray:
        """Predict mean of both Q networks."""
        return 0.5 * (self.q1(*args, **kwargs) + self.q2(*args, **kwargs))
