import jax.numpy as jnp
from flax import nnx


class ContinuousDoubleQNet(nnx.Module):
    """Double Q network for continuous action spaces.

    Thin wrapper around two action-value networks. To avoid overestimation
    bias, we take the minimum of the prediction of the two networks.
    This is the idea of Double Q-learning [1]_ applied to neural networks
    in Deep Double Q-learning [2]_.

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
    .. [1] Hasselt, H. (2010). Double Q-learning. In Advances in Neural
       Information Processing Systems 23.
       https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html

    .. [2] Hasselt, H., Guez, A., Silver, D. (2016). Deep reinforcement
       learning with double Q-Learning. In Proceedings of the Thirtieth AAAI
       Conference on Artificial Intelligence (AAAI'16). AAAI Press, 2094–2100.
       https://arxiv.org/abs/1509.06461
    """

    q1: nnx.Module
    q2: nnx.Module

    def __init__(self, q1: nnx.Module, q2: nnx.Module):
        self.q1 = q1
        self.q2 = q2

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.minimum(self.q1(x), self.q2(x))
