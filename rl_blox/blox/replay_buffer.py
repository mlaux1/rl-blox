from collections import OrderedDict, namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numpy import typing as npt


class ReplayBuffer:
    """Replay buffer that returns jax arrays.

    For each quantity, we store all samples in NumPy array that will be
    preallocated once the size of the quantities is know, that is, when the
    first transition sample is added. This makes sampling faster than when
    we use a deque.

    Parameters
    ----------
    buffer_size : int
        Maximum size of the buffer.

    keys : list[str], optional
        Names of the quantities that should be stored in the replay buffer.
        Defaults to ['observation', 'action', 'reward', 'next_observation',
        'termination']. These names have to be used as key word arguments when
        adding a sample. When sampling a batch, the arrays will be returned in
        this order.

    dtypes : list[dtype], optional
        dtype used for each buffer. Defaults to float for everything except
        'termination'.

    discrete_actions : bool, optional
        Changes the default dtype for actions to int.
    """

    buffer: OrderedDict[str, npt.NDArray[float]]
    Batch: type
    buffer_size: int
    current_len: int
    insert_idx: int

    def __init__(
        self,
        buffer_size: int,
        keys: list[str] | None = None,
        dtypes: list[npt.DTypeLike] | None = None,
        discrete_actions: bool = False,
    ):
        if keys is None:
            keys = [
                "observation",
                "action",
                "reward",
                "next_observation",
                "termination",
            ]
        if dtypes is None:
            dtypes = [
                float,
                int if discrete_actions else float,
                float,
                float,
                int,
            ]
        self.buffer = OrderedDict()
        for k, t in zip(keys, dtypes, strict=True):
            self.buffer[k] = np.empty(0, dtype=t)
        self.Batch = namedtuple("Batch", self.buffer)
        self.buffer_size = buffer_size
        self.current_len = 0
        self.insert_idx = 0

    def add_sample(self, **sample):
        """Add transition sample to the replay buffer.

        Note that the individual arguments have to be passed as keyword
        arguments with keys matching the ones passed to the constructor or
        the default keys respectively.
        """
        if self.current_len == 0:
            for k, v in sample.items():
                assert k in self.buffer, f"{k} not in {self.buffer.keys()}"
                self.buffer[k] = np.empty(
                    (self.buffer_size,) + np.asarray(v).shape,
                    dtype=self.buffer[k].dtype,
                )
        for k, v in sample.items():
            self.buffer[k][self.insert_idx] = v
        self.insert_idx = (self.insert_idx + 1) % self.buffer_size
        self.current_len = min(self.current_len + 1, self.buffer_size)

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[jnp.ndarray]:
        """Sample a batch of transitions.

        Note that the individual quantities will be returned in the same order
        as the keys were given to the constructor or the default order
        respectively.

        Parameters
        ----------
        batch_size : int
            Size of the sampled batch.

        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        batch : Batch
            Named tuple with order defined by keys. Content is also accessible
            via names, e.g., ``batch.observation``.
        """
        indices = rng.integers(0, self.current_len, batch_size)
        return self.Batch(
            **{k: jnp.asarray(self.buffer[k][indices]) for k in self.buffer}
        )

    def __len__(self):
        """Return current number of stored transitions in the replay buffer."""
        return self.current_len

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["Batch"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.Batch = namedtuple("Batch", self.buffer)


class LAP(ReplayBuffer):
    r"""Prioritized replay buffer.

    This replay buffer can be used for loss-adjusted PER (LAP) [1]_ and
    prioritized experience replay (PER) [2]_. PER is a sampling scheme for
    replay buffers, in which transitions are sampled in proportion to their
    temporal-difference (TD) error. The intuitive argument behind PER is that
    training on the highest error samples will result in the largest
    performance gain.

    PER changes the traditional uniformly sampled replay buffers. The
    probability of sampling a transition i is proportional to the absolute TD
    error :math:`|\delta_i|`, set to the power of a hyper-parameter
    :math:`\alpha` to smooth out extremes:

    .. math::

        p(i)
        =
        \frac{|\delta_i|^{\alpha} + \epsilon}
        {\sum_j |\delta_j|^{\alpha} + \epsilon},

    where a small constant :math:`\epsilon` is added to ensure each transition
    is sampled with non-zero probability. This is necessary as often the
    current TD error is approximated by the TD error when i was last sampled.

    LAP changes this to (:func:`lap_priority`)

    .. math::

        p(i)
        =
        \frac{\max(|\delta_i|^{\alpha}, 1)}
        {\sum_j \max(|\delta_j|^{\alpha}, 1)},

    which leads to uniform sampling of transitions with a TD error smaller than
    1 to avoid the bias introduced from using MSE and prioritization. A LAP
    replay buffer is supposed to be paired with a Huber loss with a threshold
    of 1 to switch between MSE and L1 loss.

    Parameters
    ----------
    buffer_size : int
        Maximum size of the buffer.

    keys : list[str], optional
        Names of the quantities that should be stored in the replay buffer.
        Defaults to ['observation', 'action', 'reward', 'next_observation',
        'termination']. These names have to be used as key word arguments when
        adding a sample. When sampling a batch, the arrays will be returned in
        this order.

    dtypes : list[dtype], optional
        dtype used for each buffer. Defaults to float for everything except
        'termination'.

    discrete_actions : bool, optional
        Changes the default dtype for actions to int.

    References
    ----------
    .. [1] Fujimoto, S., Meger, D., Precup, D. (2020). An Equivalence between
       Loss Functions and Non-Uniform Sampling in Experience Replay. In
       Advances in Neural Information Processing Systems 33.
       https://papers.nips.cc/paper/2020/hash/a3bf6e4db673b6449c2f7d13ee6ec9c0-Abstract.html

    .. [2] Schaul, T., Quan, J., Antonoglou, I., Silver, D. (2016). Prioritized
       Experience Replay. In International Conference on Learning
       Representations. https://arxiv.org/abs/1511.05952
    """

    insert_idx: int
    max_priority: float
    priority: npt.NDArray[float]
    sampled_indices: npt.NDArray[int]

    def __init__(
        self,
        buffer_size: int,
        keys: list[str] | None = None,
        dtypes: list[npt.DTypeLike] | None = None,
        discrete_actions: bool = False,
    ):
        super().__init__(buffer_size, keys, dtypes, discrete_actions)
        self.max_priority = 1.0
        self.priority = np.empty(buffer_size, dtype=float)
        self.sampled_indices = np.empty(0, dtype=int)

    def add_sample(self, **sample):
        """Add transition sample to the replay buffer.

        Note that the individual arguments have to be passed as keyword
        arguments with keys matching the ones passed to the constructor or
        the default keys respectively.
        """
        self.priority[self.insert_idx] = self.max_priority
        super().add_sample(**sample)

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> list[jnp.ndarray]:
        """Sample a batch of transitions.

        Note that the individual quantities will be returned in the same order
        as the keys were given to the constructor or the default order
        respectively.

        Parameters
        ----------
        batch_size : int
            Size of the sampled batch.

        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        batch : Batch
            Named tuple with order defined by keys. Content is also accessible
            via names, e.g., ``batch.observation``.
        """
        probabilities = np.cumsum(self.priority[: self.current_len])
        random_uniforms = rng.uniform(0, 1, size=batch_size) * probabilities[-1]
        self.sampled_indices = np.searchsorted(probabilities, random_uniforms)
        return self.Batch(
            **{
                k: jnp.asarray(self.buffer[k][self.sampled_indices])
                for k in self.buffer
            }
        )

    def update_priority(self, priority):
        self.priority[self.sampled_indices] = priority
        self.max_priority = max(np.max(priority), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = np.max(self.priority[: self.current_len])


@partial(jax.jit, static_argnames=["min_priority", "alpha"])
def lap_priority(
    abs_td_error: jnp.ndarray, min_priority: float, alpha: float
) -> jnp.ndarray:
    r"""Compute sample priority for loss-adjusted PER (LAP).

    Loss-adjusted prioritized experience replay (LAP) [1]_ is based on
    prioritized experience replay (PER) [2]_. LAP uses the priority

    .. math::

        p(i)
        =
        \frac{\max(|\delta_i|^{\alpha}, p_{\min})}
        {\sum_j \max(|\delta_j|^{\alpha}, p_{\min})},

    which leads to uniform sampling of transitions with a TD error smaller than
    :math:`p_{\min}` (usually 1) to avoid the bias introduced from using MSE
    and prioritization. A LAP replay buffer is supposed to be paired with a
    Huber loss with a threshold of :math:`p_{\min}` to switch between MSE and
    L1 loss.

    Parameters
    ----------
    abs_td_error : array
        A batch of :math:`|\delta_i|`.

    min_priority : float
        Minimum priority :math:`p_{\min}`.

    alpha : float
        Smoothing exponent :math:`\alpha`.

    Returns
    -------
    p : array
        A batch of priorities :math:`p(i)`.

    References
    ----------
    .. [1] Fujimoto, S., Meger, D., Precup, D. (2020). An Equivalence between
       Loss Functions and Non-Uniform Sampling in Experience Replay. In
       Advances in Neural Information Processing Systems 33.
       https://papers.nips.cc/paper/2020/hash/a3bf6e4db673b6449c2f7d13ee6ec9c0-Abstract.html

    .. [2] Schaul, T., Quan, J., Antonoglou, I., Silver, D. (2016). Prioritized
       Experience Replay. In International Conference on Learning
       Representations. https://arxiv.org/abs/1511.05952
    """
    return jnp.maximum(abs_td_error, min_priority) ** alpha
