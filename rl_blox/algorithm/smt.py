def train_smt(mt_env):
    """Scheduled Multi-Task (SMT) training.

    Multi-task RL faces the challenge of varying task difficulties, often
    leading to negative transfer when simpler tasks overshadow the learning of
    more complex ones. To overcome this challenge, SMT strategically prioritizes
    more challenging tasks, thereby enhancing overall learning efficiency. SMT
    uses a dynamic task prioritization strategy, underpinned by an effective
    metric for assessing task difficulty. This metric ensures an efficient and
    targeted allocation of training resources.

    References
    ----------
    .. [1] Cho, M., Park, J., Lee, S., Sung, Y. (2024). Hard tasks first:
       multi-task reinforcement learning through task scheduling. In
       Proceedings of the 41st International Conference on Machine Learning,
       Vol. 235. JMLR.org, Article 340, 8556–8577.
       https://icml.cc/virtual/2024/poster/33388
    """
