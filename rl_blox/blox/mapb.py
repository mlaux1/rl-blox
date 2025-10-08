"""Tools for solving non-stationary multi-armed bandits."""

import numpy as np


class DUCB:
    """Discounted Upper Confidence Bound.

    Parameters
    ----------
    n_arms: int
        number of arms
    upper_bound: float
        upper boundary of reward
    gamma: float
        discounting factor
    zeta: float
        some constant that has to be set appropriately (> 0)

    References
    ----------
    .. [1] Kocsis, L., Szepesvari, C. (2006). Discounted UCB. In 2nd PASCAL
       Challenges Workshop, 2006.

    .. [2] Garivier, A., Moulines, E. (2008). On Upper-Confidence Bound
       Policies for Non-Stationary Bandit Problems.
       http://arxiv.org/abs/0805.3415
    """

    def __init__(self, n_arms, upper_bound, gamma, zeta=0.002, verbose=0):
        self.n_arms = n_arms
        self.upper_bound = upper_bound
        self.gamma = gamma
        self.zeta = zeta
        self.verbose = verbose
        self.chosen_arms = []
        self.rewards = []
        self.discounted_frequencies = np.zeros(self.n_arms)
        self.total_frequency = 0.0
        # TODO only keep the last ceil(log(threshold)/log(gamma)) episodes,
        # where threshold is the minimum weight of an example, e.g. 1e-5

    def _discounted_empirical_mean(self, arm_idx):
        t = len(self.chosen_arms)
        discounted_rewards = np.sum(
            [
                self.gamma ** (t - 1 - s) * self.rewards[s]
                for s in range(max(0, t - 250), t)
                if self.chosen_arms[s] == arm_idx
            ]
        )
        return discounted_rewards / self.discounted_frequencies[arm_idx]

    def _padding_function(self, arm_idx):
        return (
            2
            * self.upper_bound
            * np.sqrt(
                self.zeta
                * np.log(self.total_frequency)
                / self.discounted_frequencies[arm_idx]
            )
        )

    def _episode_finished(self):
        self.discounted_frequencies[:] = 0.0
        t = len(self.chosen_arms)
        for s in range(max(0, t - 250), t):
            self.discounted_frequencies[self.chosen_arms[s]] += self.gamma ** (
                t - 1 - s
            )
        self.total_frequency = np.sum(self.discounted_frequencies)

    def choose_arm(self):
        if len(self.rewards) < 2 * self.n_arms:
            arm_idx = len(self.rewards) % self.n_arms
        else:
            mean = np.array(
                [
                    self._discounted_empirical_mean(arm_idx)
                    for arm_idx in range(self.n_arms)
                ]
            )
            padding = np.array(
                [
                    self._padding_function(arm_idx)
                    for arm_idx in range(self.n_arms)
                ]
            )
            ducb = mean + padding
            arm_idx = np.argmax(ducb)

            if self.verbose:
                print(f"Rewards: {len(self.rewards)}")
                print(f"Means:   {np.round(mean, 1)}")
                print(f"Padding: {np.round(padding, 1)}")
                print(f"D-UCB:   {np.round(ducb, 1)}")
                print(f"Arm:     {arm_idx}")
        self.chosen_arms.append(arm_idx)
        return arm_idx

    def reward(self, r):
        self.rewards.append(r)
        self._episode_finished()
