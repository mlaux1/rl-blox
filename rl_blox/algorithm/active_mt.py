import numpy as np

from ..blox.mapb import DUCB


class TaskSelector:
    def __init__(self, targets):
        self.targets = targets
        self.waiting_for_reward = False

    def select(self):
        assert (
            not self.waiting_for_reward
        ), "You have to provide a reward for the last target"
        self.waiting_for_reward = True

    def feedback(self, reward):
        assert self.waiting_for_reward, "Cannot assign reward to any target"
        self.waiting_for_reward = False


class DUCBGeneralized(TaskSelector):
    def __init__(
        self,
        targets,
        upper_bound,
        ducb_gamma,
        zeta,
        baseline,
        op,
        verbose=False,
        **kwargs,
    ):
        super().__init__(targets)
        self.baseline = baseline
        self.op = op
        self.verbose = verbose
        self.heuristic_params = kwargs

        self.n_contexts = targets.shape[0]
        self.ducb = DUCB(
            n_arms=self.n_contexts,
            upper_bound=upper_bound,
            gamma=ducb_gamma,
            zeta=zeta,
        )
        self.last_rewards = [[] for _ in range(self.n_contexts)]
        self.chosen_arm = -1

    def select(self):
        super().select()
        self.chosen_arm = self.ducb.choose_arm()
        return self.targets[self.chosen_arm]

    def feedback(self, reward):
        last_rewards = np.array(self.last_rewards[self.chosen_arm])[::-1]

        if len(last_rewards) == 0:
            self.ducb.chosen_arms = self.ducb.chosen_arms[:-1]
        else:
            if self.baseline == "max":
                b = np.max(last_rewards)
            elif self.baseline == "avg":
                b = np.mean(last_rewards)
            elif self.baseline == "davg":
                gamma = self.heuristic_params["heuristic_gamma"]
                b = np.sum(
                    last_rewards * gamma ** np.arange(1, len(last_rewards) + 1)
                ) * (1.0 / gamma - 1.0)
            elif self.baseline == "last":
                b = last_rewards[0]
            else:
                b = 0.0
            intrinsic_reward = reward - b
            if self.op == "max-with-0":
                intrinsic_reward = np.maximum(0.0, intrinsic_reward)
            elif self.op == "abs":
                intrinsic_reward = np.abs(intrinsic_reward)
            elif self.op == "neg":
                intrinsic_reward *= -1
            self.ducb.reward(intrinsic_reward)

        super().feedback(reward)

        self.last_rewards[self.chosen_arm].append(reward)


class RoundRobinSelector(TaskSelector):
    def __init__(self, targets, **kwargs):
        super().__init__(targets)
        self.i = 0

    def select(self):
        super().select()
        self.i += 1
        return self.targets[self.i % len(self.targets)]

    def feedback(self, reward):
        super().feedback(reward)


class SAGG_RIAC(TaskSelector):
    def __init__(
        self,
        _,
        x_train_range,
        y_train_range,
        g_max,
        max_close_dist,
        interest_window_size,
        n_split_samples,
        verbose=0,
        **kwargs,
    ):
        super().__init__(None)
        # Regions will be split after g_max goals have been attempted inside
        self.g_max = g_max
        # Samples that are "close" to another goal should be within this radius
        self.max_close_dist = max_close_dist
        # Window size for the interest value
        self.interest_window_size = interest_window_size
        # Number of splits that will be sampled and tested
        self.n_split_samples = n_split_samples
        # Verbosity level
        self.verbose = verbose

        if y_train_range is None:
            space = np.array(x_train_range).astype(float)[:, np.newaxis]
        else:
            space = np.array([x_train_range, y_train_range]).T.astype(float)
        self.space = np.sort(space, axis=0)
        # Regions are defined by their upper and lower bound
        self.regions = [self.space]
        # All goals that have been visited in each region
        self.goals = [[]]
        # All rewards that have been obtained in each region
        self.rewards = [[]]
        # The interest values of each region
        self.interests = [1.0]
        # The region has been split before
        self.split = [False]

        # Temporary variables
        self.goal = None
        if y_train_range is None:
            self.n_context_dims = 1
        else:
            self.n_context_dims = 2

    def select(self):
        super().select()

        if len(self.regions) == 1:
            # Initial phase: random explorations of the whole space
            region = self.regions[0]
            self.goal = np.random.uniform(region[0], region[1])
        else:
            p = np.random.rand()
            if p < 0.7:
                # Mode 1: a random goal is chosen along a uniform distribution
                # inside a region which is selected with a probability
                # proportional to its interest value
                P = self._interests_to_probs()
                region_idx = np.random.choice(len(P), p=P)
                region = self.regions[region_idx]
                self.goal = np.random.uniform(region[0], region[1])
            elif p < 0.7 + 0.2:
                # Mode 2: a random goal is chosen inside the whole space
                self.region_idx = 0
                region = self.regions[0]
                self.goal = np.random.uniform(region[0], region[1])
            else:
                # Mode 3: a region is first selected according to the interest
                # value and then a new goal is generated close to the already
                # experimented one which received the lowest competence
                # estimation
                P = self._interests_to_probs()
                region_idx = np.random.choice(len(P), p=P)
                region = self.regions[region_idx]
                goal_idx = np.argmin(self.rewards[region_idx])
                goal = self.goals[region_idx][goal_idx]
                self.goal = self._sample_near(goal)

        return self.goal

    def _interests_to_probs(self):
        interests = np.array(self.interests)
        P = interests - np.min(interests)
        P_sum = np.sum(P)
        if P_sum == 0:
            return np.ones(len(self.interests)) / len(self.interests)
        else:
            return P / P_sum

    def _sample_near(self, goal):
        goal = np.asarray(goal)
        lo = goal - self.max_close_dist
        hi = goal + self.max_close_dist
        sample = np.random.uniform(lo, hi)
        while np.linalg.norm(sample - goal) > self.max_close_dist:
            sample = np.random.uniform(lo, hi)
            if self.verbose >= 1:
                print(f"Resampling, goal {sample!r} out of bounds!")
        return sample

    def feedback(self, reward):
        super().feedback(reward)

        check_split_indices = []

        # Save goal and reward for all regions that contain the goal
        for region_idx in range(len(self.regions)):
            if self._in_region(self.goal, self.regions[region_idx]):
                self.goals[region_idx].append(self.goal)
                self.rewards[region_idx].append(reward)
                self._update_interest(region_idx)
                check_split_indices.append(region_idx)

        for region_idx in check_split_indices:
            n_goals = len(self.goals[region_idx])
            if not self.split[region_idx] and n_goals >= self.g_max:
                self.split[region_idx] = True
                region = self.regions[region_idx]

                if self.verbose >= 1:
                    print(f"Splitting region #{region_idx}")
                if self.verbose >= 2:
                    print("From")
                    print(region)

                split = None
                quality = -np.inf
                for _ in range(self.n_split_samples):
                    j = np.random.choice(self.n_context_dims)
                    vj = np.random.uniform(region[0, j], region[1, j])

                    R1 = region.copy()
                    R1[1, j] = vj
                    R2 = region.copy()
                    R2[0, j] = vj

                    r1_indices = np.where(
                        [
                            self._in_region(self.goals[region_idx][i], R1)
                            for i in range(n_goals)
                        ]
                    )[0]
                    r2_indices = np.where(
                        [
                            self._in_region(self.goals[region_idx][i], R2)
                            for i in range(n_goals)
                        ]
                    )[0]

                    rewards = np.asarray(self.rewards[region_idx])
                    rewards1 = rewards[r1_indices]
                    interest1 = self._interest(rewards1)
                    rewards2 = rewards[r2_indices]
                    interest2 = self._interest(rewards2)

                    q = (
                        len(rewards1)
                        * len(rewards2)
                        * np.abs(interest1 - interest2)
                    )
                    if q > quality:
                        goals = np.asarray(self.goals[region_idx])
                        goals1 = goals[r1_indices]
                        goals2 = goals[r2_indices]
                        q = quality
                        split = (
                            (R1, rewards1, goals1, interest1),
                            (R2, rewards2, goals2, interest2),
                        )

                if self.verbose >= 2:
                    print("To")
                for region, rewards, goals, interest in split:
                    if self.verbose >= 2:
                        print(region)
                    self.regions.append(region)
                    self.rewards.append(rewards.tolist())
                    self.goals.append(goals.tolist())
                    self.interests.append(interest)
                    self.split.append(False)

    def _interest(self, rewards):
        start_idx = -np.min((self.interest_window_size, len(rewards)))
        separation_idx = start_idx // 2
        return np.abs(
            (
                np.sum(rewards[start_idx:separation_idx])
                - np.sum(rewards[separation_idx:])
            )
            / self.interest_window_size
        )

    def _update_interest(self, region_idx):
        rewards = self.rewards[region_idx]
        interest = self._interest(rewards)
        self.interests[region_idx] = interest
        if self.verbose >= 1:
            print(f"Updated interest of region #{region_idx} to {interest:.2f}")

    def _in_region(self, context, region):
        context = np.asarray(context)
        inside = np.all(context > region[0]) and np.all(context < region[1])
        return inside
