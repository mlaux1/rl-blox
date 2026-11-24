import gymnasium as gym

import rl_blox.algorithm.sarsa as sarsa
from rl_blox.algorithm.sarsa import train_sarsa
from rl_blox.blox.value_policy import make_q_table


def test_sarsa(tabular_test_env):
    q_table = make_q_table(tabular_test_env)

    _ = train_sarsa(
        tabular_test_env,
        q_table,
        total_timesteps=10,
        seed=0,
    )


class RecordActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = []
        self.episode_ends = []

    def step(self, action):
        result = self.env.step(action)
        self.actions.append(int(action))
        self.episode_ends.append(result[2] or result[3])
        return result


def test_sarsa_takes_the_next_action_of_its_target(monkeypatch):
    target_actions = []
    update_policy = sarsa._update_policy

    def record_update_policy(*args):
        target_actions.append(int(args[5]))
        return update_policy(*args)

    monkeypatch.setattr(sarsa, "_update_policy", record_update_policy)
    env = RecordActions(gym.make("FrozenLake-v1", is_slippery=False))

    train_sarsa(
        env,
        make_q_table(env),
        epsilon=0.5,
        total_timesteps=200,
        seed=0,
        progress_bar=False,
    )

    pairs = [
        (target_actions[t], env.actions[t + 1])
        for t in range(len(env.actions) - 1)
        if not env.episode_ends[t]
    ]
    assert len(pairs) > 100
    assert all(target == taken for target, taken in pairs)
