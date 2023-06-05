import gymnasium as gym
import numpy as np

from collections import defaultdict
from modular_rl.policy.tabular_model import TabularModel

np.set_printoptions(precision=4)

train_env = gym.make("FrozenLake-v1", render_mode="human", desc=["SFFH", "FFFF", "FFFF", "FFFG"])

#observation, _ = env.reset()

visits = TabularModel(train_env, init_value=0)

value = TabularModel(train_env, init_value=0)

return_dict = defaultdict(list)

print(return_dict)


def generate_rollout(env, policy):

    observation, _ = env.reset()
    terminated = False
    truncated = False

    obs = []
    acts = []
    rews = []

    obs.append(observation)

    while not terminated and not truncated:
        # action = policy.get_action(observation)

        action = np.random.choice(np.flatnonzero(value.table[observation] == value.table[observation].max()))
        #action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        obs.append(observation)
        acts.append(action)
        rews.append(reward)

    return np.array(obs), np.array(acts), np.array(rews)


ep_returns = []

for _ in range(100):
    observations, actions, rewards = generate_rollout(train_env, None)

    ep_return = np.sum(rewards)
    ep_returns.append(ep_return)

    print(f"Episode terminated with return {ep_return}.")

    for i in range(len(observations)-1):
        return_dict[observations[i], actions[i]].append(ep_return)

    # print(return_dict)

    for k in return_dict:
        # print(k)
        # print(return_dict[k])
        value.table[k] = np.mean(np.array(return_dict[k])).copy()
        # print(value.table[k])

    print(value.table)

print(f"Data collection ended with average return of {np.mean(np.array(ep_returns))}")




# print(visits.table)

env.close()

