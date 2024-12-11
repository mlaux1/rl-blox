import abc
import logging

import numpy as np
import numpy.typing as npt
import torch
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.utils import flatdim

from ..policy.base_model import NeuralNetwork, ReplayBuffer, Transition


class ValueFunction(abc.ABC):
    """Base value function class interface."""

    @abc.abstractmethod
    def get_state_value(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def update(self, observations, values) -> None:
        pass


class TabularValueFunction(ValueFunction):
    """Tabular state value function."""

    def __init__(self, observation_space: Discrete, initial_value: float = 0.0):
        self.values = np.full(
            shape=flatdim(observation_space), fill_value=initial_value
        )

    def get_state_value(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        return self.values[observation]

    def update(self, observations, step) -> None:
        self.values[observations] += step


class QFunction(abc.ABC):
    """Base Q function class."""

    @abc.abstractmethod
    def get_action_value(
        self, observation: npt.ArrayLike, action: npt.ArrayLike
    ) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def update(self, observations, actions, step) -> None:
        pass


class TabularQFunction(QFunction):
    """Tabular action value function."""

    def __init__(
        self,
        observation_space: Discrete,
        action_space: Discrete,
        initial_value: float = 0.0,
    ):
        self.values = np.full(
            shape=(flatdim(observation_space), flatdim(action_space)),
            fill_value=initial_value,
        )

    def get_action_value(
        self, observation: npt.ArrayLike, action: npt.ArrayLike
    ) -> npt.ArrayLike:
        return self.values[observation, action]

    def update(self, observations, actions, step) -> None:
        self.values[observations, actions] += step

        logging.debug(f"Updating Q Table: {observations=}, {actions=}, {step=}")
        logging.debug(f"New Q table: {self.values}")


class NNQFunction(QFunction):
    """Neural network Q function."""

    def __init__(self, observation_space, action_space):
        self.device = "cpu"
        self.q_network = NeuralNetwork(observation_space.n, action_space.n).to(
            self.device
        )
        self.target_network = NeuralNetwork(
            observation_space.n, action_space.n
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.replay_buffer = ReplayBuffer(size=10_000)
        self.batch_size = 64
        self.gamma = 0.99
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(), lr=self.learning_rate, amsgrad=True
        )

    def update(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda o: o is not None, batch.next_observation)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [o for o in batch.next_observation if o is not None]
        )

        obs_batch = torch.cat(batch.observation)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.q_network(obs_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(
                non_final_next_states
            ).max(1)[0]

        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimise the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()
