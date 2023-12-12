import abc


class BaseAlgorithm:
    @abc.abstractmethod
    def train(self, num_episodes: int):
        pass
