from typing import Any


class Logger:
    """Logger class to record experiment statistics.

    What to track? - https://www.reddit.com/r/reinforcementlearning/comments/j6lp7v/i_asked_rlexpert_what_and_why_he_logstracks_in/

    TODO save checkpoints with orbax
    """

    verbose: int
    env_name: str | None
    algorithm_name: str | None
    n_episodes: int
    n_steps: int
    stats_loc: dict[str, list[tuple[int | None, int | None]]]
    stats: dict[str, list[Any]]
    epoch_loc: dict[str, list[tuple[int | None, int | None]]]
    epoch: dict[str, int]
    checkpoint_frequencies: dict[str, int]
    checkpoint_path: dict[str, list[int]]

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.stats_loc = {}
        self.stats = {}
        self.epoch_loc = {}
        self.epoch = {}
        self.checkpoint_frequencies = {}
        self.checkpoint_path = {}
        self.n_episodes = 0
        self.n_steps = 0
        self.define_experiment()

    def start_new_episode(self):
        self.n_episodes += 1

    def stop_episode(self, total_steps):
        self.n_steps += total_steps

    def define_experiment(
        self, env_name: str | None = None, algorithm_name: str | None = None
    ):
        """Define the experiment.

        Parameters
        ----------
        env_name : str, optional
            The name of the gym environment.

        algorithm_name : str, optional
            The name of the reinforcement learning algorithm.
        """
        self.env_name = env_name
        self.algorithm_name = algorithm_name

    def register_checkpoint_frequency(self, key: str, frequency: int):
        """Register the checkpoint frequency for a function approximator.

        Parameters
        ----------
        key : str
            The name of the function approximator.

        frequency : int
            Frequency at which the function approximator should be saved.
        """
        self.checkpoint_frequencies[key] = frequency
        self.checkpoint_path[key] = []

    def record_stat(
        self,
        key: str,
        value: Any,
        episode: int | None = None,
        step: int | None = None,
    ):
        """Record statistics.

        Parameters
        ----------
        key : str
            The name of the statistic.

        value : Any
            Value that should be recorded.

        episode : int, optional
            Episode which we record the statistic.

        step : int, optional
            Step at which we record the statistic.
        """
        if key not in self.stats:
            self.stats_loc[key] = []
            self.stats[key] = []
        if episode is None:
            episode = self.n_episodes
        if step is None:
            step = self.n_steps
        self.stats_loc[key].append((episode, step))
        self.stats[key].append(value)
        if self.verbose:
            print(
                f"[{self.env_name}|{self.algorithm_name}] "
                f"({episode}|{step}) {key}: {value}"
            )

    def record_epoch(
        self,
        key: str,
        value: Any,
        episode: int | None = None,
        step: int | None = None,
    ):
        """Record training epoch of function approximator.

        Parameters
        ----------
        key : str
            The name of the function approximator.

        value : Any
            Function approximator.

        episode : int, optional
            Episode which we record the statistic.

        step : int, optional
            Step at which we record the statistic.
        """
        if key not in self.epoch:
            self.epoch_loc[key] = []
            self.epoch[key] = 0
        if episode is None:
            episode = self.n_episodes
        if step is None:
            step = self.n_steps
        self.epoch_loc[key].append((episode, step))
        self.epoch[key] += 1
        if self.verbose:
            print(
                f"[{self.env_name}|{self.algorithm_name}] "
                f"({episode}|{step}) {key}: {self.epoch[key]} epochs trained"
            )
        if key in self.checkpoint_frequencies:
            if self.epoch[key] % self.checkpoint_frequencies[key] == 0:
                checkpoint_path = "TBD"
                self.checkpoint_path[key].append(checkpoint_path)
                if self.verbose >= 2:
                    print(
                        f"[{self.env_name}|{self.algorithm_name}] "
                        f"({episode}|{step}) {key}: "
                        f"checkpoint saved at {checkpoint_path}"
                    )
                raise NotImplementedError("Checkpointing not implemented yet.")
