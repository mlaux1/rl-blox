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
    location_info: dict[str, list[tuple[int | None, int | None]]]
    stats: dict[str, list[Any]]

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.location_info = {}
        self.stats = {}
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
            self.location_info[key] = []
            self.stats[key] = []
        if episode is None:
            episode = self.n_episodes
        if step is None:
            step = self.n_steps
        self.location_info[key].append((episode, step))
        self.stats[key].append(value)
        if self.verbose:
            print(
                f"[{self.env_name}|{self.algorithm_name}] "
                f"({episode}|{step}) {key}: {value}"
            )
