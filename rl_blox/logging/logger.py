import os
import time
from typing import Any
import shutil

import orbax.checkpoint as ocp
from flax import nnx


class Logger:
    """Logger class to record experiment statistics.

    What to track?
    https://www.reddit.com/r/reinforcementlearning/comments/j6lp7v/i_asked_rlexpert_what_and_why_he_logstracks_in/

    Parameters
    ----------
    checkpoint_dir : str, optional
        Directory in which we store checkpoints. This directory will be deleted
        before the experiment starts!

    verbose : int, optional
        Verbosity level.
    """

    checkpoint_dir: str
    verbose: int
    env_name: str | None
    algorithm_name: str | None
    start_time: float
    n_episodes: int
    n_steps: int
    stats_loc: dict[str, list[tuple[int | None, int | None]]]
    stats: dict[str, list[Any]]
    epoch_loc: dict[str, list[tuple[int | None, int | None]]]
    epoch: dict[str, int]
    checkpointer: ocp.StandardCheckpointer | None
    checkpoint_frequencies: dict[str, int]
    checkpoint_path: dict[str, list[str]]

    def __init__(self, checkpoint_dir="/tmp/rl-blox/", verbose=0):
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose

        self.env_name = None
        self.algorithm_name = None
        self.stats_loc = {}
        self.stats = {}
        self.epoch_loc = {}
        self.epoch = {}
        self.checkpointer = None
        self.checkpoint_frequencies = {}
        self.checkpoint_path = {}
        self.start_time = 0.0
        self.n_episodes = 0
        self.n_steps = 0
        self.define_experiment()

    def start_new_episode(self):
        """Increase episode counter."""
        self.n_episodes += 1

    def stop_episode(self, total_steps):
        """Increase step counter and records 'episode_length'.

        Parameters
        ----------
        total_steps : int
            Total number of steps in the episode that just terminated.
        """
        self.n_steps += total_steps
        self.record_stat("episode_length", total_steps, verbose=0)

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
        self.start_time = time.time()

    def define_checkpoint_frequency(self, key: str, frequency: int):
        """Define the checkpoint frequency for a function approximator.

        Parameters
        ----------
        key : str
            The name of the function approximator.

        frequency : int
            Frequency at which the function approximator should be saved.
        """
        if self.checkpointer is None:
            self._init_checkpointer()

        self.checkpoint_frequencies[key] = frequency
        self.checkpoint_path[key] = []

    def _init_checkpointer(self):
        self.checkpointer = ocp.StandardCheckpointer()
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir)

    def record_stat(
        self,
        key: str,
        value: Any,
        episode: int | None = None,
        step: int | None = None,
        verbose: int | None = None,
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

        verbose : int, optional
            Overwrite verbosity level.
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
        verbose = self.verbose if verbose is None else verbose
        if verbose:
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

        if (
            key in self.checkpoint_frequencies
            and self.epoch[key] % self.checkpoint_frequencies[key] == 0
        ):
            checkpoint_path = (
                f"{self.checkpoint_dir}"
                f"{self.start_time}_{self.env_name}_{self.algorithm_name}_"
                f"{key}_{self.epoch[key]}/"
            )
            _, state = nnx.split(value)
            self.checkpointer.save(f"{checkpoint_path}", state)
            self.checkpoint_path[key].append(checkpoint_path)
            if self.verbose:
                print(
                    f"[{self.env_name}|{self.algorithm_name}] {key}: "
                    f"checkpoint saved at {checkpoint_path}"
                )
