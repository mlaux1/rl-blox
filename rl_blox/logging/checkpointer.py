import os
import time
from typing import Any

import orbax.checkpoint as ocp
import tqdm
from flax import nnx

from .logger import LoggerBase


class OrbaxCheckpointer(LoggerBase):
    """Checkpoint networks with Orbax.

    This logger saves checkpoints to disk with Orbax. When the verbosity level
    is > 0, it will also print on stdout.

    Parameters
    ----------
    checkpoint_dir : str, optional
        Directory in which we store checkpoints.

        .. warning::

            This directory will be created if it does not exist.

    verbose : int, optional
        Verbosity level.
    """

    checkpoint_dir: str
    verbose: int
    env_name: str | None
    algorithm_name: str | None
    start_time: float
    _n_episodes: int
    n_steps: int
    lpad_keys: int
    epoch: dict[str, int]
    last_checkpoint_step: dict[str, int]
    checkpointer: ocp.StandardCheckpointer | None
    checkpoint_frequencies: dict[str, int]
    checkpoint_path: dict[str, list[str]]

    def __init__(self, checkpoint_dir="/tmp/rl-blox/", verbose=0):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.verbose = verbose

        self.env_name = None
        self.algorithm_name = None
        self.start_time = 0.0
        self._n_episodes = 0
        self.n_steps = 0
        self.lpad_keys = 0
        self.epoch = {}
        self.last_step = {}
        self.checkpointer = ocp.StandardCheckpointer()
        self.checkpoint_frequencies = {}
        self.checkpoint_path = {}

        self._make_checkpoint_dir()

    def _make_checkpoint_dir(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    def start_new_episode(self):
        """Register start of new episode."""
        self._n_episodes += 1

    def stop_episode(self, total_steps: int):
        """Register end of episode.

        Increase step counter and records 'episode_length'.

        Parameters
        ----------
        total_steps : int
            Total number of steps in the episode that just terminated.
        """
        self.n_steps += total_steps

    def define_experiment(
        self,
        env_name: str | None = None,
        algorithm_name: str | None = None,
        hparams: dict | None = None,
    ):
        """Define the experiment.

        Parameters
        ----------
        env_name : str, optional
            The name of the gym environment.

        algorithm_name : str, optional
            The name of the reinforcement learning algorithm.

        hparams : dict, optional
            Hyperparameters of the experiment.
        """
        self.env_name = env_name
        self.algorithm_name = algorithm_name
        self.start_time = time.time()

    def record_stat(
        self,
        key: str,
        value: Any,
        episode: int | None = None,
        step: int | None = None,
        t: float | None = None,
        verbose: int | None = None,
        format_str: str = "{0:.3f}",
    ):
        """Does nothing."""

    def define_checkpoint_frequency(self, key: str, checkpoint_interval: int):
        """Define the checkpoint frequency for a function approximator.

        Parameters
        ----------
        key : str
            The name of the function approximator.

        checkpoint_interval : int
            Number of steps after which the function approximator should be
            saved.
        """
        self.checkpoint_frequencies[key] = checkpoint_interval
        self.checkpoint_path[key] = []
        self.last_step[key] = 0

    def record_epoch(
        self,
        key: str,
        value: Any,
        episode: int | None = None,
        step: int | None = None,
        t: float | None = None,
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

        t : float, optional
            Wallclock time, measured with time.time().
        """
        if key not in self.epoch:
            self.epoch[key] = 0
            self.lpad_keys = max(self.lpad_keys, len(key))
        self.epoch[key] += 1

        if episode is None:
            episode = self._n_episodes
        if step is None:
            step = self.n_steps

        if self.verbose >= 2:
            if t is None:
                t = time.time() - self.start_time
            tqdm.tqdm.write(
                f"[{self.env_name}|{self.algorithm_name}] "
                f"({episode:04d}|{step:06d}|{t:.2f}) E "  # E: epoch
                f"{key.rjust(self.lpad_keys)}: "
                f"{self.epoch[key]} epochs trained"
            )

        if key in self.checkpoint_frequencies:
            # check if the step counter wrapped around as we cannot rely on
            # x % y == 0 because of delayed updates (e.g., for the policy)
            if (
                self.last_step[key] % self.checkpoint_frequencies[key]
                > step % self.checkpoint_frequencies[key]
            ) or (
                (step - self.last_step[key]) >= self.checkpoint_frequencies[key]
            ):
                self._save_checkpoint(key, value, step)

        self.last_step[key] = step

    def _save_checkpoint(self, key: str, value: Any, step: int):
        checkpoint_path = os.path.join(
            f"{self.checkpoint_dir}",
            f"{self.env_name}_{self.algorithm_name}_{self.start_time}_"
            f"{key}_step_{step:09d}_epoch_{self.epoch[key]}/",
        )

        self.save_model(checkpoint_path, value)

        self.checkpoint_path[key].append(checkpoint_path)
        if self.verbose:
            tqdm.tqdm.write(
                f"[{self.env_name}|{self.algorithm_name}] {key}: "
                f"checkpoint saved at {checkpoint_path}"
            )

    def save_model(self, path: str, model: nnx.Module):
        """Save model with Orbax.

        Parameters
        ----------
        path : str
            Full path to model.

        model : nnx.Module
            Function approximator to be stored.
        """
        state = nnx.state(model)
        self.checkpointer.save(path, state)
        self.checkpointer.wait_until_finished()
