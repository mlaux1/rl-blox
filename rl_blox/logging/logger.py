import abc
import atexit
import contextlib
import os
import pprint
import time
from typing import Any

import numpy as np
import orbax.checkpoint as ocp
import tqdm
from flax import nnx

try:
    import aim
except ImportError:
    aim = None


class LoggerBase(abc.ABC):
    """Logger interface definition."""

    @property
    def n_episodes(self) -> int:
        """Number of episodes."""
        return 0

    @abc.abstractmethod
    def start_new_episode(self):
        """Register start of new episode."""

    @abc.abstractmethod
    def stop_episode(self, total_steps: int):
        """Register end of episode.

        Parameters
        ----------
        total_steps : int
            Total number of steps in the episode that just terminated.
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
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

        t : float, optional
            Wallclock time, measured with time.time().

        verbose : int, optional
            Overwrite verbosity level.

        format_str : str, optional
            Format string for stdout logging.
        """

    def define_checkpoint_frequency(  # noqa: B027
        self, key: str, checkpoint_interval: int
    ):
        """Define the checkpoint frequency for a function approximator.

        Parameters
        ----------
        key : str
            The name of the function approximator.

        checkpoint_interval : int
            Number of steps after which the function approximator should be
            saved.
        """

    def record_epoch(  # noqa: B027
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


class StandardLogger(LoggerBase):
    """Logger class to record experiment statistics.

    This logger stores experiment statistics in memory and saves checkpoints
    to disk. When the verbosity level is > 0, it will also print on stdout.

    What to track?
    https://www.reddit.com/r/reinforcementlearning/comments/j6lp7v/i_asked_rlexpert_what_and_why_he_logstracks_in/

    .. warning::

        This logger is deprecated. Use logging.checkpointer.OrbaxCheckpointer,
        MemoryLogger, or StdoutLogger. You can combine them with LoggerList.

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
    hparams: dict | None = None
    _n_episodes: int
    n_steps: int
    lpad_keys: int
    stats_loc: dict[str, list[tuple[int | None, int | None, float | None]]]
    stats: dict[str, list[Any]]
    epoch_loc: dict[str, list[tuple[int | None, int | None, float | None]]]
    epoch: dict[str, int]
    checkpointer: ocp.StandardCheckpointer | None
    checkpoint_frequencies: dict[str, int]
    checkpoint_path: dict[str, list[str]]

    def __init__(self, checkpoint_dir="/tmp/rl-blox/", verbose=0):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.verbose = verbose

        self.env_name = None
        self.algorithm_name = None
        self.start_time = 0.0
        self.hparams = None
        self._n_episodes = 0
        self.n_steps = 0
        self.lpad_keys = 0
        self.stats_loc = {}
        self.stats = {}
        self.epoch_loc = {}
        self.epoch = {}
        self.checkpointer = None
        self.checkpoint_frequencies = {}
        self.checkpoint_path = {}

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
        self.record_stat("episode_length", total_steps, verbose=0)

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
        self.hparams = hparams

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

        t : float, optional
            Wallclock time, measured with time.time().

        verbose : int, optional
            Overwrite verbosity level.

        format_str : str, optional
            Format string for stdout logging.
        """
        if key not in self.stats:
            self.stats_loc[key] = []
            self.stats[key] = []
            self.lpad_keys = max(self.lpad_keys, len(key))
        if episode is None:
            episode = self._n_episodes
        if step is None:
            step = self.n_steps
        if t is None:
            t = time.time() - self.start_time
        self.stats_loc[key].append((episode, step, t))
        self.stats[key].append(value)
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            tqdm.tqdm.write(
                f"[{self.env_name}|{self.algorithm_name}] "
                f"({episode:04d}|{step:06d}|{t:.2f}) S "  # S: statistics
                f"{key.rjust(self.lpad_keys)}: "
                f"{format_str.format(value)}"
            )

    def get_stat(self, key: str, x_key="episode"):
        """Get statistics.

        Parameters
        ----------
        key : str
            The name of the statistic.

        x_key : str in ['episode', 'step', 'time'], optional
            x-values.

        Returns
        -------
        x : array, shape (n_measurements,)
            Either episodes or steps at recorded value.

        y : array, shape (n_measurements,)
            Requested statistics.
        """
        assert key in self.stats
        X_KEYS = ["episode", "step", "time"]
        assert x_key in X_KEYS
        x_idx = X_KEYS.index(x_key)
        x = np.asarray(list(map(lambda x: x[x_idx], self.stats_loc[key])))
        y = np.asarray(self.stats[key])
        return x, y

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
        if self.checkpointer is None:
            self._init_checkpointer()

        self.checkpoint_frequencies[key] = checkpoint_interval
        self.checkpoint_path[key] = []

    def _init_checkpointer(self):
        self.checkpointer = ocp.StandardCheckpointer()
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

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
            self.epoch_loc[key] = []
            self.epoch[key] = 0
            self.lpad_keys = max(self.lpad_keys, len(key))
        if episode is None:
            episode = self._n_episodes
        if step is None:
            step = self.n_steps
        if t is None:
            t = time.time() - self.start_time
        self.epoch_loc[key].append((episode, step, t))
        self.epoch[key] += 1
        if self.verbose:
            tqdm.tqdm.write(
                f"[{self.env_name}|{self.algorithm_name}] "
                f"({episode:04d}|{step:06d}|{t:.2f}) E "  # E: epoch
                f"{key.rjust(self.lpad_keys)}: "
                f"{self.epoch[key]} epochs trained"
            )

        if (
            key in self.checkpoint_frequencies
            and self.epoch[key] % self.checkpoint_frequencies[key] == 0
        ):
            self._save_checkpoint(key, value)

    def _save_checkpoint(self, key: str, value: Any):
        checkpoint_path = os.path.join(
            f"{self.checkpoint_dir}",
            f"{self.start_time}_{self.env_name}_{self.algorithm_name}_"
            f"{key}_{self.epoch[key]}/",
        )
        _, state = nnx.split(value)
        self.checkpointer.save(f"{checkpoint_path}", state)
        self.checkpointer.wait_until_finished()
        self.checkpoint_path[key].append(checkpoint_path)
        if self.verbose:
            tqdm.tqdm.write(
                f"[{self.env_name}|{self.algorithm_name}] {key}: "
                f"checkpoint saved at {checkpoint_path}"
            )


class MemoryLogger(LoggerBase):
    """Logger class to record experiment statistics in memory.

    This logger stores experiment statistics in memory.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level.
    """

    env_name: str | None
    algorithm_name: str | None
    start_time: float
    hparams: dict | None = None
    _n_episodes: int
    n_steps: int
    stats_loc: dict[str, list[tuple[int | None, int | None, float | None]]]
    stats: dict[str, list[Any]]

    def __init__(self):
        self.env_name = None
        self.algorithm_name = None
        self.start_time = 0.0
        self.hparams = None
        self._n_episodes = 0
        self.n_steps = 0
        self.stats_loc = {}
        self.stats = {}

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
        self.record_stat("episode_length", total_steps, verbose=0)

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
        self.hparams = hparams

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

        t : float, optional
            Wallclock time, measured with time.time().

        verbose : int, optional
            Overwrite verbosity level.

        format_str : str, optional
            Format string for stdout logging.
        """
        if key not in self.stats:
            self.stats_loc[key] = []
            self.stats[key] = []
        if episode is None:
            episode = self._n_episodes
        if step is None:
            step = self.n_steps
        if t is None:
            t = time.time() - self.start_time
        self.stats_loc[key].append((episode, step, t))
        self.stats[key].append(value)

    def get_stat(self, key: str, x_key="episode"):
        """Get statistics.

        Parameters
        ----------
        key : str
            The name of the statistic.

        x_key : str in ['episode', 'step', 'time'], optional
            x-values.

        Returns
        -------
        x : array, shape (n_measurements,)
            Either episodes or steps at recorded value.

        y : array, shape (n_measurements,)
            Requested statistics.
        """
        assert key in self.stats
        X_KEYS = ["episode", "step", "time"]
        assert x_key in X_KEYS
        x_idx = X_KEYS.index(x_key)
        x = np.asarray(list(map(lambda x: x[x_idx], self.stats_loc[key])))
        y = np.asarray(self.stats[key])
        return x, y

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
        """Does nothing."""

    def record_epoch(
        self,
        key: str,
        value: Any,
        episode: int | None = None,
        step: int | None = None,
        t: float | None = None,
    ):
        """Does nothing."""


class StdoutLogger(LoggerBase):
    """Print experiment statistics to stdout.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level.
    """

    env_name: str | None
    algorithm_name: str | None
    start_time: float
    _n_episodes: int
    n_steps: int
    lpad_keys: int

    def __init__(self):
        self.env_name = None
        self.algorithm_name = None
        self.start_time = 0.0
        self._n_episodes = 0
        self.n_steps = 0
        self.lpad_keys = 0

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
        print(f"[{self.env_name}|{self.algorithm_name}] Hyperparameters:")
        pprint.pprint(hparams)

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

        t : float, optional
            Wallclock time, measured with time.time().

        verbose : int, optional
            Overwrite verbosity level.

        format_str : str, optional
            Format string for stdout logging.
        """
        self.lpad_keys = max(self.lpad_keys, len(key))
        if episode is None:
            episode = self._n_episodes
        if step is None:
            step = self.n_steps
        if t is None:
            t = time.time() - self.start_time
        if verbose is None or verbose:
            tqdm.tqdm.write(
                f"[{self.env_name}|{self.algorithm_name}] "
                f"({episode:04d}|{step:06d}|{t:.2f}) S "  # S: statistics
                f"{key.rjust(self.lpad_keys)}: "
                f"{format_str.format(value)}"
            )

    def define_checkpoint_frequency(self, key: str, checkpoint_interval: int):
        """Does nothing."""

    def record_epoch(
        self,
        key: str,
        value: Any,
        episode: int | None = None,
        step: int | None = None,
        t: float | None = None,
    ):
        """Does nothing."""


class AIMLogger(LoggerBase):
    """Use AIM to log experiment statistics.

    Parameters
    ----------
    step_counter : str, one of ['episode', 'step', 'time'], optional
        Define which value should be used as a step counter.

    log_system_params : bool, optional
        Log system parameters, e.g. memory and CPU consumption.
    """

    counter_idx: int
    log_system_params: bool
    start_time: float
    hparams: dict | None
    _n_episodes: int
    n_steps: int

    def __init__(
        self, step_counter: str = "step", log_system_params: bool = False
    ):
        if aim is None:
            raise ImportError(
                "Aim is required to use this logger, but is not installed."
            )
        assert step_counter in ["episode", "step", "time"]
        self.counter_idx = ["episode", "step", "time"].index(step_counter)
        self.log_system_params = log_system_params
        self.run = None
        self.start_time = 0.0
        self.hparams = None
        self._n_episodes = 0
        self.n_steps = 0

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
        self.record_stat("episode_length", total_steps, verbose=0)

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
        self.run = aim.Run(
            experiment=f"{env_name}-{algorithm_name}",
            log_system_params=self.log_system_params,
        )
        atexit.register(self.run.close)
        self.run["hparams"] = hparams if hparams is not None else {}

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
        """Record statistics.

        Parameters
        ----------
        key : str
            The name of the statistic.

        value : Any
            Value that should be recorded.

        episode : int, optional
            Episode which we record the statistic. Will be mapped to epochs
            in the AIM run.

        step : int, optional
            Step at which we record the statistic.

        t : float, optional
            Wallclock time, measured with time.time().

        verbose : int, optional
            Overwrite verbosity level.

        format_str : str, optional
            Format string for stdout logging.
        """
        if episode is None:
            episode = self._n_episodes
        if step is None:
            step = self.n_steps
        if t is None:
            t = time.time() - self.start_time
        s = [episode, step, t][self.counter_idx]
        with contextlib.suppress(TypeError):
            value = float(value)
        self.run.track(value=value, name=key, step=s, epoch=episode)


class LoggerList(LoggerBase):
    """Combine multiple loggers."""

    loggers: list[LoggerBase]

    def __init__(self, loggers: list[LoggerBase]):
        assert len(loggers) > 0
        self.loggers = loggers

    @property
    def n_episodes(self) -> int:
        """Number of episodes."""
        return self.loggers[0].n_episodes

    def start_new_episode(self):
        """Register start of new episode."""
        for logger in self.loggers:
            logger.start_new_episode()

    def stop_episode(self, total_steps: int):
        """Register end of episode.

        Parameters
        ----------
        total_steps : int
            Total number of steps in the episode that just terminated.
        """
        for logger in self.loggers:
            logger.stop_episode(total_steps)

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
        for logger in self.loggers:
            logger.define_experiment(env_name, algorithm_name, hparams)

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

        t : float, optional
            Wallclock time, measured with time.time().

        verbose : int, optional
            Overwrite verbosity level.

        format_str : str, optional
            Format string for stdout logging.
        """
        for logger in self.loggers:
            logger.record_stat(
                key, value, episode, step, t, verbose, format_str
            )

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
        for logger in self.loggers:
            logger.define_checkpoint_frequency(key, checkpoint_interval)

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
        for logger in self.loggers:
            logger.record_epoch(key, value, episode, step, t)
