import time

from rl_blox.logging.logger import LoggerBase


class Timer:
    """Profile program execution time between some explicitly specified start and stop times.

    The timer has time accounts (in a 'timetable') for different tasks. For
    each task, only the total execution time is recorded.

    Time is in seconds.

    The timer object can be used as follows:

    >>> from rl_blox.logging.logger import StdoutLogger
    >>> timer = Timer()
    >>> timer.start('mytask')
    >>> for i in range(10): time.sleep(0.001)
    >>> timer.stop('mytask')
    >>> timer.start('mytask')
    >>> for i in range(20): time.sleep(0.001)
    >>> timer.stop('mytask')
    >>> timer.log(StdoutLogger())
    [None|None] (0000|000000|...) S duration:mytask: 0.03...
    timer_warnings:

    If a task is started while another is running, the new one is considered
    to be below the already running one:

    >>> from rl_blox.logging.logger import StdoutLogger
    >>> timer = Timer()
    >>> timer.start('toptask')
    >>> timer.start('subtask')
    >>> for i in range(10): time.sleep(0.001)
    >>> timer.stop('subtask')
    >>> timer.stop('toptask')
    >>> timer.log(StdoutLogger())
    [None|None] (0000|000000|...) S duration:toptask/subtask: 0.01...
    [None|None] (0000|000000|...) S duration:toptask: 0.01...
    timer_warnings:

    A task has a separate timetable account for each level in the hierarchy of
    active tasks.

    Make sure to stop tasks in the reverse order that you started them in.
    """

    time_table: dict[str, float]
    active_tasks_and_starts: list[tuple[str, float]]
    warnings: list[str]

    def __init__(self):
        self.time_table = dict()
        self.active_tasks_and_starts = []
        self.warnings = []

    def _active_tasks(self) -> list[str]:
        """List of names of currently active tasks."""
        return list(map(lambda tas: tas[0], self.active_tasks_and_starts))

    def _show_active_tasks(self) -> str:
        """String showing the hierarchy of currently active tasks.

        The names of tasks are separated by a /.
        """
        return "/".join(self._active_tasks())

    def is_task_active(self, task: str) -> bool:
        """Whether the task with the given name is currently active, regardless
        of which level in the hierarchy it is active in.
        """
        return task in map(lambda tas: tas[0], self.active_tasks_and_starts)

    def _start_time(self, task: str) -> float | None:
        """Return the last time an active task was started, or None if it is
        not active."""
        for t, start_time in self.active_tasks_and_starts:
            if t == task:
                return start_time
        return None

    def _add_time(self, task: str, time: float):
        """Add the given amount of time to the timetable for the task with
        the given name."""
        if task not in self.time_table:
            self.time_table[task] = 0
        self.time_table[task] = self.time_table[task] + time

    def _active_tasks_below(self, task: str) -> list[str]:
        """Return the list of active tasks that are below the task with the
        given name in the hierarchy."""
        active_tasks = self._active_tasks()
        active_index = active_tasks.index(task)
        return active_tasks[active_index:]

    def _deepest_active_task(self) -> str | None:
        """Return the deepest active task in the hierarchy of active tasks
        or None if no task is active."""
        active_tasks = self._active_tasks()
        if len(active_tasks) == 0:
            return None
        return active_tasks[-1]

    def _stop_deepest_task(self, task: str, stop_time: float):
        """Stop the deepest active task in the hierarchy of active tasks
        and ensure that it is the expected task (the one with the given
        name)."""
        deepest_active_task = self._deepest_active_task()
        if task != deepest_active_task:
            raise RuntimeError(
                f'Programming error: '
                f'Predicted deepest active task "{task}" does not match actual '
                f'value: "{deepest_active_task}". '
                f'Did you forget to stop "{deepest_active_task}"?'
            )
        duration = stop_time - self._start_time(task)
        self._add_time(self._show_active_tasks(), duration)
        self.active_tasks_and_starts.pop(-1)

    def start(self, task: str):
        """Take note that execution of the task with the given name has started
        or resumed.

        If the task is already running, do nothing.

        Parameters
        ----------
        task
            Name of the task whose start to take note of.
        """
        if self.is_task_active(task):
            return
        self.active_tasks_and_starts.append((task, time.time()))

    def stop(self, task: str):
        """Take note that execution of the task with the given name has stopped.

        If the task is not running, a warning will be output to STDOUT on
        log().

        Parameters
        ----------
        task
            Name of the task whose stop to take note of.
        """
        stop_time = time.time()
        if task == 'training':
            print('training stopped')
        if not self.is_task_active(task):
            msg = (
                f'Inactive task "{task}" stopped. '
                f'Hierarchy was "{self._show_active_tasks()}"'
            )
            self.warnings.append(msg)
            return
        tasks_to_stop = self._active_tasks_below(task)
        tasks_to_stop.reverse()
        for t in tasks_to_stop:
            self._stop_deepest_task(t, stop_time)

    def log(self, logger: LoggerBase | None):
        """Log the execution times (the timetable) to the logger.

        If any warnings occurred, print them to STDOUT.
        """
        if logger is None:
            return
        for task, duration in self.time_table.items():
            statistic_key = 'duration:' + task
            logger.record_stat(statistic_key, duration)
        warnings_txt = "\n".join(self.warnings)
        print('timer_warnings: ')
        print(warnings_txt)
