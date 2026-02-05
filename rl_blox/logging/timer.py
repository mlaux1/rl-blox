import time

from rl_blox.logging.logger import LoggerBase


def intercalate(a: str, bs: list[str]) -> str:
    result = ""
    for i, b in enumerate(bs):
        result = result + b
        if i < len(bs)-1:
            result = result + a
    return result


class Timer:
    time_table: dict[str, float]
    active_tasks_and_starts: list[tuple[str, float]]
    warnings: list[str]

    def __init__(self):
        self.time_table = dict()
        self.active_tasks_and_starts = []
        self.warnings = []

    def _active_tasks(self) -> list[str]:
        return list(map(lambda tas: tas[0], self.active_tasks_and_starts))

    def _show_active_tasks(self) -> str:
        return intercalate("/", self._active_tasks())

    def is_task_active(self, task: str) -> bool:
        return task in map(lambda tas: tas[0], self.active_tasks_and_starts)

    def _start_time(self, task: str) -> float | None:
        for t, start_time in self.active_tasks_and_starts:
            if t == task:
                return start_time
        return None

    def _add_time(self, task: str, time: float):
        if task not in self.time_table:
            self.time_table[task] = 0
        self.time_table[task] = self.time_table[task] + time

    def _active_tasks_below(self, task: str) -> list[str]:
        active_tasks = self._active_tasks()
        active_index = active_tasks.index(task)
        return active_tasks[active_index:]

    def _deepest_active_task(self) -> str | None:
        active_tasks = self._active_tasks()
        if len(active_tasks) == 0:
            return None
        return active_tasks[-1]

    def _stop_deepest_task(self, task: str, stop_time: float):
        deepest_active_task = self._deepest_active_task()
        if task != deepest_active_task:
            raise RuntimeError("Programming error: Predicted deepest active task does not match actual value")
        duration = stop_time - self._start_time(task)
        self._add_time(self._show_active_tasks(), duration)
        self.active_tasks_and_starts.pop(-1)

    def start(self, task: str):
        if self.is_task_active(task):
            return
        self.active_tasks_and_starts.append((task, time.time()))

    def stop(self, task: str):
        stop_time = time.time()
        if task == "training":
            print("training stopped")
        if not self.is_task_active(task):
            msg = f"Inactive task '{task}' stopped. Hierarchy was '{self._show_active_tasks()}'"
            self.warnings.append(msg)
            return
        tasks_to_stop = self._active_tasks_below(task)
        tasks_to_stop.reverse()
        for t in tasks_to_stop:
            self._stop_deepest_task(t, stop_time)

    def log(self, logger: LoggerBase | None):
        if logger is None:
            return
        for task, duration in self.time_table.items():
            statistic_key = "duration:" + task
            logger.record_stat(statistic_key, duration)
        warnings_txt = intercalate("\n", self.warnings)
        print("timer_warnings: ")
        print(warnings_txt)
