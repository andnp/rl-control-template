"""
Set of tools to handle preempted jobs (i.e. jobs that timed out).
Checkpointing requires enough logic to be handled in a separate file.
"""

import signal
import logging
from typing import Any, Callable, List

Callback = Callable[[], Any]

class TimeoutHandler:
    def __init__(self):
        self._todos: List[Callback] = []
        signal.signal(signal.SIGUSR1, self._handler)

    def before_cancel(self, todo: Callback):
        self._todos.append(todo)

    def _handler(self, sig, frame):
        logging.info('Received preemption signal.')
        for todo in self._todos:
            todo()
