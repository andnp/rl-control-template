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
        self._times_received = 0
        signal.signal(signal.SIGTERM, self._handler)

    def before_cancel(self, todo: Callback):
        self._todos.append(todo)

    def _handler(self, sig, frame):
        self._times_received += 1
        logging.info(f'Received preemption signal. Times: {self._times_received}')

        if self._times_received > 1:
            exit(130)

        for todo in self._todos:
            try:
                todo()
            except Exception as e:
                print(e)

        logging.info('Exiting gracefully now')
        exit()
