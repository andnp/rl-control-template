import os
import json
import time
import shutil
import pickle
import logging
from typing import Any, Callable, Dict, Optional, Sequence, Type, TypeVar, Protocol
from PyExpUtils.models.ExperimentDescription import ExperimentDescription

T = TypeVar('T')
Builder = Callable[[], T]

class Checkpoint:
    def __init__(self, exp: ExperimentDescription, idx: int, base_path: str = './', save_every: float = -1) -> None:
        self._storage: Dict[str, Any] = {}
        self._exp = exp
        self._idx = idx

        self._last_save: Optional[float] = None
        self._save_every = save_every * 60

        self._ctx = self._exp.buildSaveContext(idx, base=base_path)

        self._params = exp.getPermutation(idx)
        self._base_path = f'{idx}'
        self._params_path = f'{idx}/params.json'
        self._data_path = f'{idx}/chk.pkl'

    def __getitem__(self, name: str):
        return self._storage[name]

    def __setitem__(self, name: str, v: T) -> T:
        self._storage[name] = v
        return v

    def build(self, name: str, builder: Builder[T]) -> T:
        if name in self._storage:
            return self._storage[name]

        self._storage[name] = builder()
        return self._storage[name]

    def initial_value(self, name: str, val: T) -> T:
        if name in self._storage:
            return self._storage[name]

        self._storage[name] = val
        return val

    def save(self):
        params_path = self._ctx.resolve(self._params_path)

        logging.info('Dumping checkpoint')
        if not os.path.exists(params_path):
            params_path = self._ctx.ensureExists(self._params_path, is_file=True)
            with open(params_path, 'w') as f:
                json.dump(self._params, f)

        data_path = self._ctx.ensureExists(self._data_path, is_file=True)
        with open(data_path, 'wb') as f:
            pickle.dump(self._storage, f)

        logging.info('Finished dumping checkpoint')

    def maybe_save(self):
        if self._save_every < 0:
            return

        if self._last_save is None:
            self._last_save = time.time()

        if time.time() - self._last_save > self._save_every:
            self.save()
            self._last_save = time.time()

    def delete(self):
        base_path = self._ctx.resolve(self._base_path)
        if os.path.exists(base_path):
            shutil.rmtree(base_path)

    def load(self):
        params_path = self._ctx.resolve(self._params_path)

        try:
            with open(params_path, 'r') as f:
                params = json.load(f)

            assert params == self._params, 'The idx->params mapping has changed between checkpoints!!'

        except Exception as e:
            print('Failed to load checkpoint')
            print(e)

        path = self._ctx.resolve(self._data_path)
        try:
            with open(path, 'rb') as f:
                self._storage = pickle.load(f)
        except Exception as e:
            print(f'Failed to load checkpoint: {path}')
            print(e)

    def load_if_exists(self):
        if self._ctx.exists(self._data_path):
            print('Found a checkpoint! Loading...')
            self.load()


class Checkpointable(Protocol):
    def __setstate__(self, state) -> None: ...
    def __getstate__(self) -> Dict[str, Any]: ...

C = TypeVar('C', bound=Type[Checkpointable])
def checkpointable(props: Sequence[str]):
    def _inner(c: C) -> C:
        o_getter = getattr(c, '__getstate__')
        o_setter = getattr(c, '__setstate__')

        def setter(self, state):
            if o_setter is not None:
                o_setter(self, state)

            for p in props:
                setattr(self, p, state[p])

        def getter(self):
            out = {}
            for p in props:
                out[p] = getattr(self, p)

            out2 = {}
            if o_getter is not None:
                out2 = o_getter(self)
            elif getattr(c.__bases__[0], '__getstate__'):
                _getter = getattr(c.__bases__[0], '__getstate__')
                out2 = _getter(self)

            out2 |= out
            return out2

        c.__getstate__ = getter
        c.__setstate__ = setter

        return c

    return _inner
