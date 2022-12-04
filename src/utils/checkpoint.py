import os
import json
import pickle
import time
from typing import Any, Callable, Dict, Optional, TypeVar
from experiment.ExperimentModel import ExperimentModel

T = TypeVar('T')
Builder = Callable[[], T]

class Checkpoint:
    def __init__(self, exp: ExperimentModel, idx: int, base_path: str = './', save_every: float = 15) -> None:
        self._storage: Dict[str, Any] = {}
        self._exp = exp
        self._idx = idx

        self._last_save: Optional[float] = None
        self._save_every = save_every * 60

        self._ctx = self._exp.buildSaveContext(idx, base=base_path)

        self._params = exp.getPermutation(idx)
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

    def save(self):
        params_path = self._ctx.resolve(self._params_path)

        if not os.path.exists(params_path):
            params_path = self._ctx.ensureExists(self._params_path, is_file=True)
            with open(params_path, 'w') as f:
                json.dump(self._params, f)

        data_path = self._ctx.ensureExists(self._data_path, is_file=True)
        with open(data_path, 'wb') as f:
            pickle.dump(self._storage, f)

    def maybe_save(self):
        if self._last_save is None:
            self._last_save = time.time()

        if time.time() - self._last_save > self._save_every:
            self.save()
            self._last_save = time.time()

    def delete(self):
        params_path = self._ctx.resolve(self._params_path)
        if os.path.exists(params_path):
            os.remove(params_path)

        data_path = self._ctx.resolve(self._data_path)
        if os.path.exists(data_path):
            os.remove(data_path)

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
            self.load()