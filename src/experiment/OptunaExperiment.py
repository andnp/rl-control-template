import sys
import json
import optuna
from typing import Any, Dict
from PyExpUtils.models.ExperimentDescription import ExperimentDescription
from PyExpUtils.utils.permute import reconstructParameters

class ExperimentModel(ExperimentDescription):
    def __init__(self, d, path: str):
        super().__init__(d, path)
        self.agent = d['agent']
        self.problem = d['problem']

        self.episode_cutoff = d.get('episode_cutoff', -1)
        self.evaluation_steps = d['evaluation_steps']
        self.evaluation_runs = d['evaluation_runs']
        self.search_epochs = d['search_epochs']
        self.sim_epochs = d.get('simultaneous_epochs', 2)

        self.config_defs = d['configuration_definitions']

        self._global_idx = 0
        self.study: optuna.Study | None = None
        d, c = _deserialize_distributions(self.config_defs)

        self.dists = d
        self._consts = c

        self._params: Dict[int, Any] = {}
        self._trials: Dict[int, Any] = {}

    @property
    def run(self):
        return self.getRun(self._global_idx)

    def set_idx(self, idx: int):
        self._global_idx = idx

    def next_hypers(self, idx: int):
        if self.study is None:
            warm_start = int(self.search_epochs // 2)
            warm_start = min(8, warm_start)
            self.study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(
                    seed=self.getRun(self._global_idx),
                    n_startup_trials=warm_start,
                ),
            )
        trial = self.study.ask(self.dists)
        self._trials[idx] = trial
        self._params[idx] = trial.params | self._consts

        return trial

    def get_hypers(self, idx: int):
        params = self._params[idx]
        return reconstructParameters(params)

    def get_flat_hypers(self, idx: int):
        params = self._params[idx]
        return params

    def record_metric(self, idx: int, v: float):
        assert self.study is not None
        trial = self._trials[idx]
        del self._trials[idx]
        self.study.tell(trial, v)

    def get_hyper_names(self):
        return set(self.dists.keys()) | set(self._consts.keys())


def _deserialize_distributions(config: Dict[str, Any]):
    flat = _flattendists(config)

    out = {}
    consts = {}
    for c, d in flat.items():
        if not isinstance(d, dict) or 't' not in d:
            consts[c] = d

        elif d['t'] == 'f':
            log = d.get('log', False)
            lo = d['lo']
            hi = d['hi']
            out[c] = optuna.distributions.FloatDistribution(lo, hi, log=log)

        elif d['t'] == 'i':
            log = d.get('log', False)
            lo = d['lo']
            hi = d['hi']
            out[c] = optuna.distributions.IntDistribution(lo, hi, log=log)

        elif d['t'] == 'b':
            out[c] = optuna.distributions.CategoricalDistribution([True, False])

        elif d['t'] == 'c':
            vals = d['vals']
            out[c] = optuna.distributions.CategoricalDistribution(vals)

    return out, consts

def _flattendists(config: Dict[str, Any], path: str = '', out: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out = out or {}

    for k, v in config.items():
        p = path + k
        if isinstance(v, dict) and 't' in v:
            out[p] = v

        elif isinstance(v, dict):
            _flattendists(v, p + '.', out)

        else:
            out[p] = v

    return out


def load(path: str | None = None):
    path = path if path is not None else sys.argv[1]
    with open(path, 'r') as f:
        d = json.load(f)

    exp = ExperimentModel(d, path)
    return exp
