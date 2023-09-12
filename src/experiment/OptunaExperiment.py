import sys
import json
import optuna
import numpy as np
from typing import Any, Dict, Sequence
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

        self.config_defs = d['configuration_definitions']

        self._idx = 0
        self.study: optuna.Study | None = None
        self.trial: optuna.Trial | None = None
        d, c = _deserialize_distributions(self.config_defs)

        self._dists = d
        self._consts = c

    def set_idx(self, idx: int):
        self._idx = idx

    def next_hypers(self):
        if self.study is None:
            self.study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(
                    seed=self.getRun(self._idx),
                    n_ei_candidates=self.evaluation_runs,
                    n_startup_trials=int((self.search_epochs * self.evaluation_runs) // 2),
                ),
            )
        self.trial = self.study.ask(self._dists)

    def get_hypers(self, idx: int):
        assert self.trial is not None
        params = reconstructParameters(self.trial.params | self._consts)
        return params

    def get_flat_hypers(self):
        assert self.trial is not None
        params = self.trial.params | self._consts
        return params

    def record_metric(self, v: Sequence[float]):
        assert self.trial is not None
        assert self.study is not None

        trials = [optuna.trial.create_trial(
            params=self.trial.params,
            distributions=self.trial.distributions,
            value=v[i]
        ) for i in range(1, len(v))]

        self.study.tell(self.trial, v[0])
        self.study.add_trials(trials)

    def get_hyper_names(self):
        return set(self._dists.keys()) | set(self._consts.keys())


def _deserialize_distributions(config: Dict[str, Any]):
    flat = _flatten_dists(config)

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

def _flatten_dists(config: Dict[str, Any], path: str = '', out: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out = out or {}

    for k, v in config.items():
        p = path + k
        if isinstance(v, dict) and 't' in v:
            out[p] = v

        elif isinstance(v, dict):
            _flatten_dists(v, p + '.', out)

        else:
            out[p] = v

    return out


def load(path: str | None = None):
    path = path if path is not None else sys.argv[1]
    with open(path, 'r') as f:
        d = json.load(f)

    exp = ExperimentModel(d, path)
    return exp
