from collections.abc import Callable, Iterable, Sequence
import importlib
from pathlib import Path
from PyExpUtils.models.ExperimentDescription import ExperimentDescription, loadExperiment
from PyExpUtils.results.tools import getHeader, getParamsAsDict
from PyExpUtils.results.indices import listIndices
from ml_instrumentation.reader import load_all_results, get_run_ids

import polars as pl

class Result[Exp: ExperimentDescription]:
    def __init__(self, exp_path: str | Path, exp: Exp, metrics: Sequence[str] | None = None):
        self.exp_path = str(exp_path)
        self.exp = exp
        self.metrics = metrics

    def load(self):
        db_path = self.exp.buildSaveContext(0).resolve('results.db')

        if not Path(db_path).exists():
            return None

        dfs: list[pl.DataFrame] = []
        for param_id in range(self.exp.numPermutations()):
            params = getParamsAsDict(self.exp, param_id)
            run_ids = get_run_ids(db_path, params)

            df = load_all_results(db_path, self.metrics, run_ids)
            dfs.append(df)

        return pl.concat(dfs)

    @property
    def filename(self):
        return self.exp_path.split('/')[-1].removesuffix('.json')


class ResultCollection[Exp: ExperimentDescription]:
    def __init__(self, path: str | Path | None = None, metrics: Sequence[str] | None = None, Model: type[Exp] = ExperimentDescription):
        self.metrics = metrics
        self.Model = Model

        if path is None:
            main_file = importlib.import_module('__main__').__file__
            assert main_file is not None
            path = Path(main_file).parent

        self.path = Path(path)

        project = Path.cwd()
        paths = self.path.glob('**/*.json')
        paths = map(lambda p: p.relative_to(project), paths)
        paths = map(str, paths)
        self.paths = list(paths)


    def _result(self, path: str):
        exp = loadExperiment(path, self.Model)
        return Result[Exp](path, exp, self.metrics)


    def get_hyperparameter_columns(self):
        hypers = set[str]()

        for path in self.paths:
            exp = loadExperiment(path, self.Model)
            hypers |= set(getHeader(exp))

        return sorted(hypers)


    def groupby_directory(self, level: int):
        uniques = set(
            p.split('/')[level] for p in self.paths
        )

        for group in uniques:
            group_paths = [p for p in self.paths if p.split('/')[level] == group]
            results = map(self._result, group_paths)
            yield group, list(results)


    def __iter__(self):
        return map(self._result, self.paths)


def detect_missing_indices(exp: ExperimentDescription, runs: int, base: str = './'):
    context = exp.buildSaveContext(0, base=base)
    header = getHeader(exp)
    path = context.resolve('results.db')

    if not context.exists('results.db'):
        yield from listIndices(exp, runs)
        return

    n_params = exp.numPermutations()
    for param_id in range(n_params):
        run_ids = set(get_run_ids(path, getParamsAsDict(exp, param_id, header=header)))

        for seed in range(runs):
            run_id = seed * n_params + param_id
            if run_id not in run_ids:
                yield run_id


def gather_missing_indices(experiment_paths: Iterable[str], runs: int, loader: Callable[[str], ExperimentDescription] = loadExperiment, base: str = './'):
    path_to_indices: dict[str, list[int]] = {}

    for path in experiment_paths:
        exp = loader(path)
        indices = detect_missing_indices(exp, runs, base=base)
        indices = sorted(indices)
        path_to_indices[path] = indices

        size = exp.numPermutations() * runs
        print(path, f'{len(indices)} / {size}')

    return path_to_indices
