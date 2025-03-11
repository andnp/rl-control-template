from importlib import import_module
from problems.BaseProblem import BaseProblem


def getProblem(name: str) -> type[BaseProblem]:
    if name in ['Asterix', 'Breakout', 'Freeway', 'Seaquest', 'SpaceInvaders']:
        mod = import_module('problems.Minatar')
    else:
        mod = import_module(f'problems.{name}')

    return getattr(mod, name)
