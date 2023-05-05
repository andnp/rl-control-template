from importlib import import_module

def getAgent(name):
    mod = import_module(f'algorithms.{name}')
    return getattr(mod, name)
