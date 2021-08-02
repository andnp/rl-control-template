from problems.MountainCar import MountainCar
from problems.Cartpole import Cartpole

def getProblem(name):
    if name == 'MountainCar':
        return MountainCar

    if name == 'Cartpole':
        return Cartpole

    raise NotImplementedError()
