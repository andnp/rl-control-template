from problems.Acrobot import Acrobot
from problems.MountainCar import MountainCar
from problems.Cartpole import Cartpole
from problems.Breakout import Breakout
from problems.Atari import Atari

def getProblem(name):
    if name == 'MountainCar':
        return MountainCar

    if name == 'Cartpole':
        return Cartpole

    if name == 'AcrobotGym':
        return Acrobot

    if name == 'Breakout':
        return Breakout

    if name == 'Atari':
        return Atari

    raise NotImplementedError()
