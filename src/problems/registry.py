from src.problems.MountainCar import MountainCar

def getProblem(name):
    if name == 'MountainCar':
        return MountainCar

    raise NotImplementedError()