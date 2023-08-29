from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem
from environments.Atari import Atari as AtariEnv
from PyExpUtils.collection.Collector import Collector

def upperFirst(s: str):
    f = s[0].upper()
    return f + s[1:]

def toGymStr(s: str):
    if '_' in s:
        parts = s.split('_')
        return ''.join(map(upperFirst, parts))

    return upperFirst(s)

class Atari(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        game = toGymStr(self.env_params['game'])

        self.env = AtariEnv(game, self.seed)
        self.actions = self.env.num_actions()

        self.observations = (84, 84, 4)
        self.gamma = 0.99

        # enable reward clipping
        self.params['reward_clip'] = 1
