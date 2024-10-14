import numpy as np
import numba
import gymnasium
import ale_py
from typing import Optional
from collections import deque
from PIL import Image
from RlGlue.environment import BaseEnvironment

# add ALE envs to gym register for name lookup
gymnasium.register_envs(ale_py)

class Atari(BaseEnvironment):
    def __init__(self, game: str, seed: int, max_steps: Optional[int] = None):
        self.env = gymnasium.make(f'ALE/{game}-v5', max_episode_steps=max_steps, frameskip=5, repeat_action_probability=0.25)
        # self.env = gym.make(f'ALE/{game}-v5', max_episode_steps=max_steps, render_mode='human')
        self.seed = seed

        self.max_steps = max_steps

        # preprocessor state
        self._stacker = FrameStacker(size=4)
        self._last_lives = 0

    def num_actions(self) -> int:
        space = self.env.action_space

        # get around some faulty type-checking here
        assert hasattr(space, 'n')
        return getattr(space, 'n')

    def start(self):
        self._stacker.clear()
        s, info = self.env.reset(seed=self.seed)
        self.seed += 1

        self._last_lives = info['lives']

        s = process_image(s)
        return self._stacker.next(s)

    def step(self, a):
        sp, r, t, _, info = self.env.step(a)

        gamma = 1
        if info['lives'] < self._last_lives:
            gamma = 0
            self._last_lives = info['lives']

        if t:
            gamma = 0

        # do preprocessing steps
        sp = process_image(sp)
        sp = self._stacker.next(sp)

        return (r, sp, t, {'gamma': gamma})

class FrameStacker:
    def __init__(self, size: int):
        self._size = size
        self._q = deque(maxlen=size)
        self._pad: Optional[np.ndarray] = None

    def clear(self):
        self._q.clear()

    def next(self, obs: np.ndarray):
        self._q.append(obs)

        if self._pad is None:
            self._pad = np.zeros_like(obs)

        frames = list(self._q)
        if len(frames) < self._size:
            missing = self._size - len(frames)
            frames += [self._pad] * missing

        return np.stack(frames, axis=-1)

def process_image(img: np.ndarray):
    gray = grayscale(img)
    image = Image.fromarray(gray).resize((84, 84), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)

@numba.njit(cache=True, nogil=True, fastmath=True)
def grayscale(obs: np.ndarray):
    obs = obs.astype(np.float32)
    avg = np.array([0.299, 0.587, 1 - (0.299 + 0.587)], dtype=np.float32)

    w, h, _ = obs.shape
    out = np.zeros((w, h), dtype=np.float32)

    for i in range(w):
        out[i] = obs[i].dot(avg)

    return out.astype(np.uint8)

# ----------------------------------

atari_scores = {
    'alien': (227.8, 7127.7),
    'amidar': (5.8, 1719.5),
    'assault': (222.4, 742.0),
    'asterix': (210.0, 8503.3),
    'asteroids': (719.1, 47388.7),
    'atlantis': (12850.0, 29028.1),
    'bank_heist': (14.2, 753.1),
    'battle_zone': (2360.0, 37187.5),
    'beam_rider': (363.9, 16926.5),
    'berzerk': (123.7, 2630.4),
    'bowling': (23.1, 160.7),
    'boxing': (0.1, 12.1),
    'breakout': (1.7, 30.5),
    'centipede': (2090.9, 12017.0),
    'chopper_command': (811.0, 7387.8),
    'crazy_climber': (10780.5, 35829.4),
    'defender': (2874.5, 18688.9),
    'demon_attack': (152.1, 1971.0),
    'double_dunk': (-18.6, -16.4),
    'enduro': (0.0, 860.5),
    'fishing_derby': (-91.7, -38.7),
    'freeway': (0.0, 29.6),
    'frostbite': (65.2, 4334.7),
    'gopher': (257.6, 2412.5),
    'gravitar': (173.0, 3351.4),
    'hero': (1027.0, 30826.4),
    'ice_hockey': (-11.2, 0.9),
    'jamesbond': (29.0, 302.8),
    'kangaroo': (52.0, 3035.0),
    'krull': (1598.0, 2665.5),
    'kung_fu_master': (258.5, 22736.3),
    'montezuma_revenge': (0.0, 4753.3),
    'ms_pacman': (307.3, 6951.6),
    'name_this_game': (2292.3, 8049.0),
    'phoenix': (761.4, 7242.6),
    'pitfall': (-229.4, 6463.7),
    'pong': (-20.7, 14.6),
    'private_eye': (24.9, 69571.3),
    'qbert': (163.9, 13455.0),
    'riverraid': (1338.5, 17118.0),
    'road_runner': (11.5, 7845.0),
    'robotank': (2.2, 11.9),
    'seaquest': (68.4, 42054.7),
    'skiing': (-17098.1, -4336.9),
    'solaris': (1236.3, 12326.7),
    'space_invaders': (148.0, 1668.7),
    'star_gunner': (664.0, 10250.0),
    'surround': (-10.0, 6.5),
    'tennis': (-23.8, -8.3),
    'time_pilot': (3568.0, 5229.2),
    'tutankham': (11.4, 167.6),
    'up_n_down': (533.4, 11693.2),
    'venture': (0.0, 1187.5),
    # Note the random agent score on Video Pinball is sometimes greater than the
    # human score under other evaluation methods.
    'video_pinball': (16256.9, 17667.9),
    'wizard_of_wor': (563.5, 4756.5),
    'yars_revenge': (3092.9, 54576.9),
    'zaxxon': (32.5, 9173.3),
}

atari_games = list(atari_scores.keys())
