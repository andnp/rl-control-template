import sys
import os
sys.path.append(os.getcwd() + '/src')

from environments.Atari import atari_games

games = ', '.join(f'"{game}"' for game in atari_games)

dqn = f"""{{
    "agent": "DQN",
    "problem": "Atari",
    "total_steps": {20_000_000},
    "episode_cutoff": 30000,
    "metaParameters": {{
        "environment": {{
            "game": [{games}]
        }},

        "epsilon": 0.01,
        "buffer_type": "uniform",
        "buffer_size": {200_000},
        "batch": 32,
        "target_refresh": 40000,
        "update_freq": 2,
        "optimizer": {{
            "name": "ADAM",
            "alpha": {4**-7},
            "beta1": 0.9,
            "beta2": 0.999
        }},

        "representation": {{
            "type": "AtariNet",
            "hidden": 512
        }}
    }}
}}
"""

eqrc = f"""{{
    "agent": "EQRC",
    "problem": "Atari",
    "total_steps": {20_000_000},
    "episode_cutoff": 30000,
    "metaParameters": {{
        "environment": {{
            "game": [{games}]
        }},

        "epsilon": 0.01,
        "buffer_size": {200_000},
        "batch": 32,
        "update_freq": 2,
        "optimizer": {{
            "name": "ADAM",
            "alpha": {4**-7},
            "beta1": 0.9,
            "beta2": 0.999
        }},

        "representation": {{
            "type": "AtariNet",
            "hidden": 512
        }}
    }}
}}
"""

specs = [
    ('DQN', dqn),
    ('EQRC', eqrc),
]

base = 'experiments/example/Atari'

for (agent, spec) in specs:
    path = f'{base}/{agent}.json'
    os.makedirs(base, exist_ok=True)
    with open(path, 'w') as f:
        f.write(spec)
