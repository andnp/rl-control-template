from typing import Sequence

import PyExpUtils.utils.path as Path

def findExpPath(arr: Sequence[str], alg: str):
    for exp_path in arr:
        if f'{alg.lower()}.json' == Path.fileName(exp_path.lower()):
            return exp_path

    raise Exception(f'Expected to find exp_path for {alg}')
