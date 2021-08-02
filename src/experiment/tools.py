import sys
import glob
from collections import namedtuple
from typing import Generator
import PyExpUtils.utils.path as Path

NON_DOMAIN_FOLDERS = ['plots']

Domain = namedtuple(
    'NamedTuple',
    ['path', 'name', 'exp_paths', 'save_path']
)

def iterateDomains(experiment_dir: str) -> Generator[Domain, None, None]:
    domains = glob.glob(f'{experiment_dir}/*')
    domains = filter(lambda p: '.' not in p and Path.fileName(p) not in NON_DOMAIN_FOLDERS, domains)
    domains = filter(lambda p: len(glob.glob(f'{p}/*.json')) > 0, domains)

    for domain_path in domains:
        domain_name = Path.fileName(domain_path)

        exp_paths = glob.glob(f'{domain_path}/*.json')
        save_path = f'{experiment_dir}/plots'
        yield Domain(domain_path, domain_name, exp_paths, save_path)

def parseCmdLineArgs():
    path = sys.argv[0]
    path = Path.up(path)
    save = False
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        save = True

    save_type = 'png'
    if len(sys.argv) > 2:
        save_type = sys.argv[2]

    return (path, save, save_type)
