#! /bin/bash

module load python/3.11

# make sure home folder has a venv
if [ ! -d "~/.venv" ]; then
  python -m venv ~/.venv
fi

source ~/.venv
pip install PyExpUtils-andnp

sbatch salloc --time=0:15:0 --ntasks=1 --mem-per-cpu="4G" --account=rrg-whitem --export=path="$(pwd)" scripts/local_node_venv.sh
