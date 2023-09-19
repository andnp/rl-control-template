#!/bin/bash

module load python/3.11

# make sure home folder has a venv
if [ ! -d ~/.venv ]; then
  echo "making a new virtual env in ~/.venv"
  python -m venv ~/.venv
fi

source ~/.venv/bin/activate
echo "installing PyExpUtils"
pip install PyExpUtils-andnp

echo "scheduling a job to install project dependencies"
sbatch --ntasks=1 --mem-per-cpu="12G" --export=path="$(pwd)" scripts/local_node_venv.sh
