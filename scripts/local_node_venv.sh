#!/bin/bash

#SBATCH --time=00:15:00
#SBATCH --account=rrg-whitem

cp $path/requirements.txt $SLURM_TMPDIR/
cd $SLURM_TMPDIR
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
tar -cavf venv.tar.xz .venv
cp venv.tar.xz $path/
