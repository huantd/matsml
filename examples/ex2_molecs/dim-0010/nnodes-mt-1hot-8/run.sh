#!/usr/bin/env bash
#SBATCH --job-name="d0010n8"
#SBATCH --output="data.%j.%N.out"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --account=ucn109
#SBATCH --export=ALL
#SBATCH -t 48:00:00

#source activate tf
python /home/huantran/appls/bin/deepdl/src/ml.py  ml.in 

