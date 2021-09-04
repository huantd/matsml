#!/usr/bin/env bash
#SBATCH --job-name="d0010n8"
#SBATCH --output="data.%j.%N.out"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --account=ucn109
#SBATCH --export=ALL
#SBATCH -t 48:00:00

#conda activate tf
for i in 16 32 48 64 80 96 112 128 144 160 176 ; do
    for j in 0 1 2 3 4 5 6 7 8 9 ; do
      echo "  >>>> " ${i} ${j}
      cp ml_saved.in ml.in
      sed -i "s/ml_ntrains_start = 16/ml_ntrains_start = $i/g" ml.in
      python /home/huantran/appls/bin/deepdl/src/ml.py  ml.in 
      cp training.csv training-${i}-${j}.csv
      cp test.csv test-${i}-${j}.csv
    done
done

