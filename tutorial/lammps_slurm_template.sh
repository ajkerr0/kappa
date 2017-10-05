#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --job-name Z
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH -t 12:00:00

for i in {1..10}; do
RAND=$(echo $RANDOM)  # random initial velocity seed
mpirun lmp_mpi < X -var random $RAND -var iter $i -var name Z
done
