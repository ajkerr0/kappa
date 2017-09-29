#!/usr/bin/env bash
for i in `seq 1 1 40`; do
cp lammps_sim_template.txt cnt'$i'.sim
cp lammps_slurm_template.sh cnt'$i'.slurm
sed -i 's/name equal X/name equal '$i'/g' cnt'$i'.sim
sed -i 's/X/cnt'$i'.sim/g' cnt'$i'.slurm
sed -i 's/Z/cnt'$i'/g' cnt'$i'.slurm
sbatch cnt'$i'.slurm
done