#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0-00:15:00

# Load required modules
module load Python/3.7.3-spartan_gcc-8.1.0 

# Launch application on 1 node and 1 core.
echo 'COMP90024 Assignment 1, 1 node and 8 cores.'
time mpirun -np 8 python3.7 assignment1mpi.py bigTwitter.json > output_1node_8cores.txt