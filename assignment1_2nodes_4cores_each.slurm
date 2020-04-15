#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=0-02:00:00

# Load required modules
module load Python/3.7.3-spartan_gcc-8.1.0 

# Launch application on 1 node and 1 core.
echo 'COMP90024 Assignment 1, 2 nodes and 4 cores each.'
time mpirun python3.7 assignment1mpi.py bigTwitter.json