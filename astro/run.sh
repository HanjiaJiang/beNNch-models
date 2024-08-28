#!/bin/bash -x
#SBATCH --account=jinb33
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=dc-cpu
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err

# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

source $PROJECT/jiang4/benchmark/env.sh
#module load nest-simulator/3.7/default

# Path to python script
pwd=$PWD
script=$pwd/$1

export NUMEXPR_MAX_THREADS=128
export OMP_DISPLAY_ENV=VERBOSE
export OMP_DISPLAY_AFFINITY=TRUE
export OMP_PROC_BIND=TRUE

export OMP_NUM_THREADS=16

# Bind by threads
srun --cpus-per-task=16 --threads-per-core=1 --cpu-bind=verbose,threads --distribution=block:cyclic:fcyclic python3 $script 16
