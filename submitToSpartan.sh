!/bin/bash
#SBATCH --ntasks=10
#SBATCH --nodes=10
#SBATCH --cpus-per-task=4
#SBATCH --time=2:0
#SBATCH --job-name="rf"
#SBATCH --output="result"

# module load OpenMPI/1.10.2-GCC-6.2.0
# make
srun ./exec

