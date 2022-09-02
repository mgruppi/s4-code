#!/bin/bash -x
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate ~/barn/s4-env/


#Slurm Directives
#SBATCH --mail-user=gouvem@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH -n 1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

#source ~/miniconda3/etc/profile.d/conda.sh 
#conda activate ~/barn/s4-env/

time=30  # Time in minutes
gpus=1  # No. of gpus
mail_user='gouvem@rpi.edu'
mail_type=end,fail
n=1

srun -t $time --gres=gpu:$gpus --mail-user=$mail_user --mail-type=$mail_type \
	python parameter_search.py r --normalized

#srun -t 30 python parameter_search.py r --normalized
#srun -t 30 python parameter_search.py r --normalized --flip-direction

#salloc -t 100 --gres=gpu:1 srun python parameter_search.py r --normalized
