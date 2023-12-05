#!/bin/bash
#
#SBATCH --job-name=training_navi_lstm
#SBATCH --array=0-4
#SBATCH --time=48:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --partition=gpu
#
#SBATCH --mail-user=robin.ghyselinck@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lysmed


# ------------------------- work -------------------------

# Setting the number of worker for data loading
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

# Job 0 is responsible for creating the data base
echo "Starting Task #: $SLURM_ARRAY_TASK_ID"
python main.py --w_b_api_key a64b32e1f56e76998845a8ec40f28c1292986e31 --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID 
echo "Finished Task #: $SLURM_ARRAY_TASK_ID"

echo "Exiting the program."