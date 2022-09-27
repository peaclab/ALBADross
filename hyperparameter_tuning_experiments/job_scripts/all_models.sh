#!/bin/bash -l

#$ -P peaclab-mon

#$ -l h_rt=12:00:00

# Node with 256G memory, 16 threads
#$ -l mem_per_core=16G
#$ -pe omp 16

# Merge stderr and stdout to same file
#$ -e job_outputs/
#$ -o job_outputs/
#$ -j y

source /project/peaclab-mon/mvts_monitoring_venv.sh
chmod +x $HOME/projectx/AI4HPCAnalytics/src/hyperparameter_tuning_experiments/scripts/all_models.py
$HOME/projectx/AI4HPCAnalytics/src/hyperparameter_tuning_experiments/scripts/all_models.py $1 $2 $3
