#!/bin/bash -l

#$ -P peaclab-mon

#$ -l h_rt=04:00:00

# Node with 256G memory, 16 threads
#$ -l mem_per_core=8G
#$ -pe omp 4

# Merge stderr and stdout to same file
#$ -e unseen_inputs_job_outputs/
#$ -o unseen_inputs_job_outputs/
#$ -j y


#Example: qsub submit_unseen_inputs_active_learning.sh volta tsfresh 2000 uncertainty 0 3 250 rf
source /project/peaclab-mon/monitoring_venv.sh
chmod +x $HOME/projectx/AI4HPCAnalytics/src/active_learning_experiments/scripts/unseen_inputs_active_learning.py
$HOME/projectx/AI4HPCAnalytics/src/active_learning_experiments/scripts/unseen_inputs_active_learning.py $1 $2 $3 $4 $5 $6 $7 $8
