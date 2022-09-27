#!/bin/bash -l

#$ -P peaclab-mon

#$ -l h_rt=08:00:00

# Node with 256G memory, 16 threads
#$ -l mem_per_core=16G
#$ -pe omp 12

# Merge stderr and stdout to same file
#$ -e job_outputs/
#$ -o job_outputs/
#$ -j y


#Example: qsub submit_active_learning.sh eclipse mvts 2000 uncertainty 0 0 250 rf
source /project/peaclab-mon/tsfresh_monitoring_venv.sh
chmod +x $HOME/projectx/AI4HPCAnalytics/src/active_learning_experiments/scripts/investigate_active_learning.py
$HOME/projectx/AI4HPCAnalytics/src/active_learning_experiments/scripts/investigate_active_learning.py $1 $2 $3 $4 $5 $6 $7 $8
