#!/bin/bash -l

#$ -P peaclab-mon

#$ -l h_rt=18:00:00

# Node with 256G memory, 16 threads
#$ -l mem_per_core=16G
#$ -pe omp 16

# Merge stderr and stdout to same file
#$ -e unseen_apps_job_outputs/
#$ -o unseen_apps_job_outputs/
#$ -j y


#Example: qsub submit_unseen_apps_active_learning.sh volta mvts 2000 uncertainty 0 0 250 rf 2 4
#It means training data will have 4 knowns and it will include all possible combinations
source /project/peaclab-mon/monitoring_venv.sh
chmod +x $HOME/projectx/AI4HPCAnalytics/src/active_learning_experiments/scripts/unseen_apps_active_learning.py
$HOME/projectx/AI4HPCAnalytics/src/active_learning_experiments/scripts/unseen_apps_active_learning.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}
