#!/bin/bash -vv

#SBATCH --account=def-adcockb
#SBATCH --time=5-0
#SBATCH --mem-per-cpu=5024M
#SBATCH --array=1-4




## Args: FEM
# 1 =  $run_ID
# 2 =  $example
# 3 =  $problem
# 4 =  $training_ptset
# 5 =  $train_pts #total number of train
# 6 =  $testing_ptset
# 7 =  $nb_test_points
# 8 =  $FE_degree
# 9 =  $SG_level
# 10 =  $input_dim
# 11 =  $UseFEM_tn
# 12 = trial_1 + trial_2 +trial_3 + trial_4  + CPU


echo "CONTROL: training $5 point trials"

bash FEM_trial.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} $(($SLURM_ARRAY_TASK_ID+0)) > run_out/P_$1_coeff_$2_$3_dim_${10}_SG_$9_trial_$(($SLURM_ARRAY_TASK_ID+0))_sched_jobid_$((SLURM_JOB_ID)).out &
bash FEM_trial.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} $(($SLURM_ARRAY_TASK_ID+4)) > run_out/P_$1_coeff_$2_$3_dim_${10}_SG_$9_trial_$(($SLURM_ARRAY_TASK_ID+4))_sched_jobid_$((SLURM_JOB_ID)).out &
bash FEM_trial.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} $(($SLURM_ARRAY_TASK_ID+8)) > run_out/P_$1_coeff_$2_$3_dim_${10}_SG_$9_trial_$(($SLURM_ARRAY_TASK_ID+8))_sched_jobid_$((SLURM_JOB_ID)).out &
wait

 