#!/bin/bash -vv

#SBATCH --account=def-adcockb
#SBATCH --time=4-0
#SBATCH --mem-per-cpu=4024M





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


echo "CONTROL: testing $5 point trials"

 

# for each set of four trials (for running on lgpu or base gpu)
bash FEM_trial.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} 1 > run_out/Te_$1_coeff_$2_$3_SG_$9_dim_${10}_sched_jobid_$((SLURM_JOB_ID)).out

 