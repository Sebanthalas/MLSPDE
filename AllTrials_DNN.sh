#!/bin/bash -vv

#SBATCH --nodes=1               # number of nodes to use
#SBATCH --ntasks-per-node=4     # number of tasks
#SBATCH --exclusive             # run on whole node
#SBATCH --cpus-per-task=6       # There are 24 CPU cores on Cedar p100 GPU nodes
#SBATCH --gres=gpu:p100:4       # number of GPUs to use
#SBATCH --mem=0                 # memory per node (0 = use all of it)
#SBATCH --time=00:60:00         # time (DD-HH:MM)
#SBATCH --account=def-adcockb





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
# 11 =  ${DNN_activation[$l]}
# 12 =  $DNN_precision
# 13 =  $DNN_error_tol
# 14 =  $DNN_optimizer   
# 15 =  $DNN_loss_function
# 16 =  $DNN_blocktype
# 17 =  $DNN_initializer
# 18 =  $DNN_lrn_rate_schedule
# 19 =  $DNN_type_loss
# 20 =  $DNN_epochs
# 21 =  ${DNN_nb_layers[$k]} 
# 22 =  ${DNN_nb_trainingpts[$j]} 
# 23 =  $whichfun   
# 24 =  $DNN_show_epoch
# 25 =  $DNN_test_epoch
# 26 =  $Use_batching

# 27 = Trail num -> specified in next bash
# 28 =  GPU

# In the real long scale numerical results  take the comments of the other lines

# for each set of four trials (for running on lgpu or base gpu)
if [ "$3" = "poisson" ]; then
    for j in {1..1..1}
    do
        bash DNN_trial.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} $(($j+0)) 0 > run_out/DNN$1_${25}_${17}_${23}x${23}0_$5_pts_trial_$(($j+0))_${10}_dim_${24}_sched_jobid_$((SLURM_JOB_ID)).out &
        bash DNN_trial.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} $(($j+1)) 1 > run_out/DNN$1_${25}_${17}_${23}x${23}0_$5_pts_trial_$(($j+1))_${10}_dim_${24}_sched_jobid_$((SLURM_JOB_ID)).out &
        bash DNN_trial.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} $(($j+2)) 2 > run_out/DNN$1_${25}_${17}_${23}x${23}0_$5_pts_trial_$(($j+2))_${10}_dim_${24}_sched_jobid_$((SLURM_JOB_ID)).out &
        bash DNN_trial.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} $(($j+3)) 3 > run_out/DNN$1_${25}_${17}_${23}x${23}0_$5_pts_trial_$(($j+3))_${10}_dim_${24}_sched_jobid_$((SLURM_JOB_ID)).out &
        wait
    done

elif [ "$3" = "NSB" ]; then
    for j in {1..1..1}
    do
        bash DNN_trial.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} $(($j+0)) 0 > run_out/DNN$1_${25}_${17}_${23}x${23}0_$5_pts_trial_$(($j+0))_${10}_dim_${24}_sched_jobid_$((SLURM_JOB_ID)).out &
        bash DNN_trial.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} $(($j+1)) 1 > run_out/DNN$1_${25}_${17}_${23}x${23}0_$5_pts_trial_$(($j+1))_${10}_dim_${24}_sched_jobid_$((SLURM_JOB_ID)).out &
        bash DNN_trial.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} $(($j+2)) 2 > run_out/DNN$1_${25}_${17}_${23}x${23}0_$5_pts_trial_$(($j+2))_${10}_dim_${24}_sched_jobid_$((SLURM_JOB_ID)).out &
        bash DNN_trial.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} $(($j+3)) 3 > run_out/DNN$1_${25}_${17}_${23}x${23}0_$5_pts_trial_$(($j+3))_${10}_dim_${24}_sched_jobid_$((SLURM_JOB_ID)).out &
        wait
    done
fi

 