#!/bin/bash

python3 --version

echo "CONTROL: starting runs on test datasets"

declare -r input_dim=4
declare -r trainn=1
declare -r train_pts=2
declare -r SCS=0
declare -r DNN=0
declare -r run_ID="more_testing_data"
declare -r training_ptset="uniform_random"
declare -r testing_ptset="uniform_random" #"uniform_random"  CC_sparse_grid
declare -r example="other" # logKL_expansion
declare -r SCS_precision="double"
declare -r DNN_precision="single"
declare -r mesh_size=32
declare -r SCS_solver="bregman_fpc"
declare -r DNN_optimizer="adam"
declare -r DNN_loss_function="l2"
declare -r DNN_blocktype="default"
declare -r DNN_activation="tanh"
declare -r DNN_batch_size=256
declare -r make_plots=0
declare -r pmax=7
declare -r SG_level=3
declare -r SG_level_1d=13
declare -r max_nb_train_points=5

if [ "$SCS_precision" = "single" ]; then
    declare -r SCS_error_tol="1e-2"
elif [ "$SCS_precision" = "double" ]; then
    declare -r SCS_error_tol="1e-4"
fi

if [ "$DNN_precision" = "single" ]; then
    declare -r DNN_error_tol="5e-7"
elif [ "$DNN_precision" = "double" ]; then
    declare -r DNN_error_tol="5e-7"
fi



# use to set processor affinity
#taskset -c $((2*$j)) 

# pre-generate the testing results 
#python3 -m cProfile -s time FEM_parametric_PDE_example.py --to --time --things.......
python3 FEM_parametric_PDE_example.py --example $example --nb_train_points $train_pts --max_nb_train_points $max_nb_train_points --train_pointset $training_ptset --nb_trials 10 --train $trainn --make_plots $make_plots --test_pointset $testing_ptset --quiet 0 --input_dim $input_dim --trial_num 0  --pmax $pmax --SG_level $SG_level   --run_ID $run_ID --DNN 0 --SCS 0 #&