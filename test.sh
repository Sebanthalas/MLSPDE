#!/bin/bash

python3 --version

echo "CONTROL: starting runs on test datasets"

declare -r input_dim=4
declare -r trainn=1  # options:  "testing_data"=0 ; "training_data"=1 ; "result_data"=2
declare -r train_pts=10 # change this to the total number of training points if trainn =1
declare -r nb_test_points=3 # change this
declare -r SCS=0
declare -r prob=0 # options:  "poisson"=0 ; "stokes"=1 ; "NSE"=2 ; "other" =3
declare -r training_ptset="uniform_random"
declare -r testing_ptset="uniform_random" #"uniform_random"  CC_sparse_grid
declare -r example="other" # logKL_expansion ; other
declare -r DNN_precision="single"
declare -r DNN_optimizer="adam"
declare -r DNN_loss_function="l2"
declare -r DNN_blocktype="default"
declare -r DNN_activation="tanh"
declare -r DNN_batch_size=256
declare -r make_plots=0
declare -r pmax=7
declare -r SG_level=6
declare -r SG_level_1d=13
declare -r max_nb_train_points=5
declare -r mesh_num=2
declare -r FE_degree=1


if [ "$trainn" = 0 ]; then
    declare -r run_ID="testing_data" 
elif [ "$trainn" = 1 ]; then
    declare -r run_ID="training_data"  
elif [ "$trainn" = 2 ]; then
    declare -r run_ID="result_data" 
fi

if [ "$prob" = 0 ]; then
    declare -r problem="poisson" 
elif [ "$prob" = 1 ]; then
    declare -r problem="stokes"  
elif [ "$prob" = 2 ]; then
    declare -r problem="NSE" 
elif [ "$prob" = 3 ]; then
    declare -r problem="other" 
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
python3 FEM_parametric_PDE_example.py --example $example --FE_degree $FE_degree --mesh_num $mesh_num --problem $problem --nb_train_points $train_pts --nb_test_points $nb_test_points --max_nb_train_points $max_nb_train_points --train_pointset $training_ptset --nb_trials 10 --train $trainn --make_plots $make_plots --test_pointset $testing_ptset --quiet 0 --input_dim $input_dim --trial_num 0  --pmax $pmax --SG_level $SG_level   --run_ID $run_ID --DNN 0 #&