#!/bin/bash

python3 --version

echo "CONTROL: starting runs on test datasets"

# SHARED PARAMETERS
declare -r training_ptset="uniform_random"
declare -r testing_ptset="CC_sparse_grid" #"uniform_random"  CC_sparse_grid
declare -r example="other" # logKL_expansion ; other;
declare -r SG_level=1
declare -r max_nb_train_points=5
declare -r mesh_num=2
declare -r FE_degree=1
declare -r input_dim=10

# FEM PARAMETERS
declare -r trainn=0       # options:  "testing_data"=0 ; "training_data"=1 ; "result_data"=2
declare -r train_pts=20    # change this to the total number of training points TO BE CREATED if trainn =1
declare -r nb_test_points=20 # change this for uniform_random only
declare -r prob=0            # options:  "poisson"=0 ; "stokes"=1 ; "NSE"=2 ; "other" =3

# DNN PARAMETERS
declare -r UseDnn=1
declare -r DNN_nb_trainingpts=20
declare -r DNN_precision="single"
declare -r DNN_optimizer="adam"
declare -r DNN_loss_function="l2"
declare -r DNN_blocktype="default"
declare -r DNN_activation="tanh"
declare -r DNN_initializer="uniform"
declare -r DNN_lrn_rate_schedule="exp_decay"
declare -r DNN_type_loss="customize"
declare -r DNN_epochs=1000
declare -r DNN_show_epoch=10
declare -r DNN_test_epoch=50
declare -r DNN_total_trials=1
declare -r DNN_nb_layers=5
declare -r DNN_max_nb_train_pts=5 # number of points to be used in the training <= nb_test_points
declare -r make_plots=0
declare -r pmax=7



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
    declare -r DNN_error_tol="5e-4"
elif [ "$DNN_precision" = "double" ]; then
    declare -r DNN_error_tol="5e-8"
fi



# pre-generate the testing results 
#python3 -m cProfile -s time FEM_parametric_PDE_example.py --to --time --things.......
if [ "$UseDnn" = 0 ]; then
    if [ "$trainn" = 0 ]; then
        declare -r run_ID="testing_data" 
    elif [ "$trainn" = 1 ]; then
        declare -r run_ID="training_data"  
    fi
    python3 FEM_parametric_PDE_example.py --example $example --FE_degree $FE_degree --mesh_num $mesh_num --problem $problem --nb_train_points $train_pts --nb_test_points $nb_test_points --max_nb_train_points $max_nb_train_points --train_pointset $training_ptset --nb_trials 10 --train $trainn --make_plots $make_plots --test_pointset $testing_ptset --quiet 0 --input_dim $input_dim --trial_num 0   --SG_level $SG_level   --run_ID $run_ID #&
elif [ "$UseDnn" = 1 ]; then
    declare -r run_ID="result_data"
    python3 TF_parametric_PDE.py  --example $example --DNN_show_epoch $DNN_show_epoch --DNN_test_epoch $DNN_test_epoch --DNN_max_nb_train_pts $train_pts --FE_degree $FE_degree --mesh_num $mesh_num --problem $problem --nb_train_points $DNN_nb_trainingpts --nb_test_points $nb_test_points --max_nb_train_points $max_nb_train_points --train_pointset $training_ptset --nb_trials 10 --train $trainn --make_plots $make_plots --test_pointset $testing_ptset --quiet 0 --input_dim $input_dim --trial_num 0   --SG_level $SG_level   --run_ID $run_ID --DNN_activation $DNN_activation --DNN_precision $DNN_precision --DNN_error_tol $DNN_error_tol --DNN_optimizer $DNN_optimizer --DNN_loss_function $DNN_loss_function --DNN_blocktype $DNN_blocktype --DNN_initializer $DNN_initializer --DNN_lrn_rate_schedule $DNN_lrn_rate_schedule --DNN_type_loss $DNN_type_loss --DNN_epochs $DNN_epochs --DNN_total_trials $DNN_total_trials --DNN_nb_layers $DNN_nb_layers #&
fi
 










 