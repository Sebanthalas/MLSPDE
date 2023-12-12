#!/bin/bash

python3 --version

echo "CONTROL: starting runs on test datasets"

# SHARED PARAMETERS

declare -r UseFEM_tn=0
declare -r UseFEM_ts=0
declare -r UseDnn=1
declare -r training_ptset="uniform_random"
declare -r testing_ptset="CC_sparse_grid" #"uniform_random"  CC_sparse_grid
declare -r example="other" # logKL_expansion ; other;
declare -r SG_level=5  #set to 7
declare -r mesh_num=2
declare -r FE_degree=1
declare -A input_dim
input_dim[0]="5"
#input_dim[1]="10"
#input_dim[1]="10"
declare -A trials
trials[0]="1"
trials[1]="2"
trials[2]="3"
trials[3]="4"
trials[4]="5"
#trials[5]="6"
#trials[6]="7"
#trials[7]="8"
#trials[8]="9"
#trials[9]="10"


# FEM PARAMETERS
     # options:  "testing_data"=0 ; "training_data"=1 ; "result_data"=2
declare -r train_pts=3000     # change this to the total number of training points TO BE CREATED if trainn =1
declare -r nb_test_points=20 # change this for uniform_random only
declare -r prob=0            # options:  "poisson"=0 ; "stokes"=1 ; "NSE"=2 ; "other" =3
#declare -r train_pts=10
#train_pts[0]="5"
#train_pts[1]="10"

# DNN PARAMETERS
declare -A DNN_nb_trainingpts # ->  so it has to be greater than 5
DNN_nb_trainingpts[0]="25"
DNN_nb_trainingpts[1]="50"
DNN_nb_trainingpts[2]="75"
DNN_nb_trainingpts[3]="100"
DNN_nb_trainingpts[4]="125"
DNN_nb_trainingpts[5]="150"
DNN_nb_trainingpts[6]="175"
DNN_nb_trainingpts[7]="200"
DNN_nb_trainingpts[8]="225"
DNN_nb_trainingpts[9]="250"
DNN_nb_trainingpts[10]="275"
DNN_nb_trainingpts[11]="300"
DNN_nb_trainingpts[12]="325"
DNN_nb_trainingpts[13]="350"
DNN_nb_trainingpts[14]="375"
DNN_nb_trainingpts[15]="400"
DNN_nb_trainingpts[16]="425"
DNN_nb_trainingpts[17]="450"
DNN_nb_trainingpts[18]="475"
DNN_nb_trainingpts[19]="500"
declare -A DNN_activation
DNN_activation[0]="tanh"
DNN_activation[1]="relu"
declare -A DNN_nb_layers
DNN_nb_layers[0]="3"
DNN_nb_layers[1]="4"
DNN_nb_layers[2]="5"
DNN_nb_layers[3]="10"
declare -r Use_batching=5 # If not batch Use_batching=0 ; Otherwise  Use_batching=5 -> BATCH_SIZE   = int(m/5)

declare -r DNN_precision="single"
declare -r DNN_optimizer="adam"
declare -r DNN_loss_function="l2"
declare -r DNN_blocktype="default"
declare -r DNN_initializer="uniform"
declare -r DNN_lrn_rate_schedule="exp_decay"
declare -r DNN_type_loss="customize"
declare -r DNN_epochs=40501
declare -r DNN_show_epoch=100
declare -r DNN_test_epoch=13500
declare -r DNN_total_trials=2 # total number of trials, must match the number of trials below
#declare -r DNN_max_nb_train_pts=20 # number of points to be used in the training <= nb_test_points
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





max_trial=1
max_dim=0
max_trp=19
max_ratio=3
max_fun=1
for (( i=0; i <= $max_trial; ++i ))
do
    if [ "$UseFEM_tn" = 1 ]; then
        echo "CONTROL: beginning ${trials[$i]} point trials - TRAINING"
        declare -r run_ID="training_data"  
        for (( j=0; j <= $max_dim; ++j ))
        do
            python3 FEM_parametric_PDE_example.py --example $example --FE_degree $FE_degree --mesh_num $mesh_num --problem $problem --nb_train_points $train_pts --nb_test_points $nb_test_points  --train_pointset $training_ptset --trials $trials --train $UseFEM_tn --make_plots $make_plots --test_pointset $testing_ptset --quiet 0 --input_dim ${input_dim[$j]} --trial_num ${trials[$i]}   --SG_level $SG_level   --run_ID $run_ID #&
        done
    fi
done

for (( p=0; p <= $max_dim; ++p ))
do
    
    if [ "$UseFEM_ts" = 1 ]; then
        echo "CONTROL: beginning ${trials[$i]} point trials - TESTING" 
        declare -r run_ID2="testing_data" 
        python3 FEM_parametric_PDE_example.py --example $example --FE_degree $FE_degree --mesh_num $mesh_num --problem $problem --nb_train_points $train_pts --nb_test_points $nb_test_points  --train_pointset $training_ptset --trials $trials --train 0 --make_plots $make_plots --test_pointset $testing_ptset --quiet 0 --input_dim ${input_dim[$p]} --trial_num ${trials[$"0"]}   --SG_level $SG_level   --run_ID $run_ID2 #&
    fi
done


for (( i=0; i <= $max_trial; ++i ))
do   
    if [ "$UseDnn" = 1 ]; then
        echo "CONTROL: beginning ${trials[$i]} point trials - DNN"
        declare -r run_ID3="result_data"
        for (( p=0; p <= $max_dim; ++p ))       
        do
            for (( j=0; j <= $max_trp; ++j ))
            do
                for (( k=0; k <= $max_ratio; ++k ))
                do
                    for (( l=0; l <= $max_fun; ++l ))
                    do
                        python3 TF_parametric_PDE.py  --example $example --Use_batching $Use_batching --DNN_show_epoch $DNN_show_epoch --DNN_test_epoch $DNN_test_epoch --DNN_max_nb_train_pts $train_pts --FE_degree $FE_degree --mesh_num $mesh_num --problem $problem --nb_train_points ${DNN_nb_trainingpts[$j]} --nb_test_points $nb_test_points  --train_pointset $training_ptset --nb_trials 10 --make_plots $make_plots --test_pointset $testing_ptset --quiet 0 --input_dim ${input_dim[$p]}  --trial_num ${trials[$i]}   --SG_level $SG_level   --run_ID $run_ID3 --DNN_activation ${DNN_activation[$l]}  --DNN_precision $DNN_precision --DNN_error_tol $DNN_error_tol --DNN_optimizer $DNN_optimizer --DNN_loss_function $DNN_loss_function --DNN_blocktype $DNN_blocktype --DNN_initializer $DNN_initializer --DNN_lrn_rate_schedule $DNN_lrn_rate_schedule --DNN_type_loss $DNN_type_loss --DNN_epochs $DNN_epochs --DNN_total_trials $DNN_total_trials --DNN_nb_layers ${DNN_nb_layers[$k]}  #&
                    done
                done
            done
        done
    fi
done
 

 


    
 










 










 