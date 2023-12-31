#!/bin/bash

python3 --version


echo "CONTROL: starting runs on test datasets"


#SHARED VARIABLES ; NOT FREQUENLTY CHANGED  
declare -r testing_ptset="CC_sparse_grid" #"uniform_random"  CC_sparse_grid
declare -r SG_level=4  #set to 7


# SHARED PARAMETERS

declare -r UseFEM_tn=0
declare -r UseFEM_ts=1
declare -r UseDnn=0
declare -r FE_degree=0

declare -r DNN_whichfun=0     #Declare which function to approximate 0:_u_; 1:_p_; 2:_u_p_
declare -r prob=1            # options:  "poisson"=0 ; "stokes"=1 ; "NSE"=2 ; "other" =3


declare -A input_dim
input_dim[0]="4"
input_dim[1]="10"
#input_dim[1]="10"

declare -A trials
trials[0]="1"
trials[1]="2"
#trials[2]="3"
#trials[3]="4"
#trials[4]="5"
#trials[5]="6"
#trials[6]="7"
#trials[7]="8"
#trials[8]="9"
#trials[9]="10"

declare -r training_ptset="uniform_random"
declare -r example="other" # logKL_expansion ; other;

# FEM PARAMETERS

declare -r train_pts=5     # change this to the total number of training points TO BE CREATED if trainn =1
declare -r nb_test_points=10 # change this for uniform_random only

#declare -r train_pts=10
#train_pts[0]="5"
#train_pts[1]="10"

# DNN PARAMETERS
declare -A DNN_nb_trainingpts # ->  so it has to be greater than 5
DNN_nb_trainingpts[0]="5"
DNN_nb_trainingpts[1]="150"
DNN_nb_trainingpts[2]="225"
DNN_nb_trainingpts[3]="300"
DNN_nb_trainingpts[4]="375"
DNN_nb_trainingpts[5]="450"
DNN_nb_trainingpts[6]="525"
DNN_nb_trainingpts[7]="600"
DNN_nb_trainingpts[8]="675"
DNN_nb_trainingpts[9]="750"
DNN_nb_trainingpts[10]="825"
DNN_nb_trainingpts[11]="900"
DNN_nb_trainingpts[12]="975"
DNN_nb_trainingpts[13]="1050"
DNN_nb_trainingpts[14]="1125"
DNN_nb_trainingpts[15]="1200"
DNN_nb_trainingpts[16]="1275"
DNN_nb_trainingpts[17]="1350"
DNN_nb_trainingpts[18]="1425"
DNN_nb_trainingpts[19]="1500"
declare -A DNN_activation
DNN_activation[0]="tanh"
DNN_activation[1]="LeakyReLU"
#DNN_activation[2]="relu"
declare -A DNN_nb_layers
DNN_nb_layers[0]="3"
DNN_nb_layers[1]="4"
DNN_nb_layers[2]="5"
DNN_nb_layers[3]="10"
declare -r Use_batching=0 # If not batch Use_batching=0 ; Otherwise  Use_batching=5 -> BATCH_SIZE   = int(m/5)

declare -r DNN_precision="single"
declare -r DNN_optimizer="adam"
declare -r DNN_loss_function="l2"
declare -r DNN_blocktype="default"
declare -r DNN_initializer="uniform"
declare -r DNN_lrn_rate_schedule="exp_decay"
declare -r DNN_type_loss="customize"
declare -r DNN_show_epoch=1000
declare -r DNN_test_epoch=10000
declare -r DNN_total_trials=1 # total number of trials, must match the number of trials below



declare -r make_plots=0
declare -r pmax=7


if [ "$DNN_whichfun" = 0 ]; then
    declare -r whichfun="_u_" 
elif [ "$DNN_whichfun" = 1 ]; then
    declare -r whichfun="_p_"  
elif [ "$DNN_whichfun" = 2 ]; then
    declare -r whichfun="_u_p_"  
fi



if [ "$prob" = 0 ]; then
    declare -r problem="poisson" 
elif [ "$prob" = 1 ]; then
    declare -r problem="NSB"  
fi


# precision and tolerance setting
if [ "$DNN_precision" = "single" ]; then
    declare -r DNN_epochs="10001"
    declare -r DNN_error_tol="5e-7"
elif [ "$DNN_precision" = "double" ]; then
    declare -r DNN_epochs="200000"
    declare -r DNN_error_tol="5e-16"
fi





max_trial=0  # How many Trials
max_dim=0    # How many dimensions 
max_trp=0    # How many training points sets
max_ratio=0  # Number of set of Nodes
max_fun=0    # How many activation functions: tanh;relu

for (( i=0; i <= $max_trial; ++i ))
do
    if [ "$UseFEM_tn" = 1 ]; then
        echo "CONTROL: beginning ${trials[$i]} point trials - TRAINING"
        declare -r run_ID="training_data"  
        for (( j=0; j <= $max_dim; ++j ))
        do
            python3 FEM_parametric_PDE_example.py --example $example --FE_degree $FE_degree --problem $problem --nb_train_points $train_pts --nb_test_points $nb_test_points  --train_pointset $training_ptset --train $UseFEM_tn --test_pointset $testing_ptset  --input_dim ${input_dim[$j]} --trial_num ${trials[$i]}   --SG_level $SG_level   --run_ID $run_ID #&
        done
    fi
done



for (( p=0; p <= $max_dim; ++p ))
do
    
    if [ "$UseFEM_ts" = 1 ]; then
        echo "CONTROL: beginning ${trials[$i]} point trials - TESTING" 
        declare -r run_ID2="testing_data" 
        python3 FEM_parametric_PDE_example.py --example $example --FE_degree $FE_degree --problem $problem --nb_train_points $train_pts --nb_test_points $nb_test_points  --train_pointset $training_ptset --train 0 --test_pointset $testing_ptset --input_dim ${input_dim[$p]} --trial_num ${trials[$"0"]}   --SG_level $SG_level   --run_ID $run_ID2 #&
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
                        python3 TF_parametric_PDE_2FUN.py  --example $example --Use_batching $Use_batching --DNN_show_epoch $DNN_show_epoch --DNN_test_epoch $DNN_test_epoch --DNN_max_nb_train_pts $train_pts --FE_degree $FE_degree --problem $problem --nb_train_points ${DNN_nb_trainingpts[$j]} --nb_test_points $nb_test_points  --train_pointset $training_ptset --test_pointset $testing_ptset --input_dim ${input_dim[$p]}  --trial_num ${trials[$i]}   --SG_level $SG_level   --run_ID $run_ID3 --DNN_activation ${DNN_activation[$l]}  --DNN_precision $DNN_precision --DNN_error_tol $DNN_error_tol --DNN_optimizer $DNN_optimizer --DNN_loss_function $DNN_loss_function --DNN_blocktype $DNN_blocktype --DNN_initializer $DNN_initializer --DNN_lrn_rate_schedule $DNN_lrn_rate_schedule --DNN_type_loss $DNN_type_loss --DNN_epochs $DNN_epochs --DNN_nb_layers ${DNN_nb_layers[$k]} --whichfun $whichfun #&
                    done
                done
            done
        done
    fi
done
 

 


    
 

#python3 FEM_parametric_PDE_example.py 
#--run_ID $run_ID --example $example --problem $problem 
#--train_pointset $training_ptset --nb_train_points $train_pts --test_pointset $testing_ptset 
#--nb_test_points $nb_test_points --FE_degree $FE_degree 
#--SG_level $SG_level --trial_num ${trials[$i]}  
#--input_dim ${input_dim[$j]} --train $UseFEM_tn 



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
# 10 =  $trials
# 11 =  $input_dim


## Args: DNN
# 12 =  $DNN_activation
# 13 =  $DNN_precision
# 14 =  $DNN_error_tol
# 15 =  $DNN_optimizer   
# 16 =  $DNN_loss_function
# 17 =  $DNN_blocktype
# 18 =  $DNN_initializer
# 19 =  $DNN_lrn_rate_schedule
# 20 =  $DNN_type_loss
# 21 =  $DNN_epochs

# 22 =  $DNN_total_trials
# 23 =  $DNN_nb_layers
# 24 =  $DNN_nb_trainingpts
# 25 =  $whichfun   
# 26 =  $DNN_show_epoch
# 27 =  $DNN_test_epoch
# 28 =  $Use_batching
# 29 =  $nb_trials
# 30 =  GPU
# 31 =  train

#python3 TF_parametric_PDE_2FUN.py 
#--run_ID $run_ID3 --example $example
#--problem $problem --train_pointset $training_ptset
#--DNN_max_nb_train_pts $train_pts --test_pointset $testing_ptset
#--nb_test_points $nb_test_points --FE_degree $FE_degree 
#--SG_level $SG_level --trial_num ${trials[$i]}  
#--input_dim ${input_dim[$p]} 

#--DNN_activation ${DNN_activation[$l]}  
#--DNN_precision $DNN_precision --DNN_error_tol $DNN_error_tol 
#--DNN_optimizer $DNN_optimizer --DNN_loss_function $DNN_loss_function 
#--DNN_blocktype $DNN_blocktype --DNN_initializer $DNN_initializer 
#--DNN_lrn_rate_schedule $DNN_lrn_rate_schedule --DNN_type_loss $DNN_type_loss 
#--DNN_epochs $DNN_epochs --DNN_total_trials $DNN_total_trials 
#--DNN_nb_layers ${DNN_nb_layers[$k]} 
#--nb_train_points ${DNN_nb_trainingpts[$j]} 
#--whichfun $whichfun --DNN_show_epoch $DNN_show_epoch 
#--DNN_test_epoch $DNN_test_epoch --Use_batching $Use_batching --nb_trials 10 

 


  



 


#&




 










 