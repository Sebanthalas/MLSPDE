#!/bin/bash -vv

python --version

echo "CONTROL: starting runs on test datasets"

# Run parameters
# The following are boolean variables "1" True "0" False

declare -r UseFEM_tn=1      # Generate training points = 1 ; no training points = 0
declare -r UseFEM_ts=1      # Generate testing  points = 1 ; no testing points  = 0
declare -r UseDnn=1         # Train/Test DNN           = 1 ; no Train/Test DNN  = 0

# Which problem you want to approximate?
# Options: NSB"=1 ; 
declare -r prob=1 

# Declare which function to approximate 
#  DNN_whichfun = 0 -> "_u_"; 
#  DNN_whichfun = 1 -> "_p_"; 
declare -r DNN_whichfun=0  
          

# FEM degree & testing points
declare -r FE_degree=1
declare -r testing_ptset="CC_sparse_grid" #  "uniform_random" ;  "CC_sparse_grid"


# Set up the Training; Testing; and DNN usage
# This will go to the .py 
if [ "$UseFEM_tn" = 1 ]; then
    declare -r run_ID="training_data"
else
    declare -r run_ID="no"
fi

if [ "$UseFEM_ts" = 1 ]; then
    declare -r run_ID2="testing_data"
else
    declare -r run_ID2="no"
fi

if [ "$UseDnn" = 1 ]; then
    declare -r run_ID3="result_data"
else
    declare -r run_ID3="no"
fi


declare -r nb_test_points=10 # Change this for uniform_random only; if not, it does not matter.
declare -r training_ptset="uniform_random"
declare -r Use_batching=0   # If not batch Use_batching=0 ; Otherwise Use_batching=5 -> BATCH_SIZE = int(m/5)

declare -r DNN_precision="single"
declare -r DNN_optimizer="adam"
declare -r DNN_loss_function="l2"
declare -r DNN_blocktype="default"
declare -r DNN_initializer="he_uniform"
declare -r DNN_lrn_rate_schedule="exp_decay"
declare -r DNN_type_loss="customize"
declare -r DNN_show_epoch=1000
declare -r DNN_test_epoch=10000
declare -r DNN_total_trials=20 # Total number of trials, must match the number of trials below

declare -A DNN_nb_layers
DNN_nb_layers[0]="4"
DNN_nb_layers[1]="10"


declare -A DNN_activation
DNN_activation[0]="elu"
DNN_activation[1]="relu"
DNN_activation[2]="tanh"

# precision and tolerance setting
if [ "$DNN_precision" = "single" ]; then
    declare -r DNN_epochs="60001"
    declare -r DNN_error_tol="5e-8"
elif [ "$DNN_precision" = "double" ]; then
    declare -r DNN_epochs="200001"
    declare -r DNN_error_tol="5e-16"
fi

# Declare which problem to approximate
if [ "$prob" = 1 ]; then
    declare -r problem="NSB"
    # DNN PARAMETERS
    declare -A DNN_nb_trainingpts # ->  Each has to be less or equal to "train_pts" in line  189 for NSB
    DNN_nb_trainingpts[0]="30"
    DNN_nb_trainingpts[1]="60"
    DNN_nb_trainingpts[2]="90"
    DNN_nb_trainingpts[3]="120"
    DNN_nb_trainingpts[4]="150"
    DNN_nb_trainingpts[5]="180"
    DNN_nb_trainingpts[6]="210"
    DNN_nb_trainingpts[7]="240"
    DNN_nb_trainingpts[8]="270"
    DNN_nb_trainingpts[9]="300"
    DNN_nb_trainingpts[10]="330"
    DNN_nb_trainingpts[11]="360"
    DNN_nb_trainingpts[12]="390"
    DNN_nb_trainingpts[13]="420"
    DNN_nb_trainingpts[14]="450"
    DNN_nb_trainingpts[15]="480"
    DNN_nb_trainingpts[16]="510"
    DNN_nb_trainingpts[17]="540"
    DNN_nb_trainingpts[18]="570"
    DNN_nb_trainingpts[19]="600" 
fi

# Declare which function to approximate "u" or "p" with a DNN
if [ "$DNN_whichfun" = 0 ]; then
    declare -r whichfun="_u_" 
elif [ "$DNN_whichfun" = 1 ]; then
    declare -r whichfun="_p_"  
fi

# Iteration loop over the affine coefficients
for i in {1..2..1}
do
    # The following tells which affine coefficient goes to the code
    if [ "$i" -eq "0" ]; then
        declare  example="aff_F9" # affine from .pdf eq (5.3) (Faster decay)
    elif [ "$i" -eq "1" ]; then
        declare  example="aff_S3" # affine from .pdf eq (5.4) (Slow decay)
    elif [ "$i" -eq "2" ]; then
        declare  example="logKL" # logKL_expansion ;  
    else
        echo "must be 0 through 2"
    fi



    # main iteration loop over the dimensions
    for d in {0..1..1}
    do
        if [ "$d" -eq "0" ]; then
            declare input_dim=4
        elif [ "$d" -eq "1" ]; then
            declare input_dim=8
        else
            echo "must be 0 through 2"
        fi
        # If problem = possion choose how many training points "train_pts" and SG level for each dimension
        if [ "$prob" = 1 ]; then
            declare  train_pts=500  # ~0.3G each set of 500
            if [ "$input_dim" -eq "4" ]; then
                declare  SG_level=5  #1105  points 8.5G
            elif [ "$input_dim" -eq "8" ]; then
                declare  SG_level=4  #849  points 6.5G
            else
                echo "input dim is 4,8,16"
            fi
        else
            echo "Problem must be NSB"
        fi


        ############## WARNING ! ##############################
        for trialss in {1..12..1}
        do
            if [ "$UseFEM_tn" = 1 ]; then
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

                if [ "$i" -eq "0" ]; then
                    # Affine coefficient "aff_F9"
                    python3 FEM_parametric_PDE_example.py --run_ID $run_ID --example $example --problem $problem --train_pointset $training_ptset --nb_train_points $train_pts --test_pointset $testing_ptset --nb_test_points $nb_test_points --FE_degree $FE_degree --SG_level $SG_level --input_dim $input_dim --train $UseFEM_tn --trial_num $trialss 
                    #sbatch NTF.sh $run_ID $example $problem $training_ptset $train_pts $testing_ptset $nb_test_points $FE_degree $SG_level $input_dim $UseFEM_tn   #--trial_num 1 
                elif [ "$i" -eq "1" ]; then
                    # Affine coefficient "aff_S3"
                    python3 FEM_parametric_PDE_example.py --run_ID $run_ID --example $example --problem $problem --train_pointset $training_ptset --nb_train_points $train_pts --test_pointset $testing_ptset --nb_test_points $nb_test_points --FE_degree $FE_degree --SG_level $SG_level --input_dim $input_dim --train $UseFEM_tn --trial_num $trialss 
                    #sbatch NTS.sh $run_ID $example $problem $training_ptset $train_pts $testing_ptset $nb_test_points $FE_degree $SG_level $input_dim $UseFEM_tn   #--trial_num 1 
                elif [ "$i" -eq "2" ]; then
                    # Affine coefficient "logKL"
                    python3 FEM_parametric_PDE_example.py --run_ID $run_ID --example $example --problem $problem --train_pointset $training_ptset --nb_train_points $train_pts --test_pointset $testing_ptset --nb_test_points $nb_test_points --FE_degree $FE_degree --SG_level $SG_level --input_dim $input_dim --train $UseFEM_tn --trial_num $trialss 
                    #sbatch NTLK.sh $run_ID $example $problem $training_ptset $train_pts $testing_ptset $nb_test_points $FE_degree $SG_level $input_dim $UseFEM_tn  #--trial_num 1 
                else
                    echo "must be 0 through 2"
                fi
            else
                echo "No training" 
                #process_idTr=0 
            fi
        done  

        if [ "$UseFEM_ts" = 1 ]; then 
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
            # 11 =  $UseFEM_tn = 0 (needs to be 0 because it is only testing)
            if [ "$i" -eq "0" ]; then
                # Affine coefficient "aff_F9"
                python3 FEM_parametric_PDE_example.py --run_ID $run_ID2 --example $example --problem $problem --train_pointset $training_ptset --nb_train_points $train_pts --test_pointset $testing_ptset --nb_test_points $nb_test_points --FE_degree $FE_degree --SG_level $SG_level --input_dim $input_dim --train 0 --trial_num 1 
                #sbatch NTeF.sh $run_ID2 $example $problem $training_ptset $train_pts $testing_ptset $nb_test_points $FE_degree $SG_level $input_dim 0  #--trial_num 1 
            elif [ "$i" -eq "1" ]; then
                # Affine coefficient "aff_S3"
                python3 FEM_parametric_PDE_example.py --run_ID $run_ID2 --example $example --problem $problem --train_pointset $training_ptset --nb_train_points $train_pts --test_pointset $testing_ptset --nb_test_points $nb_test_points --FE_degree $FE_degree --SG_level $SG_level --input_dim $input_dim --train 0 --trial_num 1 
                #sbatch NTeS.sh $run_ID2 $example $problem $training_ptset $train_pts $testing_ptset $nb_test_points $FE_degree $SG_level $input_dim 0  #--trial_num 1 
            elif [ "$i" -eq "2" ]; then
                # Affine coefficient "logKL"
                python3 FEM_parametric_PDE_example.py --run_ID $run_ID2 --example $example --problem $problem --train_pointset $training_ptset --nb_train_points $train_pts --test_pointset $testing_ptset --nb_test_points $nb_test_points --FE_degree $FE_degree --SG_level $SG_level --input_dim $input_dim --train 0 --trial_num 1 
                #sbatch NTeLK.sh $run_ID2 $example $problem $training_ptset $train_pts $testing_ptset $nb_test_points $FE_degree $SG_level $input_dim 0  #--trial_num 1 
            else
                echo "must be 0 through 2"
            fi     
        else
            echo "No Testing" 
        fi

        if [ "$UseDnn" = 1 ]; then
            for l in {0..2..1} # {DNN_activation[$l]} 
            do
                for k in {0..1..1} # {DNN_nb_layers[$k]}
                do
                    for j in {0..15..1} #{DNN_nb_trainingpts[$j]}
                    do
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
                        ## Args: DNN
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
                        python3 TF_parametric_PDE_2FUN.py --run_ID $run_ID3 --example $example --problem $problem --train_pointset $training_ptset --DNN_max_nb_train_pts $train_pts --test_pointset $testing_ptset --nb_test_points $nb_test_points --FE_degree $FE_degree --SG_level $SG_level --input_dim $input_dim --DNN_activation ${DNN_activation[$l]} --DNN_precision $DNN_precision --DNN_error_tol $DNN_error_tol --DNN_optimizer $DNN_optimizer --DNN_loss_function $DNN_loss_function --DNN_blocktype $DNN_blocktype --DNN_initializer $DNN_initializer --DNN_lrn_rate_schedule $DNN_lrn_rate_schedule --DNN_type_loss $DNN_type_loss --DNN_epochs $DNN_epochs  --DNN_nb_layers ${DNN_nb_layers[$k]} --nb_train_points ${DNN_nb_trainingpts[$j]} --whichfun $whichfun --DNN_show_epoch $DNN_show_epoch --DNN_test_epoch $DNN_test_epoch --Use_batching $Use_batching --trial_num 1
                        echo "Training the DNN" 
                    done
                done
            done  
        else
            echo "No DNN"  
        fi
    done
done

 
 