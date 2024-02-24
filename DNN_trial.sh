#!/bin/bash

hostname
module load
module load cuda cudnn
source /home/semoraga/fenics/bin/activate
source /home/semoraga/fenics/share/dolfin/dolfin.conf
module load gcc/9.3.0
module load intel/2020.1.217  intelmpi/2019.7.217 hdf5-mpi/1.10.6
module load boost/1.72.0
module load eigen
module load scipy-stack/2023b
module load gcc/9.3.0  openmpi/4.0.3 petsc/3.17.1
module load fftw-mpi/3.3.8
module load ipp/2020.1.217
module load swig
module load flexiblas
module load intel/2020.1.217  cuda/11.4  openmpi/4.0.3
module load mpi4py/3.1.3



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
# 28 = GPU

CUDA_VISIBLE_DEVICES=${28}
echo "CONTROL: DNN ${22} point trials"

python3 TF_parametric_PDE_2FUN.py --run_ID $1 --example $2 --problem $3 --train_pointset $4 --DNN_max_nb_train_pts $5 --test_pointset $6 --nb_test_points $7 --FE_degree $8 --SG_level $9 --input_dim ${10} --DNN_activation ${11} --DNN_precision ${12} --DNN_error_tol ${13} --DNN_optimizer ${14} --DNN_loss_function ${15} --DNN_blocktype ${16} --DNN_initializer ${17} --DNN_lrn_rate_schedule ${18} --DNN_type_loss ${19} --DNN_epochs ${20} --DNN_nb_layers ${21} --nb_train_points ${22} --whichfun ${23} --DNN_show_epoch ${24} --DNN_test_epoch ${25} --Use_batching ${26} --trial_num ${27}  
  



 