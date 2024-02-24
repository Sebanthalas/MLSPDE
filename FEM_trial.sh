#!/bin/bash


hostname
module load
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
module load mpi4py/3.1.3


## Args: FEM
# 1 =  $run_ID
# 2 =  $example
# 3 =  $problem
# 4 =  $training_ptset
# 5 =  $train_pts
# 6 =  $testing_ptset
# 7 =  $nb_test_points
# 8 =  $FE_degree
# 9 =  $SG_level
# 10 = $input_dim
# 11 = $UseFEM_tn
# 12 = trial

echo "CONTROL: training trial ${12} problem $3 with $4 total points"
python3 FEM_parametric_PDE_example.py --run_ID $1 --example $2 --problem $3 --train_pointset $4 --nb_train_points $5 --test_pointset $6 --nb_test_points $7 --FE_degree $8 --SG_level $9 --input_dim ${10} --train ${11} --trial_num ${12} 



