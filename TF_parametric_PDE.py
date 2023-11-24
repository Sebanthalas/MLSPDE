#########################################################################
# ==============================================================
# Code approximating the possion problem using DNN 
# -a(x,y) \nalba u (x,y) = f ; u = g on \gamma
# Formulation: Mixed formulation  u \in L^2 sigma \in H(div)
# Boundary conditions:  Natural    = True
#                       Essential  = False
# - tensor products
# ==============================================================
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from fenics import *
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import linalg as la
import time, os, argparse, io, shutil, sys, math, socket
from dolfin import *
import sympy2fenics as sf

import random
import argparse
# ==============================================================
# Boundary conditions
# Zero on the boundary; one on the inner obstacle
#                    _______________________
#                    |                     |
#                    |     --              |
#                    |    |  |             |
#                    |     --              |
#                    |                     |
#                    _______________________ 
#
# ==============================================================
def extract_specific_function(T_list,k):
    Z = []
    num = len(T_list)
    for  j in range(num):
      Z.append(T_list[j][k])
    coeff = np.array(Z).T
    return coeff
def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))
if __name__ == '__main__': 
    
    # parse the arguments from the command line
    parser = argparse.ArgumentParser()
    # General settngs 
    parser.add_argument("--test_pointset", default = 'CC_sparse_grid', type = str, help = "Type of points to use in testing (default CC_sparse_grid)")
    parser.add_argument("--nb_trials", default = 1, type = int, help = "Number of trials to run for averaging results (default 1)")
    parser.add_argument("--train", default = 0, type = int, help = "Switch for training or testingi (default 0=test)")
    parser.add_argument("--run_ID", type = str, help = "String for naming batch of trials in this run (default timestamp)")
    parser.add_argument("--input_dim", default = 1, type = int, help = "Dimension of the input (default 1)")
    parser.add_argument("--nb_train_points", default = 1, type = int, help = "Number of points to use in training (default 1)")
    parser.add_argument("--max_nb_train_points", default = 500, type = int, help = "Maximum number of points to use in training for this run (default 500)")
    parser.add_argument("--train_pointset", default = 'uniform_random', type = str, help = "Type of points to use in training (default uniform_random)")
    parser.add_argument("--precision", default = 'double', type = str, help = "Switch for double vs. single precision")
    parser.add_argument("--nb_test_points", default = 1, type = int, help = "Number of points to use in testing (default 1)")
    # PDE solver settings
    parser.add_argument("--problem", default = 'other', type = str, help = "Defines the PDE problem to solve")
    parser.add_argument("--mesh_num", default = 2, type = int, help = "Defines the refiniment of the mesh, 1,2,3,4 (default mesh number 2)")
    parser.add_argument("--FE_degree", default = 1, type = int, help = "Defines FE polynomial degree (default mesh number 2)")
    parser.add_argument("--example", default = 'other', type = str, help = "Example function to use in the PDE (default other)")
    parser.add_argument("--quiet", default = 1, type = int, help = "Switch for verbose output (default 1)")
    parser.add_argument("--trial_num", default = 0, type = int, help = "Number for the trial to run (default 0)")
    parser.add_argument("--make_plots", default = 0, type = int, help = "Switch for generating plots (default 0)")
    parser.add_argument("--error_tol", default = "1e-4", type = str, help = "Stopping tolerance for the solvers (default 1e-4)")
    parser.add_argument("--SG_level", default = 5, type = int, help = "Maximum order p of the polynomial space")
    parser.add_argument("--fenics_log_level", default = 30, type = int, help = "Log level for the FEniCS solver (default 30 = WARNING)")
    #DNN parameters
    parser.add_argument("--DNN_activation", default = 'relu', type = str, help = "Defines the activation function")
    parser.add_argument("--DNN_precision", default = 'single', type = str, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_optimizer", default = 'adam', type = str, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_loss_function", default = 'l2', type = str, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_blocktype", default = 'default', type = str, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_initializer", default = 'uniform', type = str, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_lrn_rate_schedule", default = 'exp_decay', type = str, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_type_loss", default = 'customize', type = str, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_error_tol", default = '5e-4', type = str, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_epochs", default = 20, type = int, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_total_trials", default = 1, type = int, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_nb_layers", default = 2, type = int, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_max_nb_train_pts", default = 5, type = int, help = "Defines the PDE problem to solve")





    args = parser.parse_args()
    set_log_level(args.fenics_log_level)
    # set the unique run ID used in many places, e.g., directory names for output
    if args.run_ID is None:
      unique_run_ID = timestamp
    else:
      unique_run_ID = args.run_ID
    # record the trial number
    trial   = args.trial_num
    tf_seed = trial
    np_seed = trial
    np.random.seed(np_seed)
    print('Starting DNN training with tf_seed = %d and np_seed = %d' % (trial, np_seed))
    # set the input dimension
    d         = args.input_dim
    nk        = args.mesh_num
    example   = args.example
    meshname  = "meshes/obstac%03g.xml"%nk
    mesh      = Mesh(meshname)
    nn        = FacetNormal(mesh)
    #================================================================
    #  *********** Finite Element spaces ************* #
    #================================================================
    deg = args.FE_degree  
    Pk  = FiniteElement('DG', mesh.ufl_cell(), deg)
    RTv = FiniteElement('RT', mesh.ufl_cell(), deg+1)
    Hh  = FunctionSpace(mesh, MixedElement([Pk,RTv]))
    nvec = Hh.dim()
    #================================================================
    # *********** Trial and test functions ********** #
    #================================================================
    Utrial       = TrialFunction(Hh)
    Usol         = Function(Hh)
    W_trainsol   = Function(Hh)
    #================================================================
    # *********** DNN settings ********** #
    #================================================================
    s = args.DNN_nb_layers  
    nb_layers          = 1*s
    nb_nodes_per_layer = 10*s
    nb_nodes_per_layer_RT = 10*s
    Re = Constant(7.5)

    start_time         = time.time() 
    activation         = args.DNN_activation  
    activation_te      = args.DNN_activation 
    optimizer          = args.DNN_optimizer 
    initializer        = args.DNN_initializer 
    lrn_rate_schedule  = args.DNN_lrn_rate_schedule #'exp_decay'
    type_loss          = args.DNN_type_loss #'customize'
    blocktype          = args.DNN_blocktype #'default'
    sigma              = 0.1 
    error_tol          = args.DNN_error_tol 
    quiet              = 0
    nb_epochs          = args.DNN_epochs 
    nb_trials          = args.DNN_total_trials
    best_loss          = 10
    Id = Constant(((1.,0.),(0.,1.)))
    pi = 3.14159265359
    m_test = args.nb_test_points

    # Look for a different number of m_test (possibly from using Tasmanian)
    if args.test_pointset == 'CC_sparse_grid':
        test_results_filename = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/testing_data_'+ args.example + '/'+str(d)+'d_'+str(args.SG_level)+'_SG_test_results.mat'
        getting_m_test    = sio.loadmat(test_results_filename)
        sorted(getting_m_test.keys())
        m_test = getting_m_test['m_test'][0,0]
        print('foundm_m:',m_test)
    # Gives m the   number of usable  trining points ; Gives the number of total points available to training
    m       = args.nb_train_points      # Actual number of trianing points used during training
    m_train = args.DNN_max_nb_train_pts # Max number of training points available 

    # unique key for naming results
    key     = str(m_train).zfill(6) + '_pnts_%2.2e' % (float(args.error_tol)) + '_tol_'+str(d)+'_d'
    key_DNN = str(m).zfill(6) + '_pnts_%2.2e' % (float(args.DNN_error_tol)) + '_tol_' + args.DNN_optimizer + '_optimizer_' \
              + args.DNN_loss_function + '_loss_' + args.DNN_activation  + '_' + str(args.DNN_nb_layers) + 'x' \
              + str(nb_nodes_per_layer) + '_' + args.DNN_blocktype
    scratchdir_train    = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/training_data_' + args.example + '/' + key
    scratchdir_tests    = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/testing_data_'+ args.example + '/' + key
    scratchdir_resul    = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/' + unique_run_ID + '_' + args.example + '/' + key_DNN
    result_folder       = scratchdir_resul

    if not os.path.exists(result_folder):
      try:
          os.makedirs(result_folder)    
      except FileExistsError:
        print('===================================================================')
        print('skipping making', result_folder)
        print('===================================================================')
    
    #m_test_check = 9
    run_data_filename       = scratchdir_train + '/trial_' + str(trial) + '_run_data.mat'
    test_data_filename      = scratchdir_tests + '/test_data' + str(m_test).zfill(8) + '_' + args.test_pointset + '_pts_test_data.mat'
    DNN_results_filename    = result_folder  + '/data_m_'+str(m).zfill(6)+'_deg_'+str(deg)+'_mesh_'+str(nk)+'_af_'+activation+''+str(nb_layers)+'x'+str(nb_nodes_per_layer)+'_final.mat'
    #DNN_model_final_savedir = result_folder + '/DNN_finalModel_trial_' + str(trial)
    print('path:',run_data_filename)

    if os.path.exists(run_data_filename):
        print('Found FEM train_data file:', run_data_filename)
        train_data    = sio.loadmat(run_data_filename)
        sorted(train_data.keys())
        y_in_train_data  = train_data['y_in_train_data']
        All_Train_coeff  = train_data['All_Train_coeff']
        All_Train_coeff  =  All_Train_coeff[range(m),:]
        print('===================================================================')
        print('TRAIN DATA FOUND number of training points available',len(All_Train_coeff))
        print('===================================================================')
        #if m_test != m_test_check:
        #    errstr = ('Testing data does not match command line arguments m_test from file is %d but m_test from SG_level is %d, aborting.' % (m_test, m_test_check))
        #    sys.exit(errstr)
    if os.path.exists(test_data_filename):
        test_data    = sio.loadmat(test_data_filename)
        sorted(test_data.keys())
        y_in_test_data  = test_data['y_in_test_data']
        All_Test_coeff  = test_data['All_Test_coeff']
        if args.test_pointset == 'CC_sparse_grid':
            w_test_weights  = test_data['w_test_weights']
        m_test       = test_data['m_test'][0,0]
        K = y_in_test_data.shape[1]
        print('===================================================================')
        print('TEST DATA FOUND number of testing points available',len(All_Test_coeff))
        print('===================================================================')
    else: 
        errstr = ('no testing data')
        sys.exit(errstr)
    # Extract the coefficients of all the functions and output dimensions
    u_train_data =  extract_specific_function(All_Train_coeff,0).T
    output_dim   = u_train_data.shape[1]

    print('Running problem (key): ' + str(key_DNN))
    print("")
    print("*************************************")
    print('Starting trial: ' + str(trial))
    print("*************************************")
    print("")
    # set the precision variable to initialize weights and biases in either double or single precision
    if args.DNN_precision == 'double':
        print('Using double precision for DNN') 
        precision         = np.float64
        error_tol         = float(args.DNN_error_tol)
    elif args.DNN_precision == 'single':
        print('Using single precision for DNN')
        precision         = np.float32
        error_tol         = float(args.DNN_error_tol)
    print('Beginning DNN training')

    #==============================================================================
    # Version AND CONFIGURATIONS
    #==============================================================================
    #np_seed = 0 #trial
    #np.random.seed(np_seed)
    #python_random.seed(np_seed)
    tf.random.set_seed(tf_seed)
    print('Starting DNN training with tf_seed = %d and np_seed = %d' % (tf_seed, np_seed))
    start_time = time.time()

    #==============================================================================
    # DEFAULT SETTINGS
    #==============================================================================
    DNN_run_data = {}
    #DNN_run_data['fenics_params']                  = fenics_params
    #DNN_run_data['init_rate']                      = 1e-3
    #DNN_run_data['decay_steps']                    = 1e3
    DNN_run_data['initializer']                    = initializer
    DNN_run_data['optimizer']                      = args.DNN_optimizer
    DNN_run_data['lrn_rate_schedule']              = lrn_rate_schedule
    DNN_run_data['error_tol']                      = error_tol 
    DNN_run_data['nb_epochs']                      = nb_epochs

    print('=================================================================================')
    print('Running problem (key_DNN): ' + str(key_DNN))
    print('Saving to (projectdir_DNN): ' + str(DNN_results_filename))
    print('Starting trial: ' + str(trial))
    print('=================================================================================')



 

