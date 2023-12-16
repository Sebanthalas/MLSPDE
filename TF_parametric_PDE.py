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
import hdf5storage
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
    #parser.add_argument("--train", default = 0, type = int, help = "Switch for training or testingi (default 0=test)")
    parser.add_argument("--run_ID", type = str, help = "String for naming batch of trials in this run (default timestamp)")
    parser.add_argument("--input_dim", default = 1, type = int, help = "Dimension of the input (default 1)")
    parser.add_argument("--nb_train_points", default = 1, type = int, help = "Number of points to use in training (default 1)")
    #parser.add_argument("--max_nb_train_points", default = 500, type = int, help = "Maximum number of points to use in training for this run (default 500)")
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
    parser.add_argument("--DNN_test_epoch", default = 50, type = int, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_show_epoch", default = 10, type = int, help = "Defines the PDE problem to solve")
    parser.add_argument("--Use_batching", default = 0, type = int, help = "Defines the use of batching. =0 (no batching) = NUM (Batch m/NUM)")



    timestamp = str(int(time.time()));

    

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
    if args.problem =="poisson":
        meshname  = "meshes/obsta%03g.xml"%nk
    elif args.problem =="NSB":
        meshname  = "meshes/ComplexChannel.xml"
    mesh      = Mesh(meshname)
    nn        = FacetNormal(mesh)
    #================================================================
    #  *********** Finite Element spaces ************* #
    #================================================================
    deg = args.FE_degree  
    if args.problem =="poisson":
        Pk  = FiniteElement('DG', mesh.ufl_cell(), deg)
        RTv = FiniteElement('BDM', mesh.ufl_cell(), deg+1)
        Hh  = FunctionSpace(mesh, MixedElement([Pk,RTv]))
    elif args.problem =="NSB":
        Ht   = VectorElement('DG', mesh.ufl_cell(), deg+1, dim = 3)
        Hsig = FiniteElement('BDM', mesh.ufl_cell(), deg+1)# In FEniCS, Hdiv tensors need to be defined row-wise
        Hu   = VectorElement('DG', mesh.ufl_cell(), deg)
        Hgam = FiniteElement('DG', mesh.ufl_cell(), deg)    
        Hh   = FunctionSpace(mesh, MixedElement([Hu,Ht,Hsig,Hsig,Hgam]))
    nvec = Hh.dim()
    #================================================================
    # *********** Trial and test functions ********** #
    #================================================================
    #Utrial       = TrialFunction(Hh)
    soltrue = Function(Hh)
    solDNN  = Function(Hh)
    #================================================================
    # *********** DNN settings ********** #
    #================================================================
    s = args.DNN_nb_layers  
    nb_layers          = 1*s
    nb_nodes_per_layer = 10*s
    nb_nodes_per_layer_RT = 10*s
    

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
        test_results_filename = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/testing_data_'+ args.example + '/'+str(d)+'d_'+str(args.SG_level)+'_SG_test_data.mat'
        getting_m_test    = sio.loadmat(test_results_filename)
        sorted(getting_m_test.keys())
        m_test = getting_m_test['m_test'][0,0]
        print('foundm_m:',m_test)
    # Gives m the   number of usable  trining points ; Gives the number of total points available to training
    m       = args.nb_train_points      # Actual number of trianing points used during training
    m_train = args.DNN_max_nb_train_pts # Max number of training points available 

    # unique key for naming results
    key          = str(m_train).zfill(6) + '_pnts_%2.2e' % (float(args.error_tol)) + '_tol_'+str(d)+'_d'
    key_test     = str(m_test).zfill(6) + '_pnts_%2.2e' % (float(args.error_tol)) + '_tol_'+str(d)+'_d'
    key_DNN = str(m).zfill(6) + '_pnts_%2.2e' % (float(args.DNN_error_tol)) + '_tol_' + args.DNN_optimizer +'_d_'+str(d)+ '_optimizer_' \
              + args.DNN_loss_function + '_loss_' + args.DNN_activation  + '_' + str(args.DNN_nb_layers) + 'x' \
              + str(nb_nodes_per_layer) + '_' + args.DNN_blocktype
    scratchdir_train    = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/training_data_' + args.example + '/' + key
    scratchdir_tests    = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/testing_data_'+ args.example + '/' + key_test
    scratchdir_resul    = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/' + unique_run_ID + '_' + args.example +'/' + str(trial) + '/' + key_DNN
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
        train_data       = hdf5storage.loadmat(run_data_filename)
        #sorted(train_data.keys())
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
        #sorted(test_data.keys())
        y_in_test_data  = test_data['y_in_test_data']
        All_Test_coeff  = test_data['All_Test_coeff']
        if args.test_pointset == 'CC_sparse_grid':
            w_test_weights  = test_data['w_test_weights'][0]
            print(w_test_weights)
        m_test       = test_data['m_test'][0,0]
        K = y_in_test_data.shape[1]
        print('===================================================================')
        print('TEST DATA FOUND number of testing points available',len(All_Test_coeff))
        print('===================================================================')
    else: 
        errstr = ('no testing data')
        sys.exit(errstr)
    

    print('Running problem (key): ' + str(key_DNN))
    print("")
    print("***************************************************************************************************************")
    print("***************************************************************************************************************")
    print("")
    print('       STARTING TRIAL:' + str(trial) + '      DIMENSION:' +str(d) + '      TRAINING POINTS:'+str(m)+'   ACTIVATION: '+ activation)
    print("")
    print("***************************************************************************************************************")
    print("***************************************************************************************************************")
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
    DNN_run_data['init_rate']                      = 1e-3
    DNN_run_data['decay_steps']                    = 1e3
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
    # set up the learning rate schedule from either exp_decay, linear, or constant 
    if DNN_run_data['lrn_rate_schedule'] == 'exp_decay':

        # calculate the base so that the learning rate schedule with 
        # exponential decay follows (init_rate)*(base)^(current_epoch/decay_steps)
        DNN_run_data['base'] = np.exp(DNN_run_data['decay_steps']/DNN_run_data['nb_epochs']
               *(np.log(DNN_run_data['error_tol'])-np.log(DNN_run_data['init_rate'])))

        # based on the above, the final learning rate is (init_rate)*(base)^(total_epochs/decay_steps)
        print('based on init_rate = ' + str(DNN_run_data['init_rate'])
            + ', decay_steps = ' + str(DNN_run_data['decay_steps'])
            + ', calculated base = ' + str(DNN_run_data['base']) 
            + ', so that after ' + str(DNN_run_data['nb_epochs'])
            + ' epochs, we have final learning rate = '
            + str(DNN_run_data['init_rate']*DNN_run_data['base']**(DNN_run_data['nb_epochs']/DNN_run_data['decay_steps'])))
        decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            DNN_run_data['init_rate'], DNN_run_data['decay_steps'], DNN_run_data['base'], staircase=False, name=None
        )

    elif args.lrn_rate_schedule == 'linear':

        print('using a linear learning rate schedule')
        decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            DNN_run_data['init_rate'], DNN_run_data['decay_steps'], end_learning_rate=DNN_run_data['error_tol'], power=1.0
        )

    elif args.lrn_rate_schedule == 'constant':

        decay_schedule = DNN_run_data['init_rate']
        print('using a constant learning rate')
    # define the initializers for the weights and biases
    if args.DNN_initializer == 'normal':
        weight_bias_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=sigma, seed=trial)
    elif args.DNN_initializer == 'uniform':
        weight_bias_initializer = tf.keras.initializers.RandomUniform(minval=-sigma, maxval=sigma, seed=trial)
    elif args.DNN_initializer == 'constant':
        weight_bias_initializer = tf.keras.initializers.Constant(value=sigma)
    elif args.DNN_initializer == 'he_normal':
        weight_bias_initializer = initializers.HeNormal(seed=trial)
    elif args.DNN_initializer == 'he_uniform':
        weight_bias_initializer = tf.keras.initializers.HeUniform(seed=trial)
    elif args.DNN_initializer == 'glorot_normal':
        weight_bias_initializer = tf.keras.initializers.GlorotNormal(seed=trial)
    elif args.DNN_initializer == 'glorot_uniform':
        weight_bias_initializer = tf.keras.initializers.GlorotUniform(seed=trial)
    else: 
        sys.exit('args.DNN_initializer must be one of the supported types, e.g., normal, uniform, etc.')
    DNN_run_data['activation']                     = args.DNN_activation
    DNN_run_data['nb_layers']                      = args.DNN_nb_layers
    DNN_run_data['nb_nodes_per_layer']             = nb_nodes_per_layer
    DNN_run_data['nb_train_points']                = m
    DNN_run_data['nb_test_points']                 = m_test
    DNN_run_data['nb_trials']                      = args.nb_trials
    DNN_run_data['trial']                          = trial
    DNN_run_data['run_ID']                         = unique_run_ID
    DNN_run_data['blocktype']                      = args.DNN_blocktype
    DNN_run_data['example']                        = args.example
    DNN_run_data['start_time']                     = start_time
    DNN_run_data['timestamp']                      = timestamp
    DNN_run_data['np_seed']                        = np_seed
    DNN_run_data['tf_seed']                        = tf_seed
    DNN_run_data['tf_version']                     = tf.__version__
    #DNN_run_data['result_folder']                  = projectdir_DNN
    #DNN_run_data['DNN_model_final_savedir']        = DNN_model_final_savedir
    DNN_run_data['run_data_filename']              = DNN_results_filename                
    DNN_run_data['key_DNN']                        = key_DNN
    DNN_run_data['input_dim']                      = d
    #DNN_run_data['x_train_data']                   = x_train_data
    #DNN_run_data['y_train_data']                   = y_train_data
   #DNN_run_data['x_test_data']                    = x_test_data
    #DNN_run_data['y_test_data']                    = y_test_data
    #DNN_run_data['w_quadrature_weights_test']      = w_quadrature_weights_test
    #DNN_run_data['test_pointset']                  = args.test_pointset
    DNN_run_data['sigma']                          = sigma
    DNN_run_data['update_ratio']                   = 0.0625
    #DNN_run_data['quiet']                          = args.quiet
    DNN_run_data['patience']                       = 1e10
    DNN_run_data['output_dim']                     = K
    DNN_run_data['precision']                      = args.DNN_precision
    DNN_run_data['intermediate_testing']           = 1
    DNN_run_data['intermediate_testing_interval']  = 1000
    DNN_run_data['DNN_loss_function']              = args.DNN_loss_function
    #DNN_run_data['G']                              = np.asarray(sparse.csr_matrix.todense(G))
    #DNN_run_data['batch_size']                     = min(args.DNN_batch_size, m)
    print('mesh                 : ' + str(nk))
    print('Finite element degree: ' + str(deg))
    print('Training points      : ' + str(m))
    print('Testing points       : ' + str(m_test))
    #==============================================================================
    # DNN SETTING 
    #==============================================================================
    # Extract the coefficients of all the functions and output dimensions
    u_train_data       =  extract_specific_function(All_Train_coeff,0).T
    te11_train_data    =  extract_specific_function(All_Train_coeff,1).T
    output_dim   = u_train_data.shape[1]

    if args.Use_batching==0:
        print("NO BATCH")
        BATCH_SIZE   = m # int(m/5)
    else:
        BATCH_SIZE   = int(m/5)
        print("BATCH=",BATCH_SIZE)
    BATCH_SIZE   = m # int(m/5)
    nb_train_pts = m
    nb_test_pts  = m_test
    x_train_data =  y_in_train_data.T
    x_test_data  =  y_in_test_data.T
    #==============================================================================
    # # Functional API
    #==============================================================================
    all_layers_u    = []
    all_layers_te11 = []
    inputs       = keras.Input(shape=(d,), name = 'input_y')
    branch_u     = tf.keras.layers.Dense(nb_nodes_per_layer, 
                                                        activation=activation,
                                                        name='branch_u',
                                                        kernel_initializer= weight_bias_initializer,
                                                        bias_initializer= weight_bias_initializer,
                                                        dtype=tf.float32)(inputs) 
    branch_te11  = tf.keras.layers.Dense(nb_nodes_per_layer,
                                                activation=activation,
                                                name='branch_te11',
                                                kernel_initializer= weight_bias_initializer,
                                                bias_initializer= weight_bias_initializer,
                                                dtype=tf.float32)(inputs)
    # Append each layer
    all_layers_u.append(branch_u)
    all_layers_te11.append(branch_te11)

    for layer in range(nb_layers):
        layer_name_u    = 'dense_u' + str(layer)
        layer_name_te11 = 'dense_te11' + str(layer)
        this_layer_u = tf.keras.layers.Dense(nb_nodes_per_layer, 
                                            activation=activation,
                                            name=layer_name_u,
                                            kernel_initializer= weight_bias_initializer,
                                            bias_initializer= weight_bias_initializer,
                                            dtype=tf.float32)(all_layers_u[layer])
        this_layer_te11 = tf.keras.layers.Dense(nb_nodes_per_layer,
                                                activation=activation,
                                                name=layer_name_te11,
                                                kernel_initializer= weight_bias_initializer,
                                                bias_initializer= weight_bias_initializer,
                                                dtype=tf.float32)(all_layers_te11[layer])
        all_layers_u.append(this_layer_u)
        all_layers_te11.append(this_layer_te11)                              


    final_num = layer+1
    first_output_layer_u    = tf.keras.layers.Dense(output_dim,
                                                activation= tf.keras.activations.linear,
                                                trainable=True,
                                                use_bias=False,
                                                name='output_u',
                                                kernel_initializer=weight_bias_initializer,
                                                dtype=tf.float32)(all_layers_u[final_num])
    first_output_layer_te11 = tf.keras.layers.Dense(output_dim,
                                                    activation= tf.keras.activations.linear,
                                                    trainable=True,
                                                    use_bias=False,
                                                    name='output_te11',
                                                    kernel_initializer=weight_bias_initializer,
                                                    dtype=tf.float32)(all_layers_te11[final_num])                                            

    DNN = tf.keras.Model(inputs=inputs, outputs=[first_output_layer_u,first_output_layer_te11])

    

    DNN.summary()
    #------------------------------------------------------------------------------# 
    # Define tf.functions 
    #------------------------------------------------------------------------------#
    @tf.function
    def get_u(y):
        y = tf.convert_to_tensor(y)    
        uy = DNN(y)[0]
        return uy # Cofficients 
    @tf.function
    def get_te11(y):
        y = tf.convert_to_tensor(y)
        uy = DNN(y)[1]
        return uy # Cofficients
    @tf.function
    def get_mse(y,u_true,te11):
        # minimize solution u on Omega x[0,T]
        mse_1 = tf.reduce_mean((get_u(y)- u_true)**2 )
        mse_2 = tf.reduce_mean((get_te11(y)- te11)**2)  
        return mse_1 + mse_2 
    OPT = tf.keras.optimizers.Adam(1e-4)
    @tf.function
    def GD(y,u_true,te11):
        with tf.GradientTape() as g:
            g.watch(DNN.variables)
            mse = get_mse(y,u_true,te11)
        G = g.gradient(mse,DNN.variables)
        OPT.apply_gradients(zip(G,DNN.variables)) 
    # Data inside the domain
    y_test        = tf.cast(np.asarray(x_test_data), tf.float32)
    var2          = tf.cast(np.asarray(All_Test_coeff), tf.float32)
    # Auxiliary variables
    L2_error_data   = np.array([])
    L2u_err_append  = np.array([])
    Hdiv_err_append = np.array([])
    for epoch in range(nb_epochs):
        I_ad       = np.array([]) 
        I          = np.array([])
        batch_size = int(BATCH_SIZE)
        I_i   = range(m)
        I_ad  = random.sample(I_i, batch_size)
        I     = np.append(I, I_ad).astype(int)
        # Data inside the domain
        y_in     = tf.cast(np.asarray(x_train_data)[I,:], tf.float32)
        u_in     = tf.cast(np.asarray(u_train_data)[I,:], tf.float32)
        te11_in  = tf.cast(np.asarray(te11_train_data)[I,:], tf.float32)
        # gradient descent
        GD(y_in,u_in,te11_in)
        # compute loss
        res = get_mse(y_in,u_in,te11_in)
        L2_error_data = np.append(L2_error_data, res)
        if (epoch % args.DNN_show_epoch ==0):
            print('============================================================')
            print('Epochs: ' + str(epoch) + ' | Error: ' + str("{:.4e}".format(res)) )
            print('============================================================')
        if epoch % args.DNN_test_epoch == 0:
            u_pred = DNN(y_test)  
            u_coef_pred = tf.cast(u_pred, dtype= tf.float32)
            L2u_err     = 0.0
            Hdi_err     = 0.0
            
            for i in range(m_test):
                #The coefficient
                z = x_test_data[i,:]
                #print("Loading the test results for:",i)
                #if example == 'other':
                #  pi     = str(3.14159265359)
                #  amean  = str(2)
                #  string = '1.1 + '
                #  for j in range(d):
                #    term   =  str(z[j])+ '*sin('+pi+'*(x+y)/(pow('+str(j)+'+1.0,2)))/(pow('+str(j)+'+1.0,2))'
                #    string =  string + '+' + term
                #string  =  '1.0/('+string+')' 
                #a       = Expression(str2exp(string), degree=2, domain=mesh)
                # REAL

                var_aux_u_real = np.array(var2[i,:])     
                soltrue.vector().set_local(var_aux_u_real)
                # DNN coeff
                var       = u_pred[0][0,:]*0
                #print(len(u_pred))
                for k in range(2):
                        var = var + u_pred[k][i,:]
                var_aux_u = np.array(var)
                #var_aux_u =  np.array(u_coef_pred[i, :])
                # DNN In
                solDNN.vector().set_local(var_aux_u)
                if args.problem =="poisson":
                    u_sol, sigma_FEM   = soltrue.split()
                    uh   , sigma_DNN   = solDNN.split() 
                elif args.problem =="NSB":
                    u_sol, sigma_FEM,_,_,_   = soltrue.split()
                    uh   , sigma_DNN,_,_,_   = solDNN.split()        

                error_L2u = assemble((u_sol-uh)**2*dx)  
                error_Hdi = assemble((sigma_FEM-sigma_DNN)**2*dx) # +assemble( ( div(sigma_FEM)- div(sigma_DNN) )**2*dx)  

                L2u_err += error_L2u * w_test_weights[i]
                Hdi_err += error_Hdi * w_test_weights[i]
            
            plot(uh)
            filename = 'results/_'+str(epoch)+'_DNNu.png'
            plt.savefig ( filename )
            plt.close()
            plot(sigma_DNN[0])
            filename = 'results/_'+str(epoch)+'_DNNsigma.png'
            plt.savefig ( filename )
            plt.close()

            #plot(u_sol)
            #filename = '_'+str(epoch)+'_uh.png'
            #plt.savefig ( filename )
            #plt.close()
            #plot(sigma_FEM)
            #filename = '_'+str(epoch)+'_sigmah.png'
            #plt.savefig ( filename )
            #plt.close()
 
                
            L2u_err = np.sqrt(np.abs(L2u_err/2**(d)))
            Hdi_err = np.sqrt(np.abs(Hdi_err/2**(d)))

            # Append
            L2u_err_append  = np.append(L2u_err_append, L2u_err)  
            Hdiv_err_append = np.append(Hdiv_err_append, Hdi_err) 
            print('Epochs: ' + str(epoch) + ' | Error: ' + str("{:.4e}".format(res)) )
            print('Testing errors: L4u_e = %4.3e,L2sig_e = %4.3e' % (L2u_err,Hdi_err))
            if (res <= error_tol) or (epoch == nb_epochs-1):
                if res <= best_loss:
                    best_loss   = res
                    best_epoch  = epoch
                    #DNN.save(result_folder + '/model_save_folder')
    # Save data
    test_results = {}
    test_results['L2u_err_af_'+activation +'_Npl'+str(nb_layers)+'x'+str(nb_nodes_per_layer)+'_m_'+str(m)+'_trial_'+str(trial)+'_dim_'+str(d)+'_problem_'+args.problem+'']   = L2u_err_append
    test_results['Hdi_err_af_'+activation +'_Npl'+str(nb_layers)+'x'+str(nb_nodes_per_layer)+'_m_'+str(m)+'_trial_'+str(trial)+'_dim_'+str(d)+'_problem_'+args.problem+'']   = Hdiv_err_append
    test_results['residual_af_'+activation+'_Npl'+str(nb_layers)+'x'+str(nb_nodes_per_layer)+'_m_'+str(m)+'_trial_'+str(trial)+'_dim_'+str(d)+'_problem_'+args.problem+'']   = L2_error_data
    test_results['Best_epoch']   = best_epoch

    # save the resulting mat file with scipy.io
    sio.savemat(result_folder + '/' + key_DNN+'_final.mat', test_results)
        



 

