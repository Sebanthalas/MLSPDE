#########################################################################
# ==============================================================
# CODE APPROXIMATING THE SOLUTION OF A PDE
# Examples:  NSB (u;p) with a parametric 
# variable.
#
# ==============================================================
import sys
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from fenics import *
import numpy as np
import hdf5storage
import time
#import matplotlib.pyplot as plt
from numpy import linalg as la
import time, os, argparse, io, shutil, sys, math, socket
from dolfin import *
import sympy2fenics as sf
import random
import argparse
import threading
import time
import os
from callbacks import EarlyStoppingPredictHistory


def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))




if __name__ == '__main__': 
    
    # parse the arguments from the command line
    parser = argparse.ArgumentParser()

   

    # ARGs: FEM
    parser.add_argument("--run_ID",             type = str, help = "String for naming batch of trials in this run (default timestamp)") 
    parser.add_argument("--example",            default = 'other', type = str, help = "Example function to use in the PDE (default other)")
    parser.add_argument("--problem",            default = 'other', type = str, help = "Defines the PDE problem to solve")
    parser.add_argument("--train_pointset",     default = 'uniform_random', type = str, help = "Type of points to use in training (default uniform_random)")
    parser.add_argument("--DNN_max_nb_train_pts", default = 5, type = int, help = "Defines the max number of training points available")
    parser.add_argument("--test_pointset",      default = 'CC_sparse_grid', type = str, help = "Type of points to use in testing (default CC_sparse_grid)")
    parser.add_argument("--nb_test_points",     default = 1, type = int, help = "Number of points to use in testing (default 1)")
    parser.add_argument("--FE_degree",          default = 1, type = int, help = "Defines FE polynomial degree (default mesh number 2)")
    parser.add_argument("--SG_level",           default = 5, type = int, help = "Maximum order p of the polynomial space")
    parser.add_argument("--input_dim",          default = 1, type = int, help = "Dimension of the input (default 1)")
    parser.add_argument("--DNN_activation",     default = 'relu', type = str, help = "Defines the activation function")
    parser.add_argument("--DNN_precision",      default = 'single', type = str, help = "Defines the PDE problem to solve")
    parser.add_argument("--DNN_error_tol",      default = '5e-4', type = str, help = "Defines the error tolerance")
    parser.add_argument("--DNN_optimizer",      default = 'adam', type = str, help = "Defines the optimizer")
    parser.add_argument("--DNN_loss_function",  default = 'l2', type = str, help = "Defines the loss function")
    parser.add_argument("--DNN_blocktype",      default = 'default', type = str, help = "Defines blocktype")
    parser.add_argument("--DNN_initializer",    default = 'uniform', type = str, help = "Defines the initializer")
    parser.add_argument("--DNN_lrn_rate_schedule", default = 'exp_decay', type = str, help = "Defines the learning rate schedule")
    parser.add_argument("--DNN_type_loss",      default = 'customize', type = str, help = "Defines the type of loss")
    parser.add_argument("--DNN_epochs",           default = 20, type = int, help = "Defines the number of epochs")
    parser.add_argument("--DNN_nb_layers",        default = 2, type = int, help = "Defines the number of layers")
    parser.add_argument("--nb_train_points",     default = 1, type = int, help = "Number of points to use in training (default 1)")
    parser.add_argument("--whichfun",           default = "_u_", type = str, help = "Defines the function to approximate")
    parser.add_argument("--DNN_show_epoch",     default = 10, type = int, help = "Defines how often the epochs are deplayed")
    parser.add_argument("--DNN_test_epoch",     default = 50, type = int, help = "Defines the test epochs; not useful")
    parser.add_argument("--Use_batching",       default = 0, type = int, help = "Defines the use of batching. =0 (no batching) = NUM (Batch m/NUM)")
    parser.add_argument("--trial_num",          default = 1, type = int, help = "Number for the trial to run (default 0)")
    parser.add_argument("--fenics_log_level",   default = 30, type = int, help = "Log level for the FEniCS solver (default 30 = WARNING)")
    
    
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
    tf_seed = 0
    np_seed = trial
    np.random.seed(np_seed)
    print('Starting DNN training with tf_seed = %d and np_seed = %d' % (tf_seed, np_seed))

    fenics_params = {}
    # set the input dimension
    d         = args.input_dim
    example   = args.example
    if args.problem =="poisson":
        meshname = f"meshes/poisson.xml"
    elif args.problem =="NSB":
        meshname = "meshes/poisson.xml"
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
        if args.whichfun =='_p_':
            Hh   = FunctionSpace(mesh,'DG',0)
        else:
            Hh   = FunctionSpace(mesh, MixedElement([Hu,Ht,Hsig,Hsig,Hgam]))
    nvec = Hh.dim()
    #================================================================
    # *********** Trial and test functions ********** #
    #================================================================
    fenics_params['mesh']      = mesh
    fenics_params['V']         = Hh
    fenics_params['example']   = args.example
    fenics_params['input_dim'] = d

    #================================================================
    # *********** DNN settings ********** #
    #================================================================
    s = args.DNN_nb_layers  
    nb_layers          = 1*s
    nb_nodes_per_layer = 10*s
    
    

    start_time         = time.time() 
    activation         = args.DNN_activation  
    optimizer          = args.DNN_optimizer 
    initializer        = args.DNN_initializer 
    lrn_rate_schedule  = args.DNN_lrn_rate_schedule #'exp_decay'
    type_loss          = args.DNN_type_loss #'customize'
    blocktype          = args.DNN_blocktype #'default'
    sigma              = 0.1 
    nb_epochs          = args.DNN_epochs 
    best_loss          = 10
    m_test = args.nb_test_points

    #Choose the directory. If working on cedar comment the next line; uncomment the following
    current_directory   = os.getcwd()

    if args.test_pointset == 'CC_sparse_grid':
        test_results_filename = f"{current_directory}/SCS_FEM_{args.problem}/testing_data_{args.example}/{d}d_{args.SG_level}_SG_test_data.mat"
        getting_m_test    = sio.loadmat(test_results_filename)
        sorted(getting_m_test.keys())
        m_test = getting_m_test['m_test'][0,0]
        print('Number of testing points found_m:',m_test)
    # Gives m the   number of usable  trining points ; Gives the number of total points available to training
    m       = args.nb_train_points      # Actual number of trianing points used during training
    m_train = args.DNN_max_nb_train_pts # Max number of training points available 

    # unique key for naming results
    key          = f"{str(m_train).zfill(6)}_pnts_{d}_d"
    key_test     = f"{str(m_test).zfill(6)}_pnts_{d}_d"

    key_DNN = (
    f"FUN{args.whichfun}/"
    f"{str(m).zfill(6)}_pnts_e_tol_{float(args.DNN_error_tol)}_"
    f"optimizer_{args.DNN_optimizer}_d_{d}_optimizer_{args.DNN_loss_function}_loss_"
    f"{args.DNN_activation}_{args.DNN_nb_layers}x{nb_nodes_per_layer}_{args.DNN_blocktype}")

    scratchdir_train = os.path.join(
                                    current_directory,
                                    f"SCS_FEM_{args.problem}/training_data_{args.example}/{key}",)
    scratchdir_tests = os.path.join(
                                    current_directory,
                                    f"SCS_FEM_{args.problem}/testing_data_{args.example}/{key_test}",)

    scratchdir_resul = os.path.join(
                                    current_directory,
                                     f"SCS_FEM_{args.problem}/{unique_run_ID}_{args.example}/{trial}/{key_DNN}",)

    result_folder = scratchdir_resul

    if not os.path.exists(result_folder):
        try:
            os.makedirs(result_folder)
        except FileExistsError:
            print('===================================================================')
            print('skipping making', result_folder)
            print('===================================================================')

    run_data_filename = os.path.join(
                                    scratchdir_train,
                                    f"trial_{trial}", "_run_data.mat",)

    test_data_filename = os.path.join(
                                    scratchdir_tests,
                                    f"test_data{str(m_test).zfill(8)}_{args.test_pointset}_pts_test_data.mat",)

    DNN_results_filename = os.path.join(
                                    result_folder,
                                    f"data_m_{str(m).zfill(6)}_deg_{deg}_trial_{trial}_d_{d}_{activation}{nb_layers}x{nb_nodes_per_layer}_final.mat",)

    DNN_model_final_savedir = os.path.join(
                                    result_folder, f"DNN_finalModel_trial_",)

    print('path:', run_data_filename)
 


    if os.path.exists(run_data_filename):
        print('Found FEM train_data file:', run_data_filename)
        train_data       =  hdf5storage.loadmat(run_data_filename)
        y_in_train_data  =  train_data['y_in_train_data']
        Train_coeff_u    =  train_data['Train_coeff_u']
        Train_coeff_u    =  Train_coeff_u[range(m),:]
        y_in_train_data  =  y_in_train_data.T[range(m),:]
        print('===================================================================')
        print('TRAIN DATA FOUND number of training points available',len(Train_coeff_u))
        print('===================================================================')
        if args.problem =="NSB":
            if args.whichfun =='_p_':
                Train_coeff_u  = train_data['Train_coeff_p']
                Train_coeff_u  =  Train_coeff_u[range(m),:]

           
    #####################################################################################################################
    # FROM HERE ON EVERYTHONG IS IN TERMS OF -> u
    #####################################################################################################################

    # Extract the coefficients of all the functions and output dimensions
    #u_train_data         =  Train_coeff_u # extract_specific_function(Train_coeff_u,0).T
    output_dim_u         = Train_coeff_u.shape[1]
    

    if os.path.exists(test_data_filename):
        test_data       = sio.loadmat(test_data_filename)
        #sorted(test_data.keys())
        y_in_test_data  = test_data['y_in_test_data']
        Test_coeff_u    = test_data['Test_coeff_u']

        if args.problem =="NSB":
            if args.whichfun =='_p_':
                Test_coeff_u  = test_data['Test_coeff_p']


        if args.test_pointset == 'CC_sparse_grid':
            w_test_weights  = test_data['w_test_weights'][0]
            #print(w_test_weights)
        m_test         = test_data['m_test'][0,0]
        Test_coeff_u   = Test_coeff_u.T
        y_in_test_data = y_in_test_data.T
        print('===================================================================')
        print('TEST DATA FOUND number of testing points available',len(Test_coeff_u))
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
    tf.random.set_seed(tf_seed)
    print('Starting DNN training with tf_seed = %d and np_seed = %d' % (tf_seed, np_seed))
    start_time = time.time()

    #==============================================================================
    # DEFAULT SETTINGS
    #==============================================================================
    DNN_run_data = {}
    DNN_run_data['fenics_params']                  = fenics_params
    DNN_run_data['init_rate']                      = 1e-3
    DNN_run_data['decay_steps']                    = 1e3
    DNN_run_data['initializer']                    = initializer
    DNN_run_data['optimizer']                      = args.DNN_optimizer
    DNN_run_data['lrn_rate_schedule']              = lrn_rate_schedule
    DNN_run_data['error_tol']                      = error_tol 
    DNN_run_data['nb_epochs']                      = nb_epochs
    DNN_run_data['FUNCTION']                       = args.whichfun
    DNN_run_data['PROBLEM']                        = args.problem 


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
    DNN_run_data['trial']                          = trial
    DNN_run_data['run_ID']                         = unique_run_ID
    DNN_run_data['blocktype']                      = args.DNN_blocktype
    DNN_run_data['example']                        = args.example
    DNN_run_data['start_time']                     = start_time
    DNN_run_data['timestamp']                      = timestamp
    DNN_run_data['np_seed']                        = np_seed
    DNN_run_data['tf_seed']                        = tf_seed
    DNN_run_data['tf_version']                     = tf.__version__
    DNN_run_data['result_folder']                  = DNN_results_filename
    DNN_run_data['DNN_model_final_savedir']        = DNN_model_final_savedir
    DNN_run_data['run_data_filename']              = DNN_results_filename  
    DNN_run_data['key_DNN']                        = key_DNN
    DNN_run_data['input_dim']                      = d
    

    DNN_run_data['x_train_data']                   = y_in_train_data
    #DNN_run_data['y_train_data']                   = Train_coeff_u
    DNN_run_data['x_test_data']                    = y_in_test_data
    DNN_run_data['y_test_data']                    = Test_coeff_u
    DNN_run_data['w_quadrature_weights_test']      = w_test_weights
    DNN_run_data['test_pointset']                  = args.test_pointset
    DNN_run_data['update_ratio']                   = 0.0625
    DNN_run_data['patience']                       = 1e10


    DNN_run_data['output_dim']                     = output_dim_u 
    DNN_run_data['sigma']                          = sigma
    DNN_run_data['DNN_show_epoch']                 = args.DNN_show_epoch
    

    DNN_run_data['precision']                      = args.DNN_precision
    DNN_run_data['DNN_loss_function']              = args.DNN_loss_function
   

    print('Finite element degree: ' + str(deg))
    print('Training points      : ' + str(m))
    print('Testing points       : ' + str(m_test))
    #==============================================================================
    # DNN SETTING s
    #==============================================================================
    #==============================================================================
    # DNN SETTING 
    #==============================================================================
    

    if args.Use_batching==0:
        print("NO BATCH")
        BATCH_SIZE   = m # int(m/5)
    else:
        BATCH_SIZE   = int(m/5)
        print("BATCH=",BATCH_SIZE)

    BATCH_SIZE   = int(m/5)
    nb_train_pts = m
    nb_test_pts  = m_test
    x_train_data =  y_in_train_data
    x_test_data  =  y_in_test_data.T

    DNN_run_data['batch_size']                     = BATCH_SIZE
    DNN_u = tf.keras.Sequential()
    DNN_u.add(tf.keras.Input(shape=(DNN_run_data['input_dim'])))

    for layer in range(nb_layers+1):
        DNN_u.add(tf.keras.layers.Dense(DNN_run_data['nb_nodes_per_layer'], activation=DNN_run_data['activation'],
            kernel_initializer=weight_bias_initializer,
            bias_initializer=weight_bias_initializer
        ))

    DNN_u.add(tf.keras.layers.Dense(DNN_run_data['output_dim'],
        kernel_initializer=weight_bias_initializer,
        bias_initializer=weight_bias_initializer
    ))

    print(DNN_u)

    model_num_trainable_variables = np.sum([np.prod(v.get_shape().as_list()) for v in DNN_u.trainable_variables])
    print('This model has {} trainable variables'.format(model_num_trainable_variables))
    DNN_run_data['tf_trainable_vars'] = model_num_trainable_variables
    loss = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(
                    learning_rate=decay_schedule,
                    beta_1=0.9, beta_2=0.999, epsilon=1e-07, #amsgrad=False,
                    name='Adam'
                )
    DNN_u.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    print('Using data x_train_data (point samples) size ' + str(x_train_data.shape))
    if args.whichfun =='_u_':
        print('Using data train_data (FE coefficients) size ' + str(Train_coeff_u.shape))
    elif args.whichfun =='_p_':
        print('Using data train_data (FE coefficients) size ' + str(Train_coeff_u.shape))


    if not os.path.exists(DNN_results_filename):
        prediction_history = EarlyStoppingPredictHistory(DNN_run_data)
        DNN_u.summary()
        DNN_u.fit(x_train_data, Train_coeff_u, epochs=args.DNN_epochs, batch_size=BATCH_SIZE, verbose=0, shuffle=True, callbacks=[prediction_history])
    elif os.path.exists(DNN_results_filename):
        Check_run     =  hdf5storage.loadmat(DNN_results_filename)
        Check_epochs  =  Check_run['iterations_'][0][0]
        print('**********************************************')
        print('TOTAL ITERATION:'+str(Check_epochs))
        print('**********************************************')
        aux_iter = nb_epochs-2
        if Check_epochs <= aux_iter:
            prediction_history = EarlyStoppingPredictHistory(DNN_run_data)
            DNN_u.summary()
            DNN_u.fit(x_train_data, Train_coeff_u, epochs=args.DNN_epochs, batch_size=BATCH_SIZE, verbose=0, shuffle=True, callbacks=[prediction_history])

    Check_run     =  hdf5storage.loadmat(DNN_results_filename)
    Check_epochs  =  Check_run['iterations_'][0][0]
    print('TOTAL ITERATION:'+str(Check_epochs))
    if os.path.exists(DNN_model_final_savedir):
        shutil.rmtree(DNN_model_final_savedir)


    print('**********************************************')
    print('************** THE END ***********************')
    print('**********************************************')




    




 

