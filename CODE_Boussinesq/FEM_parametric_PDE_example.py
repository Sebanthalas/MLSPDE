
#########################################################################
######################## SETUP FEM PROBLEM DATA ########################### 
#########################################################################
# ==============================================================
# Code solving the possion problem 
# D_x(a,u)=0
# Formulation: Mixed formulation  
# Boundary conditions:  Natural    = True
#                       Essential  = False
# - tensor products
# ==============================================================
import sys
import time
import socket
import logging
import shutil
import math
import io
import argparse
import numpy as np
import scipy.io as sio
#import matplotlib.pyplot as plt
from numpy import linalg as la
from fenics import *
from dolfin import *
import os
import Tasmanian
import sympy2fenics as sf
from PDE_data_B import gen_dirichlet_data_B
 


def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

if __name__ == '__main__': 
  
  start_time = time.time()
  # parse the arguments from the command line
  parser = argparse.ArgumentParser()
  # General settngs 
  parser.add_argument("--test_pointset",   default = 'CC_sparse_grid', type = str, help = "Type of points to use in testing (default CC_sparse_grid)")
  parser.add_argument("--train",           default = 0,                type = int, help = "Switch for training or testingi (default 0=test)")
  parser.add_argument("--run_ID",                                      type = str, help = "String for naming batch of trials in this run (default timestamp)")
  parser.add_argument("--input_dim",       default = 1,                type = int, help = "Dimension of the input (default 1)")
  parser.add_argument("--nb_train_points", default = 1,                type = int, help = "Number of points to use in training (default 1)")
  parser.add_argument("--train_pointset",  default = 'uniform_random', type = str, help = "Type of points to use in training (default uniform_random)")
  #parser.add_argument("--precision",       default = 'double',         type = str, help = "Switch for double vs. single precision")
  parser.add_argument("--nb_test_points",  default = 1,                type = int, help = "Number of points to use in testing (default 1)")
  # PDE solver settings
  parser.add_argument("--problem",         default = 'poisson',        type = str, help = "Defines the PDE problem to solve")
  parser.add_argument("--FE_degree",       default = 1,                type = int, help = "Defines FE polynomial degree (default mesh number 2)")
  parser.add_argument("--example",         default = 'other',          type = str, help = "Example function to use in the PDE (default other)")
  parser.add_argument("--trial_num",       default = 0,                type = int, help = "Number for the trial to run (default 0)")
  #parser.add_argument("--error_tol",       default = "1e-4",           type = str, help = "Stopping tolerance for the solvers (default 1e-4)")
  parser.add_argument("--SG_level",        default = 5,                type = int, help = "Maximum order p of the polynomial space")
  parser.add_argument("--fenics_log_level",default = 30,               type = int, help = "Log level for the FEniCS solver (default 30 = WARNING)")
  args = parser.parse_args()
  # set the unique run ID used in many places, e.g., directory names for output
  if args.run_ID is None:
      unique_run_ID = timestamp
  else:
      unique_run_ID = args.run_ID

  # record the trial number
  trial   = args.trial_num
  np_seed = trial
  np.random.seed(np_seed)



  # Set the input dimension, mesh number, and example
  d       = args.input_dim
  example = args.example

  # Define mesh name based on the problem
  if args.problem == "Boussinesq":
      meshname = f"meshes/box3d.xml"
 

  # Create mesh
  mesh = Mesh(meshname)

  # Initialize coefficient lists and norms
  Train_coeff_u, Train_coeff_p, Train_coeff_phi   = [], [], []
  Test_coeff_u, Test_coeff_p, Test_coeff_phi      = [], [], []
  _L2unorm_train, _pnorm_train, _phinorm_train    = [], [], []
  _L2unorm_test, _pnorm_test, _phinorm_test       = [], [], []


  #================================================================
  #  *********** Finite Element spaces ************* #
  #================================================================
  deg = args.FE_degree
  if args.problem =="Boussinesq":
    Pkv = VectorElement('DG', mesh.ufl_cell(), deg)
    Pk  = FiniteElement('DG', mesh.ufl_cell(), deg)
    Rtv = FiniteElement('RT', mesh.ufl_cell(), deg+1)
    R0  = FiniteElement('R', mesh.ufl_cell(), 0)
    Ph  = FunctionSpace(mesh,'CG',1)

    Hh    = FunctionSpace(mesh, MixedElement([Pkv,Pk,Pk,Pk,Pk,Pk,Pk,Pk,Pk,Rtv,Rtv,Rtv,Pk,Pkv,Rtv,R0]))
    VVh   = VectorFunctionSpace(mesh,'DG',1)
   

  nvec = Hh.dim()
  _hmin = mesh.hmin()
  _hmax = mesh.hmax()
  print (" ****** Total DoF = ", Hh.dim())
  print (" ****** hmin = ", _hmin)
  print (" ****** hmax = ", _hmax)
  #================================================================
  # *********** Trial and test functions ********** #
  #================================================================
  # set the number of training points 

  m     = args.nb_train_points

  #================================================================
  #  *********** create the sparse grid generator ************* #
  #================================================================
  if args.test_pointset == 'CC_sparse_grid': 

    grid = Tasmanian.SparseGrid()

    # generate sparse grid points and weights
    grid.makeGlobalGrid(d, 0, args.SG_level, "level", "clenshaw-curtis")

    # get the points and weights from the generator
    y_in_test      = np.transpose(grid.getPoints())
    w_test_weights = grid.getQuadratureWeights()
    m_test         = y_in_test.shape[1]

    print('===================================================================')
    print('Using Clenshaw-Curtis sparse grid points with ', m_test, ' points')
    print('Sum of weights = ', np.sum(w_test_weights))


  elif args.test_pointset == 'uniform_random':

    # get the number of test points
    m_test = args.nb_test_points

    # generate the points randomly
    y_in_test = np.transpose(np.random.uniform(-1.0,1.0,(m_test,d)))    

  else:
    sys.exit('Must be one of the options, e.g., CC_sparse_grid or uniform_random')
  print('===================================================================')
  print('Testing with the', args.test_pointset, 'rule with', m_test, 'points in', d, 'dimensions')
  print('===================================================================')
  

  # Unique key for naming results
  key = f"{str(m).zfill(6)}_pnts_{d}_d"
  key_test = f"{str(m_test).zfill(6)}_pnts_{d}_d"

  # Save the training and test
  current_directory   = os.getcwd()
  result_folder_train = f"{current_directory}/SCS_FEM_{args.problem}/training_data_{args.example}/{key}"
  result_folder_test  = f"{current_directory}/SCS_FEM_{args.problem}/testing_data_{args.example}/{key_test}"
  run_data_filename   = f"{result_folder_train}/trial_{trial}"

  print('===================================================================')
  print('Saving results to', run_data_filename)
  print('===================================================================')

  test_data_filename = f"{result_folder_test}/test_data{str(m_test).zfill(8)}_{args.test_pointset}"
  if args.test_pointset == 'CC_sparse_grid':
      test_results_filename = f"{current_directory}/SCS_FEM_{args.problem}/{unique_run_ID}_{args.example}/{d}d_{args.SG_level}_SG_test_data.mat"
  else:
      test_results_filename = f"{current_directory}/SCS_FEM_{args.problem}/{unique_run_ID}_{args.example}/{d}d_{m_test}_mUR_test_data.mat"

  # See if this is useful later on
  m_test_check = m_test

  if args.train and not os.path.exists(run_data_filename + '/_run_data.mat'):
    try:
        os.makedirs(run_data_filename)
    except FileExistsError:
        print('===================================================================')
        print(f'skipping making {run_data_filename}')
        print('===================================================================')

    print('===================================================================')
    print('Saving results to', run_data_filename)
    print('===================================================================')
    print('       ____________________________________________________________________')
    print('                                Beginning training                         ')
    print('       ____________________________________________________________________')

    y_in_train = np.transpose(np.random.uniform(-1.0,1.0,(m,d)))
    #U = []
    print('Using uniform random training points with m =', m)
    K = nvec
    print('Generating the training data')
    # Generate the training data
    for i in range(m):
      # get the training data inputs 
      z = y_in_train[:,i]
      u_coefs,p_coefs,phi_coefs, norm_u_1, norm_u_2, norm_u_3 = gen_dirichlet_data_B(z,mesh, Hh, VVh, Ph, example,i,d)
      Train_coeff_u.append(u_coefs)
      Train_coeff_p.append(p_coefs)
      Train_coeff_phi.append(phi_coefs)
 
      
      print('====================================================================')
      print(f'i = {i}, L4u = {norm_u_1:2.5g}, L2p = {norm_u_2:2.5g}, L4phi = {norm_u_3:2.5g}, y_train = {z}')
      print('====================================================================')

    run_data = {
    'd': d,
    'K': K,
    'm_train': m,
    'y_in_train_data': y_in_train,
    'FE_degree': deg,
    'Train_coeff_u': Train_coeff_u,
    'Train_coeff_p': Train_coeff_p,
    'Train_coeff_phi': Train_coeff_phi,
    'fenics_mesh_coords': np.array(mesh.coordinates()),
    'fenics_mesh_cells': np.array(mesh.cells()),
    'fenics_mesh_num_cells': np.array(mesh.num_cells()),
    'fenics_mesh_num_edges': np.array(mesh.num_edges()),
    'fenics_mesh_num_vertices': np.array(mesh.num_vertices()),
    'fenics_mesh_hmax': np.array(mesh.hmax()),
    'fenics_mesh_hmin': np.array(mesh.hmin()),
    'fenics_mesh_rmax': np.array(mesh.rmax()),
    'fenics_mesh_rmin': np.array(mesh.rmin()),
    } 

    sio.savemat(run_data_filename + '/_run_data.mat', run_data)
    print('Saved in:', run_data_filename)
  elif not args.train and not os.path.exists(test_data_filename + '_pts_test_data.mat'):
    if not os.path.exists(result_folder_test):
      try:
          os.makedirs(result_folder_test)    
      except FileExistsError:
        print('===================================================================')
        print('skipping making', result_folder_test)
        print('===================================================================')

    print('===================================================================')
    print('Saving results to', test_data_filename + '_pts_test_data.mat')
    print('===================================================================')
    print('       ____________________________________________________________________')
    print('                                Beginning testing data                     ')
    print('       ____________________________________________________________________')
    print('Generating the testing data m_test=',m_test)
    for i in range(m_test):
        print('Generating the training data1')

        z = y_in_test[:, i]
        print('Generating the training data2')

        if args.problem == "Boussinesq":
            u_coefs,p_coefs,phi_coefs, norm_u_1, norm_u_2, norm_u_3 = gen_dirichlet_data_B(z,mesh, Hh, VVh, Ph, example,i,d)
            Test_coeff_u.append(u_coefs)
            Test_coeff_p.append(p_coefs)
            Test_coeff_phi.append(phi_coefs)
 

        if i == 0:
            K = len(u_coefs)
            print('FE degrees of freedom K_u =', K)

        print('====================================================================')
        print(f'i = {i}, L4u = {norm_u_1:2.5g}, L2p = {norm_u_2:2.5g}, L4phi = {norm_u_3:2.5g}, y_train = {z}')
        print('====================================================================')

    Test_coeff_u   = np.transpose(np.array(Test_coeff_u)) 
    Test_coeff_p   = np.transpose(np.array(Test_coeff_p))
    Test_coeff_phi = np.transpose(np.array(Test_coeff_phi)) 
    # Test data
    test_data = {'m_test': m_test}
    sio.savemat(test_results_filename, test_data)

    # Run data
    run_data = {
        'd': d,
        'K': K,
        'm_test': m_test,
        'y_in_test_data': y_in_test,
        'FE_degree': deg,
        'Test_coeff_u': Test_coeff_u,
        'Test_coeff_p': Test_coeff_p,
        'Test_coeff_phi': Test_coeff_phi,
        'fenics_mesh_coords': np.array(mesh.coordinates()),
        'fenics_mesh_cells': np.array(mesh.cells()),
        'fenics_mesh_num_cells': np.array(mesh.num_cells()),
        'fenics_mesh_num_edges': np.array(mesh.num_edges()),
        'fenics_mesh_num_vertices': np.array(mesh.num_vertices()),
        'fenics_mesh_hmax': np.array(mesh.hmax()),
        'fenics_mesh_hmin': np.array(mesh.hmin()),
        'fenics_mesh_rmax': np.array(mesh.rmax()),
        'fenics_mesh_rmin': np.array(mesh.rmin()),
    }

 

 

    if args.test_pointset == 'CC_sparse_grid':
        run_data['w_test_weights'] = w_test_weights
    save_test = test_data_filename + '_pts_test_data.mat'
    sio.savemat(save_test, run_data)
    print('Saved in:', save_test)

   
