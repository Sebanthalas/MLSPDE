
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
import scipy.io as sio
from fenics import *
import Tasmanian
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import linalg as la
import time, os, argparse, io, shutil, sys, math, socket
import scipy.io as sio
from dolfin import *
import sympy2fenics as sf
import random
import argparse
from PDE_data_poisson import gen_dirichlet_data_poisson
from PDE_data_NSB     import gen_dirichlet_data_NSB
import logging

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

if __name__ == '__main__': 
  
  start_time = time.time()
  # parse the arguments from the command line
  parser = argparse.ArgumentParser()
  # General settngs 
  parser.add_argument("--test_pointset", default = 'CC_sparse_grid', type = str, help = "Type of points to use in testing (default CC_sparse_grid)")
  parser.add_argument("--train", default = 0, type = int, help = "Switch for training or testingi (default 0=test)")
  parser.add_argument("--run_ID", type = str, help = "String for naming batch of trials in this run (default timestamp)")
  parser.add_argument("--input_dim", default = 1, type = int, help = "Dimension of the input (default 1)")
  parser.add_argument("--nb_train_points", default = 1, type = int, help = "Number of points to use in training (default 1)")
  parser.add_argument("--train_pointset", default = 'uniform_random', type = str, help = "Type of points to use in training (default uniform_random)")
  parser.add_argument("--precision", default = 'double', type = str, help = "Switch for double vs. single precision")
  parser.add_argument("--nb_test_points", default = 1, type = int, help = "Number of points to use in testing (default 1)")
  # PDE solver settings
  parser.add_argument("--problem", default = 'poisson', type = str, help = "Defines the PDE problem to solve")
  parser.add_argument("--mesh_num", default = 2, type = int, help = "Defines the refiniment of the mesh, 1,2,3,4 (default mesh number 2)")
  parser.add_argument("--FE_degree", default = 1, type = int, help = "Defines FE polynomial degree (default mesh number 2)")
  parser.add_argument("--example", default = 'other', type = str, help = "Example function to use in the PDE (default other)")
  parser.add_argument("--quiet", default = 1, type = int, help = "Switch for verbose output (default 1)")
  parser.add_argument("--trial_num", default = 0, type = int, help = "Number for the trial to run (default 0)")
  parser.add_argument("--make_plots", default = 0, type = int, help = "Switch for generating plots (default 0)")
  parser.add_argument("--error_tol", default = "1e-4", type = str, help = "Stopping tolerance for the solvers (default 1e-4)")
  parser.add_argument("--SG_level", default = 5, type = int, help = "Maximum order p of the polynomial space")
  parser.add_argument("--fenics_log_level", default = 30, type = int, help = "Log level for the FEniCS solver (default 30 = WARNING)")
  args = parser.parse_args()
 
  #set_log_level(args.fenics_log_level)

  # set the unique run ID used in many places, e.g., directory names for output
  if args.run_ID is None:
      unique_run_ID = timestamp
  else:
      unique_run_ID = args.run_ID
  # record the trial number
  trial   = args.trial_num
  np_seed = trial
  np.random.seed(np_seed)



  # set the input dimension
  d         = args.input_dim
  nk        = args.mesh_num
  example   = args.example
  if args.problem =="poisson":
    meshname  = "meshes/obsta%03g.xml"%nk
  elif args.problem =="NSB":
    meshname  = "meshes/ComplexChannel.xml"
  mesh      = Mesh(meshname)
  Train_coeff_u = []
  Train_coeff_p = []
  Test_coeff_u  = []
  Test_coeff_p  = []
  _L2unorm_train  = []
  _H2snorm_train  = []
  _L2unorm_test   = []
  _H2snorm_test   = []


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
  # set the maximum number of training points 
  # set the precision variable to initialize weights and biases in either double or single precision
  if args.precision == 'double':
    print('===================================================================')
    print('Using double precision') 
    print('===================================================================')
    precision = np.float64
    error_tol = float(args.error_tol)
  elif args.precision == 'single':
    print('===================================================================')
    print('Using single precision')
    print('===================================================================')
    precision = np.float32
    error_tol = float(args.error_tol)

  #================================================================
  #  *********** create the sparse grid generator ************* #
  #================================================================
  if args.test_pointset == 'CC_sparse_grid': 
    grid = Tasmanian.SparseGrid()
    # generate sparse grid points and weights
    grid.makeGlobalGrid(d, 0, args.SG_level, "level", "clenshaw-curtis")
    # get the points and weights from the generator
    y_in_test = np.transpose(grid.getPoints())
    w_test_weights = grid.getQuadratureWeights()
    m_test = y_in_test.shape[1]
    print('===================================================================')
    print('Using Clenshaw-Curtis sparse grid points with ', m_test, ' points')
    print('Sum of weights = ', np.sum(w_test_weights))
    # scatter plot the points

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
  

  # unique key for naming results
  key      = str(m).zfill(6) + '_pnts_%2.2e' % (error_tol) + '_tol_'+str(d)+'_d'
  key_test = str(m_test).zfill(6) + '_pnts_%2.2e' % (error_tol) + '_tol_'+str(d)+'_d'

  # Save the training and test#
  
  result_folder_train = '/home/semoraga/projects/def-adcockb/semoraga/in_cedar/SCS_FEM_'+args.problem+'/training_data_' + args.example + '/' + key
  result_folder_test  = '/home/semoraga/projects/def-adcockb/semoraga/in_cedar/SCS_FEM_'+args.problem+'/testing_data_' + args.example + '/' + key_test

  
  
  run_data_filename  = result_folder_train + '/trial_' + str(trial) 
   
  print('===================================================================')
  print('Saving results to', run_data_filename)
  print('===================================================================')
 
  
  
  test_data_filename = result_folder_test + '/test_data' + str(m_test).zfill(8) + '_' + args.test_pointset + ''
  if args.test_pointset == 'CC_sparse_grid':
    test_results_filename = '/home/semoraga/projects/def-adcockb/semoraga/in_cedar/SCS_FEM_'+args.problem+'/' + unique_run_ID + '_' + args.example + '/'+str(d)+'d_'+str(args.SG_level)+'_SG_test_data.mat'
  else:
    test_results_filename = '/home/semoraga/projects/def-adcockb/semoraga/in_cedar/SCS_FEM_'+args.problem+'/' + unique_run_ID + '_' + args.example + '/'+str(d)+'d_'+str(m_test)+'_mUR_test_data.mat'


  # See if this is usefl later on
  m_test_check = m_test

  if args.train:
    if not os.path.exists(run_data_filename):
      try:
          os.makedirs(run_data_filename)    
      except FileExistsError:
        print('===================================================================')
        print('skipping making', run_data_filename)
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
      print('Generating the training data2')
      if args.problem =="poisson":
        print('Generating the training data3')
        u_coefs, norm_u_1, norm_u_2 = gen_dirichlet_data_poisson(z,mesh, Hh, example,i,d,args.train)
        print('Generating the training data4')
        Train_coeff_u.append(u_coefs)
        


      elif args.problem =="NSB":
        u_coefs,p_coefs, norm_u_1, norm_u_2 = gen_dirichlet_data_NSB(z,mesh, Hh, example,i,d,args.train)
        Train_coeff_u.append(u_coefs)
        Train_coeff_p.append(p_coefs)


    


      _L2unorm_train.append(norm_u_1)
      _H2snorm_train.append(norm_u_2)
      
      print('====================================================================')
      print('i = ', i, 'L2u=  %2.5g ' % norm_u_1, 'Lsig=  %2.5g ' % norm_u_2,'y_train= ',z )
      print('====================================================================')
    run_data = {}
    run_data['d']              = d
    run_data['K']              = K
    run_data['m_train']        = m
    run_data['y_in_train_data']= y_in_train
    run_data['mesh_op']        = nk
    run_data['FE_degree']      = deg
    run_data['Train_coeff_u']  = Train_coeff_u
    if args.problem =="NSB":
      run_data['Train_coeff_p'] = Train_coeff_p
    run_data['_L2unorm_train']           = _L2unorm_train
    run_data['_H2snorm_train']           = _H2snorm_train
    run_data['fenics_mesh_coords']       = np.array(mesh.coordinates())
    run_data['fenics_mesh_cells']        = np.array(mesh.cells())
    run_data['fenics_mesh_num_cells']    = np.array(mesh.num_cells())
    run_data['fenics_mesh_num_edges']    = np.array(mesh.num_edges())
    run_data['fenics_mesh_num_vertices'] = np.array(mesh.num_vertices())
    run_data['fenics_mesh_hmax']         = np.array(mesh.hmax())
    run_data['fenics_mesh_hmin']         = np.array(mesh.hmin())
    run_data['fenics_mesh_rmax']         = np.array(mesh.rmax())
    run_data['fenics_mesh_rmin']         = np.array(mesh.rmin())
    sio.savemat(run_data_filename+'/_run_data.mat', run_data)
    print('saved in:',run_data_filename)
  
  else:
    if not os.path.exists(result_folder_test):
      try:
          os.makedirs(result_folder_test)    
      except FileExistsError:
        print('===================================================================')
        print('skipping making', result_folder_test)
        print('===================================================================')

    print('===================================================================')
    print('Saving results to', test_data_filename)
    print('===================================================================')
    print('       ____________________________________________________________________')
    print('                                Beginning testing data                     ')
    print('       ____________________________________________________________________')
    print('Generating the testing data m_test=',m_test)
    for i in range(m_test):
      print('Generating the training data1')
      
      z = y_in_test[:,i]
      print('Generating the training data2')
          
      if args.problem =="poisson":
        print('Generating the training data3')
        logging.warning('Watch out!')
        coeff_each_m_u, norm_u_1, norm_u_2 = gen_dirichlet_data_poisson(z,mesh, Hh, example,i,d,args.train)
        Test_coeff_u.append(coeff_each_m_u)
        


      elif args.problem =="NSB":
        coeff_each_m_u,coeff_each_m_p, norm_u_1, norm_u_2 = gen_dirichlet_data_NSB(z,mesh, Hh, example,i,d,args.train)
        Test_coeff_u.append(coeff_each_m_u)
        Test_coeff_p.append(coeff_each_m_p)
 
      _L2unorm_test.append(norm_u_1)
      _H2snorm_test.append(norm_u_2)
      if i == 0:
        K = len(coeff_each_m_u)
        print('FE degrees of freedom K = ', K)

      print('====================================================================')
      print('i = ', i, 'L2u=  %2.4g ' % norm_u_1,'y_test= ', z)
      print('====================================================================')

    Test_coeff_u = np.transpose(np.array(Test_coeff_u))  
    test_data ={}
    test_data['m_test']         = m_test
    sio.savemat(test_results_filename, test_data)
    run_data = {}
    run_data['d']              = d
    run_data['K']              = K
    #run_data['m_max']          = m_max
    run_data['m_test']         = m_test
    run_data['y_in_test_data'] = y_in_test
    run_data['mesh_op']        = nk
    run_data['FE_degree']      = deg
    run_data['Test_coeff_u']   = Test_coeff_u
    if args.problem =="NSB":
      Test_coeff_p = np.transpose(np.array(Test_coeff_p))  
      run_data['Test_coeff_p'] = Test_coeff_p
    run_data['_L2unorm_test']       = _L2unorm_test
    run_data['_H2snorm_test']       = _H2snorm_test
    if args.test_pointset == 'CC_sparse_grid':
      run_data['w_test_weights']      = w_test_weights
    run_data['fenics_mesh_coords']       = np.array(mesh.coordinates())
    run_data['fenics_mesh_cells']        = np.array(mesh.cells())
    run_data['fenics_mesh_num_cells']    = np.array(mesh.num_cells())
    run_data['fenics_mesh_num_edges']    = np.array(mesh.num_edges())
    run_data['fenics_mesh_num_vertices'] = np.array(mesh.num_vertices())
    run_data['fenics_mesh_hmax']         = np.array(mesh.hmax())
    run_data['fenics_mesh_hmin']         = np.array(mesh.hmin())
    run_data['fenics_mesh_rmax']         = np.array(mesh.rmax())
    run_data['fenics_mesh_rmin']         = np.array(mesh.rmin())
    sio.savemat(test_data_filename+'_pts_test_data.mat', run_data)
    print('saved in:',test_data_filename)

   
