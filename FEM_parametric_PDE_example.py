
#########################################################################
########################SETUP FEM PROBLEM DATA########################### 
#########################################################################
# ==============================================================
# Code solving the possion problem 
# -a(x,y) \nalba u (x,y) = f ; u = g on \gamma
# Formulation: Mixed formulation  u \in L^2 sigma \in H(div)
# Boundary conditions:  Natural    = True
#                       Essential  = False
# - tensor products
# ==============================================================
import sys
sys.path.append("/home/sebanthalas/TASMANIAN_v7.3/lib/python3.10/site-packages")
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
def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))
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
def coeff_extr(j):
    #This exctracts the coefficient of the different spaces
    # j is the index of the space:
    # j=0 the vector space for uh
    # j=1 the space for  component of sigma
    # For this to work the nature of the space has to be the same
    W  = Function(Hh)
    #Getting the exact DOF location
    DoF_map   = Hh.sub(j).dofmap()
    DoF_index = DoF_map.dofs()
    AUX1 = W.vector().get_local()    # empty DoF vector 
    AUX2 = Usol.vector().get_local() # All DoF
    AUX1[DoF_index] = AUX2[DoF_index]                # corresponding assignation
    W.vector().set_local(AUX1)       # corresponding assignation to empy vector
    coeff_vector = np.array(W.vector().get_local()) 
    return coeff_vector
class MyExpression(UserExpression):
  def eval(self, value, x):
    if x[1] >= 1- DOLFIN_EPS: 
      value[0] = 0.0
    elif x[1] <= 0.0+ DOLFIN_EPS: 
      value[0] = 0.0
    elif x[0] > 1- DOLFIN_EPS:
      value[0] = 0.0 
    elif x[0] < 0.0+ DOLFIN_EPS:
      value[0] = 0.0 
    elif ( (x[0] > 0.0625 - DOLFIN_EPS) and (x[0] < 0.1875 +DOLFIN_EPS) and (x[1] > 0.4375 - DOLFIN_EPS) and (x[1] < 0.5625 + DOLFIN_EPS) ):
      value[0] = 0.1   
    else:
      value[0] = 0.0
  def value_shape(self):
    return (1,)

 
#===============================================================
if __name__ == '__main__': 
  start_time = time.time()
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
  args = parser.parse_args()
 
  set_log_level(args.fenics_log_level)

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
  meshname  = "meshes/obstac%03g.xml"%nk
  mesh      = Mesh(meshname)
  nn        = FacetNormal(mesh)
  All_Train_coeff = []
  All_Test_coeff  = []
  _L2unorm_train  = []
  _H2snorm_train  = []
  _L2unorm_test   = []
  _H2snorm_test   = []


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
  # Boundary condition
  #================================================================ 
  u_D      = MyExpression()
  u_str    = '1.0'       
  u_ex     = Expression(str2exp(u_str), degree=1, domain=mesh)    
  # set the number of training points 
  m     = args.nb_train_points
  # set the maximum number of training points 
  m_max = args.max_nb_train_points

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
    #plt.scatter(y_in_test[0,:], y_in_test[1,:])
    #plt.show()

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

  # unique key for naming results
  key = str(m).zfill(6) + '_pnts_%2.2e' % (error_tol) + '_tol_'+str(d)+'_d'

  # Save the training and test#
  scratchdir    = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/' + unique_run_ID + '_' + args.example + '/' + key
  projectdir    = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/' + unique_run_ID + '_' + args.example 
  result_folder = scratchdir
  scratch_folder = projectdir
  run_root_folder = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/' + unique_run_ID + '_' + args.example+ '/' + key

  if not os.path.exists(result_folder):
      try:
          os.makedirs(result_folder)    
      except FileExistsError:
        print('===================================================================')
        print('skipping making', result_folder)
        print('===================================================================')

  if not os.path.exists(scratch_folder): 
      try:
          os.makedirs(scratch_folder)
      except FileExistsError:
          print('skipping making', scratch_folder)
  print('===================================================================')
  print('Saving results to', result_folder)
  print('===================================================================')
 
  
  run_data_filename  = result_folder + '/trial_' + str(trial) + '_run_data.mat'
  results_filename   = result_folder + '/trial_' + str(trial) + '_results.mat'
  test_data_filename = run_root_folder + '/test_data' + str(m_test).zfill(8) + '_' + args.test_pointset + '_pts_test_data.mat'
  if args.test_pointset == 'CC_sparse_grid':
    test_results_filename = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/' + unique_run_ID + '_' + args.example + '/'+str(d)+'d_'+str(args.SG_level)+'_SG_test_data.mat'
  else:
    test_results_filename = '/home/sebanthalas/Documents/NE_NOV23/results/scratch/SCS_FEM_'+args.problem+'/' + unique_run_ID + '_' + args.example + '/'+str(d)+'d_'+str(m_test)+'_mUR_test_data.mat'


  # See if this is usefl later on
  m_test_check = m_test

  if args.train:
    print('       ____________________________________________________________________')
    print('                                Beginning training                         ')
    print('       ____________________________________________________________________')

    y_in_train = np.transpose(np.random.uniform(-1.0,1.0,(m,d)))
    U = []
    print('Using uniform random training points with m =', m)
    K = 0
    print('Generating the training data')
    
    # Generate the training data
    for i in range(m):
      coeff_each_m = []
      t_start = time.time()

      # get the training data inputs 
      z = y_in_train[:,i]
      #string  = 1/150
      string  =  '1/(6.0+(' + str(z[0]) + '*x+' + str(z[1]) + '*y+' + str(z[2]) + '+'+ str(z[3]) +' ))' 
      a       = Expression(str2exp(string), degree=2, domain=mesh)
      f       = Constant(0.0)
      u, Rsig = split(Usol)
      v, Rtau = TestFunctions(Hh)
      # *************** Variational forms ***************** #
      #================================================================
      # flow equations
      #================================================================   
      # Weak formulation  
      BigA  = a*dot(Rsig,Rtau)*dx
      BigB1 = u*div(Rtau)*dx
      BigB2 = v*div(Rsig)*dx
      F = -f*v*dx
      G = (Rtau[0]*nn[0]+Rtau[1]*nn[1])*u_ex*u_D[0]*ds
      #Stiffness matrix
      FF = BigA + BigB1 + BigB2  - F - G 
      Tang = derivative(FF, Usol, Utrial)
      solve(FF == 0, Usol, J=Tang)
      uh,Rsigh = Usol.split()
      #plot(uh)
      #filename = 'poisson_nonlinear_gradient'+str(i)+'.png'
      #plt.savefig ( filename )
      
      num_subspaces = W_trainsol.num_sub_spaces()
      
      for j in range(num_subspaces):
        coef_one_trial = coeff_extr(j)
        coeff_each_m.append(coef_one_trial)
      All_Train_coeff.append(coeff_each_m)
      norm_L2      = sqrt(assemble((uh)**2*dx)) 
      norm_Hdiv    = sqrt(assemble((Rsigh)**2*dx)  +  sqrt(assemble((div(Rsigh) )**2*dx) )  )
      _L2unorm_train.append(norm_L2)
      _H2snorm_train.append(norm_Hdiv)
      
      print('====================================================================')
      print('i = ', i, 'L2u=  %2.4g ' % norm_L2,'y_train= ', z)
      print('====================================================================')
    run_data = {}
    run_data['d']              = d
    run_data['K']              = K
    run_data['m_max']          = m_max
    run_data['m_train']        = m
    run_data['y_in_train_data']= y_in_train
    run_data['mesh_op']        = nk
    run_data['FE_degree']      = deg
    run_data['All_Train_coeff'] = All_Train_coeff
    run_data['_L2unorm_train']       = _L2unorm_train
    run_data['_H2snorm_train']       = _H2snorm_train
    run_data['fenics_mesh_coords']       = np.array(mesh.coordinates())
    run_data['fenics_mesh_cells']        = np.array(mesh.cells())
    run_data['fenics_mesh_num_cells']    = np.array(mesh.num_cells())
    run_data['fenics_mesh_num_edges']    = np.array(mesh.num_edges())
    run_data['fenics_mesh_num_vertices'] = np.array(mesh.num_vertices())
    run_data['fenics_mesh_hmax']         = np.array(mesh.hmax())
    run_data['fenics_mesh_hmin']         = np.array(mesh.hmin())
    run_data['fenics_mesh_rmax']         = np.array(mesh.rmax())
    run_data['fenics_mesh_rmin']         = np.array(mesh.rmin())
    sio.savemat(run_data_filename, run_data)
    print('saved in:',run_data_filename)
  
  else:
    print('       ____________________________________________________________________')
    print('                                Beginning testing data                     ')
    print('       ____________________________________________________________________')
    print('Generating the testing data m_test=',m_test)
    for i in range(m_test):
      coeff_each_m = []
      z = y_in_test[:,i]
      string  =  '1/(6.0+(' + str(z[0]) + '*x+' + str(z[1]) + '*y+' + str(z[2]) + '+'+ str(z[3]) +' ))' 
      a       = Expression(str2exp(string), degree=2, domain=mesh)
      f       = Constant(0.0)
      u, Rsig = split(Usol)
      v, Rtau = TestFunctions(Hh)
      # *************** Variational forms ***************** #
      #================================================================
      # flow equations
      #================================================================   
      # Weak formulation  
      BigA  = a*dot(Rsig,Rtau)*dx
      BigB1 = u*div(Rtau)*dx
      BigB2 = v*div(Rsig)*dx
      F = -f*v*dx
      G = (Rtau[0]*nn[0]+Rtau[1]*nn[1])*u_ex*u_D[0]*ds
      #Stiffness matrix
      FF = BigA + BigB1 + BigB2  - F - G 
      Tang = derivative(FF, Usol, Utrial)
      solve(FF == 0, Usol, J=Tang)
      uh,Rsigh = Usol.split()
      #plot(uh)
      #filename = 'poisson_nonlinear_gradient_test'+str(i)+'.png'
      #plt.savefig ( filename )
      
          
      coeff_each_m = Usol.vector().get_local()
      All_Test_coeff.append(coeff_each_m)

      norm_L2      = sqrt(assemble((uh)**2*dx)) 
      norm_Hdiv = sqrt(assemble((Rsigh)**2*dx)  +  sqrt(assemble((div(Rsigh) )**2*dx) )  )  
      _L2unorm_test.append(norm_L2)
      _H2snorm_test.append(norm_Hdiv)
      if i == 0:
        K = len(coeff_each_m)
        print('FE degrees of freedom K = ', K)

      print('====================================================================')
      print('i = ', i, 'L2u=  %2.4g ' % norm_L2,'y_test= ', z)
      print('====================================================================')
    test_data ={}
    test_data['m_test']         = m_test
    sio.savemat(test_results_filename, test_data)
    run_data = {}
    run_data['d']              = d
    run_data['K']              = K
    run_data['m_max']          = m_max
    run_data['m_test']         = m_test
    run_data['y_in_test_data'] = y_in_test
    run_data['mesh_op']        = nk
    run_data['FE_degree']      = deg
    run_data['All_Test_coeff'] = All_Test_coeff
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
    sio.savemat(test_data_filename, run_data)
    print('saved in:',test_data_filename)

   
