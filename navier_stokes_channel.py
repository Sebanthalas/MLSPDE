# ==============================================================
# The sympy2fenics.py file is needed to compute in a simple way:
# - derivatives
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

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))
# ==============================================================
# Boundary conditions
# ==============================================================
class MyExpression(UserExpression):
  def eval(self, value, x):
    if x[1] >= 1- DOLFIN_EPS: #-my_LK_fun(x[0],z,d):
      value[0] = 0.0
    elif x[1] <= 0.0+ DOLFIN_EPS: #+my_LK_fun(x[0],z,d):
      value[0] = 0.0
    elif x[0] > 1- DOLFIN_EPS:
      value[0] = 0.0 #( 1.0 - ( abs(abs(x[1]) - 0.5 ))/(0.5)  ) 
    elif x[0] < 0.0+ DOLFIN_EPS:
      value[0] = 0.0 #( 1.0 - ( abs(abs(x[1]) - 0.5 ))/(0.5)  ) 
    elif ( (x[0] > 0.0625 - DOLFIN_EPS) and (x[0] < 0.1875 +DOLFIN_EPS) and (x[1] > 0.4375 - DOLFIN_EPS) and (x[1] < 0.5625 + DOLFIN_EPS) ):
      value[0] = 0.1   
    else:
      value[0] = 0.0
  def value_shape(self):
    return (1,)

# ==============================================================
# The following gives the informations about the solver. 
# ==============================================================
parameters["form_compiler"]["representation"]    = "uflacs"
parameters["form_compiler"]["cpp_optimize"]      = True
parameters["form_compiler"]["quadrature_degree"] = 4
list_linear_solver_methods()
# ==============================================================
# Define the parameters for the PDE and SG points
# ==============================================================
# All parameters
default_parameters = {
    'trial_'           :0,
    'num_train_samples':1, #20
    'num_test_samples': 1, #50
    'mesh_op'         : 2, # choose mesh refiniment, 1,2,3,4. Default=2
    'FE_degree'       : 1, # Use P1 for 2D or above
    'example'         : 'other',
    'input_dim'       : 4
}
params = default_parameters
exact_FE_soln   = 0
using_SG_points = 1
#random samples
_trial  = params['trial_']
np_seed = _trial
np.random.seed(np_seed)
#================================================================
# Choose the mesh: there is 4  barycentric refininements. 
#                  - use plot(mesh) to see them.
#================================================================
nk        = params['mesh_op']
example   = params['example']
input_dim = params['input_dim']
meshname  = "meshes/obstac%03g.xml"%nk
mesh      = Mesh(meshname)
nn        = FacetNormal(mesh)
#plot(mesh)
#filename = 'poisson_nonlinear_gradient.png'
#plt.savefig ( filename )
m =1
input_dim = params['input_dim']
d         = input_dim       # parametric dimension
y_in_train = 2.0*np.random.rand(d,m) - 1.0  # training points 
for i in range(m):
    t_start = time.time()
    z       = y_in_train[:,i]
    string  = 1/150
    a       = Constant(string)
    #================================================================
    # Boundary condition
    #================================================================ 
    u_D      = MyExpression()
    u_str    = '1.0'       
    u_ex     = Expression(str2exp(u_str), degree=1, domain=mesh)    
    #================================================================
    #  *********** Finite Element spaces ************* #
    #================================================================
    deg = params['FE_degree']
    Pk  = FiniteElement('DG', mesh.ufl_cell(), deg)
    RTv = FiniteElement('RT', mesh.ufl_cell(), deg+1)
    Hh  = FunctionSpace(mesh, MixedElement([Pk,RTv]))
    nvec = Hh.dim()
    f = Constant(0.0)
    #================================================================
    # *********** Trial and test functions ********** #
    #================================================================
    Utrial = TrialFunction(Hh)
    Usol   = Function(Hh)
    W_trainsol   = Function(Hh)
    u, Rsig = split(Usol)
    v, Rtau = TestFunctions(Hh)
    # ********** Boundary conditions ******** #
    # All Dirichlet BCs become natural in this mixed form 
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
    plot(-1*Rsigh)
    filename = 'poisson_nonlinear_gradient.png'
    plt.savefig ( filename )
