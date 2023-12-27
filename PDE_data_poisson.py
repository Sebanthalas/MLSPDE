import sys
from dolfin import *
from fenics import *
import numpy as np
from scipy import sparse
import sympy2fenics as sf
import matplotlib.pyplot as plt
import logging


def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))


class MyExpression(UserExpression):
  def eval(self, value, x):
    if x[1] >= 1- DOLFIN_EPS: 
      value[0] = 1.0*x[0]*(1.-x[0])
    elif x[1] <= 0.0+ DOLFIN_EPS: 
      value[0] = 1.0*x[0]*(1.-x[0])
    elif x[0] > 1- DOLFIN_EPS:
      value[0] = 1.0*x[1]*(1.-x[1])
    elif x[0] < -0.0+ DOLFIN_EPS:
      value[0] = 0.0 
    elif ( (x[0] > 0.0625 - DOLFIN_EPS) and (x[0] < 0.1875 +DOLFIN_EPS) and (x[1] > 0.4375 - DOLFIN_EPS) and (x[1] < 0.5625 + DOLFIN_EPS) ):
      value[0] = -0.0   
    else:
      value[0] = 0.0
  def value_shape(self):
    return (1,)

def gen_dirichlet_data_poisson(z,mesh, Hh, example,i,d,train):
    #================================================================
    # Boundary condition
    #================================================================ 
    u_D      = MyExpression()
    u_str    = '10.0'       
    u_ex     = Expression(str2exp(u_str), degree=1, domain=mesh)  


    # Define variational problem
    Utrial       = TrialFunction(Hh)
    Usol         = Function(Hh)
    W_trainsol   = Function(Hh)
    nn        = FacetNormal(mesh)

    # Define the right hand side and diffusion coefficient
    if example == 'other':
        pi     = str(3.14159265359)
        string = '1.9 + '
        for j in range(d):
            term   =  str(z[j])+ '*sin('+pi+'*(x+y)*('+str(j)+'+1) )/(pow('+str(j)+'+1.0,9/5))'
            string =  string + '+' + term
    string  =  '1.0/('+string+')' 
    a       = Expression(str2exp(string), degree=2, domain=mesh)
    f       = Constant(0.0)
    u, Rsig = split(Usol)
    v, Rtau = TestFunctions(Hh)
    logging.warning('Watch out!')

    # Weak formulation  
    BigA  = a*dot(Rsig,Rtau)*dx
    BigB1 = u*div(Rtau)*dx
    BigB2 = v*div(Rsig)*dx
    F = -f*v*dx
    
    G = (Rtau[0]*nn[0]+Rtau[1]*nn[1])*u_ex*u_D[0]*ds
    FF = BigA + BigB1 + BigB2  - F - G 
    Tang = derivative(FF, Usol, Utrial)
    solve(FF == 0, Usol, J=Tang)
    uh,Rsigh = Usol.split()
    u_coefs = np.array(uh.vector().get_local())

    norm_L2      = sqrt(assemble((uh)**2*dx)) 
    norm_Hdiv    = sqrt(assemble((Rsigh)**2*dx)  +  sqrt(assemble((div(Rsigh) )**2*dx) )  )

    


    return u_coefs, norm_L2, norm_Hdiv
