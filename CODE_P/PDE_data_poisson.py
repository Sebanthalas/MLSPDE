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
    if x[1] <= 0.0+ DOLFIN_EPS: 
      value[0] = 1.0
    #elif x[1] <= 0.0+ DOLFIN_EPS: 
    #  value[0] = 0.0
    #elif x[0] > 0.6+ DOLFIN_EPS:
    #  value[0] = 1.0 #256.0*pow(1.0*x[1]*(1.-x[1]),4)
    #elif x[0] < 0.4- DOLFIN_EPS:
    #  value[0] = 1.0 
    #elif ( (x[0] > 0.4 - DOLFIN_EPS) and (x[0] < 0.6 +DOLFIN_EPS) and (x[1] < 0.6 + DOLFIN_EPS) ):
    #  value[0] = 0.0   
    else:
      value[0] = 0.0
  def value_shape(self):
    return (1,)

def gen_dirichlet_data_poisson(z,mesh, Hh, example,i,d,train):
    #================================================================
    # Boundary condition
    #================================================================ 
    u_D      = MyExpression()
    u_str    = '0.5'       
    u_ex     = Expression(str2exp(u_str), degree=1, domain=mesh)  


    # Define variational problem
    Utrial       = TrialFunction(Hh)
    Usol         = Function(Hh)
    W_trainsol   = Function(Hh)
    nn        = FacetNormal(mesh)

    # Define the right hand side and diffusion coefficient
    #===================================================================================================  
    # *********** Variable coefficients ********** #


    if example =='logKL':
        pi = 3.14159265359
        pi_s = str(pi)
        L_c = 1.0/8.0
        L_p = np.max([1.0, 2.0*L_c])
        L_c_s = str(L_c)
        L_p_s = str(L_p)
        L = L_c/L_p
        L_s = str(L)

        string = '1.0+sqrt(sqrt(' + pi_s + ')*' + L_s + '/2.0)*' + str(z[0])
        for j in range(2, d):
            term = str(z[j-1]) + '*sqrt(sqrt(' + pi_s + ')*' + L_s + ')*exp(-pow(floor(' 
            term = term + str(j) + '/2.0)*' + pi_s + '*' + L_s + ',2.0)/8.0)'
            if j % 2 == 0:
                term = term + '*sin(floor(' + str(j) + '/2.0)*' + pi_s + '*x/' + L_p_s + ')'
            else:
                term = term + '*cos(floor(' + str(j) + '/2.0)*' + pi_s + '*x/' + L_p_s + ')'

            string = string + '+' + term
        string = 'exp(' + string + ')'

    elif example == 'aff_S3':
        pi     = str(3.14159265359)
        string = '2.62 + '
        for j in range(d):
            term   =  str(z[j])+ '*sin('+pi+'*x*('+str(j)+'+1.0) )*pow('+str(j)+'+1.0,-3/2)'
            string =  string + '+' + term

    elif example == 'aff_F9': 
        pi     = str(3.14159265359)
        string = '1.89 + '
        for j in range(d):
            term   =  str(z[j])+ '*sin('+pi+'*x*('+str(j)+'+1.0) )*pow('+str(j)+'+1.0,-9/5)'
            string =  string + '+' + term

    else:
      print('error')

    #===================================================================================================  


    string  =  'pow('+string+',-1)' 
    a       = Expression(str2exp(string), degree=1, domain=mesh)
    f       = Constant(10.0)
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
    #folder1 = str('run_out/uh_REA.pvd')
    #vtkfile = File(folder1)
    #vtkfile << uh
    #if i<20:
    #    plot(uh)
    #    filename = 'run_out/pois_nonlinear_u'+str(i)+'.png'
    #    plt.savefig ( filename )
    #    plt.close()
    


    return u_coefs, norm_L2, norm_Hdiv
