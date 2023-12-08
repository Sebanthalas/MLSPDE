from dolfin import *
from fenics import *
import numpy as np
from scipy import sparse
import sympy2fenics as sf
import matplotlib.pyplot as plt

"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
# -a(x,y) nalba u (x,y) = f ; u = g on gamma
# Formulation: Mixed formulation  u in L^2 sigma in H(div)
# Boundary conditions:  Natural    = True
#                       Essential  = False
# - tensor products
"""
def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))
def coeff_extr(j,Hh,Usol):
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

def gen_dirichlet_data(z,mesh, Hh, example,i,d,train):
    #================================================================
    # Boundary condition
    #================================================================ 
    coeff_each_m =[]
    u_D      = MyExpression()
    u_str    = '100.0'       
    u_ex     = Expression(str2exp(u_str), degree=1, domain=mesh)  


    # Define variational problem
    Utrial       = TrialFunction(Hh)
    Usol         = Function(Hh)
    W_trainsol   = Function(Hh)
    nn        = FacetNormal(mesh)

    # Define the right hand side and diffusion coefficient
    if example == 'other':
        pi     = str(3.14159265359)
        amean  = str(2)
        string = '1.1 + '
        for j in range(d):
            term   =  str(z[j])+ '*sin('+pi+'*(x+y)/(pow('+str(j)+'+1.0,2)))/(pow('+str(j)+'+1.0,2))'
            string =  string + '+' + term
    string  =  '1.0/('+string+')' 
    a       = Expression(str2exp(string), degree=2, domain=mesh)
    f       = Constant(0.0)
    u, Rsig = split(Usol)
    v, Rtau = TestFunctions(Hh)

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
    if train:
        if i<10:
            plot(uh)
            filename = 'poisson_nonlinear_u'+str(i)+'.png'
            plt.savefig ( filename )
            plt.close()
            plot(Rsigh)
            filename = 'poisson_nonlinear_sigma'+str(i)+'.png'
            plt.savefig ( filename )
            plt.close()
        num_subspaces = W_trainsol.num_sub_spaces()
        print(num_subspaces)
        for j in range(num_subspaces):
            coef_one_trial = coeff_extr(j,Hh,Usol)
            coeff_each_m.append(coef_one_trial)
    else:
        coeff_each_m = Usol.vector().get_local()
        

    
    norm_L2      = sqrt(assemble((uh)**2*dx)) 
    norm_Hdiv    = sqrt(assemble((Rsigh)**2*dx)  +  sqrt(assemble((div(Rsigh) )**2*dx) )  )

    


    return coeff_each_m, norm_L2, norm_Hdiv
