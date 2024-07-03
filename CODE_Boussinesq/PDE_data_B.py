import sys
from dolfin import *
from fenics import *
import numpy as np
from scipy import sparse
import sympy2fenics as sf
import matplotlib.pyplot as plt
import os


def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

class MyExpression(UserExpression):
  def eval(self, value, x):
    if abs(x[2] - 1 ) < DOLFIN_EPS:
      value[0] = 0.0
    elif abs (x[2]    ) < DOLFIN_EPS:
      value[0] = 1.0
    elif abs (x[1]-1    ) < DOLFIN_EPS:
      value[0] = 0.0
    elif abs (x[1]    ) < DOLFIN_EPS:
      value[0] = 0.0
    elif abs (x[0]-1    ) < DOLFIN_EPS:
      value[0] = 0.0
    elif abs (x[0]    ) < DOLFIN_EPS:
      value[0] = 0.0
    else:
      value[0] = 0.0
  def value_shape(self):
    return ()

def gen_dirichlet_data_B(z,mesh, Hh, VVh, Ph, example,i,d):
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["quadrature_degree"] = 4
    #list_linear_solver_methods()

    fileO = XDMFFile("outputs/out-Ex03Convergence.xdmf")
    fileO.parameters["functions_share_mesh"] = True
    fileO.parameters["flush_output"] = True

    # ****** Constant coefficients ****** #
    nn    = FacetNormal(mesh)
    g     = Constant((0.,0.,-1.0))
    Id    = Constant(((1,0,0),(0,1,0),(0,0,1)))
    u_D   = MyExpression(degree = 3)
    u_str = '(  1.0  , 1.0    ,  0.0   )'
    
    
    hvec = []; nvec = [];  
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
            term   =  str(z[j])+ '*sin(1.0*'+pi+'*(x)*('+str(j)+'+1.0) )*pow('+str(j)+'+1.0,-3/2)'
            string =  string + '+' + term

    elif example == 'aff_F9': 
        pi     = str(3.14159265359)
        string = '1.89 + '
        for j in range(d):
            term   =  str(z[j])+ '*sin('+pi+'*z*('+str(j)+'+1.0) )*pow('+str(j)+'+1.0,-9/5)'
            string =  string + '+' + term

    else:
      print('error')


    #===================================================================================================  
    #===================================================================================================  
    # *********** Variable coefficients ********** #
    pi     = str(3.14159265359)
    string2 = '1.89 + '
    for j in range(d):
        term   =  str(z[j])+ '*sin('+pi+'*z*('+str(j)+'+1.0) )*pow('+str(j)+'+1.0,-9/5)'
        string2 =  string2 + '+' + term

    #=================================================================================================== 


    string    =  string 
    string2   =  string2 
    nu        = Expression(str2exp(string), degree=2, domain=mesh)
    nu2       = Expression(str2exp(string2), degree=2, domain=mesh)
    K1 = Expression((("exp(-x[0])","0.","0"),\
                 ("0.","exp(-x[1])","0"),\
                 ("0.","0.", "exp(-x[2])")), \
                degree=3, domain = mesh)
    K1       = K1*nu2
    mu       = lambda phi1: (0.1+exp(-phi1))*nu
    phi1_str = 'exp(4*(-(x - 0.5)*(x - 0.5)  - (y - 0.5)*(y - 0.5))  )'
    u_ex     = Expression(str2exp(u_str), degree=3, domain=mesh)
    phi1_ex  = Expression(str2exp(phi1_str), degree=3, domain=mesh)

    # *********** Variable coefficients ********** #
    # *********** Trial and test functions ********** #

    Utrial = TrialFunction(Hh)
    Usol   = Function(Hh)
    u, t11, t12, t13, t21, t22, t23, t31, t32, Rsig1, Rsig2, Rsig3, phi1, t1, sigma1, xi = split(Usol)
    v, s11, s12, s13, s21, s22, s23, s31, s32, Rtau1, Rtau2, Rtau3, psi1, s1,   tau1, ze = TestFunctions(Hh)

    t=as_tensor(((t11,t12,t13),(t21,t22,t23),(t31,t32,-t11-t22)))
    s=as_tensor(((s11,s12,s13),(s21,s22,s23),(s31,s32,-s11-s22)))

    phi   = phi1
    sigma = as_tensor((Rsig1,Rsig2,Rsig3))
    tau   = as_tensor((Rtau1,Rtau2,Rtau3))
      

    # ********** Boundary conditions ******** #

    # All Dirichlet BCs become natural in this mixed form

    # *************** Variational forms ***************** #

    # flow equations

    Aphi = 2*mu(phi1)*inner(sym(t),s)*dx
    C  = 0.5*dot(t*u,v)*dx - 0.5*inner(dev(outer(u,u)),dev(s))*dx
    Bt = - inner(sigma,s)*dx - dot(div(sigma),v)*dx
    B  = - inner(tau,t)*dx   - dot(u,div(tau))*dx
    F  =    phi1*dot(g,v)*dx 
    G  = - dot(tau*nn,u_ex*u_D)*ds


    # temperature
    Aj1 = dot(K1*t1,s1)*dx
    Cu1 = 0.5*psi1*dot(t1,u)*dx - 0.5*phi1*dot(u,s1)*dx
    B1t = - dot(sigma1,s1)*dx - psi1*div(sigma1)*dx
    B1  = - dot(tau1,t1)*dx - phi1*div(tau1)*dx

    G1 = - dot(tau1,nn)*phi1_ex*u_D*ds


    # zero (or, in this case, exact) average of trace
    Z  = (tr(2*sigma+outer(u,u)))* ze * dx + tr(tau) * xi * dx

    FF = Aphi + C + Bt + B \
         - F - G \
         + Aj1 + Cu1 + B1t + B1 \
         - G1 \
         + Z

    Tang = derivative(FF, Usol, Utrial)
    problem = NonlinearVariationalProblem(FF, Usol, J=Tang) #no BCs
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']                    = 'newton'#or snes
    solver.parameters['newton_solver']['linear_solver']      = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
    solver.parameters['newton_solver']['maximum_iterations'] = 25

    solver.solve()
    uh, t11h, t12h, t13h, t21h, t22h, t23h, t31h, t32h, Rsigh1, Rsigh2, Rsigh3, phi1h, t1h, sigma1h, xih = Usol.split()

    #th     = as_tensor(((t11h,t12h,t13h),(t21h,t22h,t23h),(t31h,t32h,-t11h-t22h)))
    phih   = phi1h
    sigmah = as_tensor((Rsigh1,Rsigh2,Rsigh3))

    cch = assemble(-1./(6.*0.25)*dot(uh,uh)*dx)

    # dimension-dependent
    ph    = project(-1./6.*tr(2.*sigmah+2.*cch*Id+outer(uh,uh)),Ph)
    uh    = project(uh,VVh)
    phi1h = project(phi1h,Ph)
    u_coefs   = np.array(uh.vector().get_local())
    p_coefs   = np.array(ph.vector().get_local())
    phi_coefs = np.array(phi1h.vector().get_local())
        
    norm_L4         = sqrt(sqrt(assemble( ((uh)**2)**2*dx)))
    norm_L2         = sqrt(assemble((ph)**2*dx))  
    norm_L4phi      = sqrt(sqrt(assemble( ((phi1h)**2)**2*dx)))
     
    folder1 = str('run_out/uh_'+str(i).zfill(1)+'.pvd')
    vtkfile = File(folder1)
    vtkfile << uh
    folder2 = str('run_out/ph_'+str(i).zfill(1)+'.pvd')
    vtkfile = File(folder2)
    vtkfile << ph
    folder3 = str('run_out/phih_'+str(i).zfill(1)+'.pvd')
    vtkfile = File(folder3)
    vtkfile << phi1h
    print (" ****** Total DoF u = ", VVh.dim())
    print (" ****** Total DoF p = ", Ph.dim())
    print (" ****** Total DoF phi = ", Ph.dim())
 

    


    return u_coefs,p_coefs, phi_coefs, norm_L4, norm_L2, norm_L4phi



