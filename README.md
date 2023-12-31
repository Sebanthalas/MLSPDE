# (M)achine (L)earning _ (S)tochastic (PDE)


This code implements the fully connected Deep Neural Network (DNN) architectures considered in the thesis "[in preparation]".

It is able to create the training points for the parametric poisson equation with Dirichlet boundary conditions and the parametric Navier-Stokes-Brikman equations.
For every parameter y, The formulation for the Poisson equation is mixed formulation in a Hilbert space. The NSB equations use a mixed formulation with solutions in Banach spaces.

It is able to create the testing points.

It is able to the train points and tests a fully connected DNN approximating each desired solution of a stochastic PDE.

Packages:
---------------------------------------------
tensorflow                   2.14.0
fenics-dijitso               2019.2.0.dev0
fenics-dolfin                2019.2.0.dev0
fenics-ffc                   2019.2.0.dev0
fenics-fiat                  2019.2.0.dev0
TASMANIAN                    7.3

Examples:
Poisson
--------------------------------------------------
![ezgif com-animated-gif-maker](https://github.com/Sebanthalas/parametric_PDE_approx_viaDNN/assets/21182719/248738b8-638d-4380-918e-9ce015b668c5)


NSB
-----------------------------------------------------
![nonlinear_uh9](https://github.com/Sebanthalas/parametric_PDE_approx_viaDNN/assets/21182719/689a6767-1b97-449e-877c-1ebdf47712d3)
