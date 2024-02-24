# (M)achine (L)earning _ (S)tochastic (PDE)
Sebastian Moraga Scheuermann, Simon Fraser University, 2024.

This code implements the fully connected Deep Neural Network (DNN) architectures considered in the thesis 
"Optimal and efficient algorithms for learning high-dimensional, Banach-valued functions from limited samples".
===============================

What the code can do:
===============================
-It can create the training points for the parametric poisson equation with Dirichlet boundary conditions and the parametric Navier-Stokes-Brikman equations.

For every parameter y, the formulation for the Poisson equation is a mixed formulation in a Hilbert space. The NSB equations use a mixed formulation with solutions in Banach spaces.

-It can create the testing points.

-It can train points and test a fully connected DNN approximating each desired solution of a stochastic PDE.


Explanation of each file:
===============================
(The following script is to run locally/slurm):
_____________________________________________________________________________________________________________________________________________________________________________________________________________
 Pbatch_run.sh    

It is the main code that sets parameters, passes them to the other scripts, and executes a large loop that submits all of the required jobs to the local machine or slurm batch system using “sbatch".
The structure is as follows:

Line 10  - 146: Set the parameters; choose which problem and function to approximate;

Line 147 - 202: Initiate the main loops over 3 different affine coefficients; loop over 3 dimensions (d=4,8,16) ; 

                Declare how many total training points you will use (L-177) for the Poisson problem and (L189) for the NSB problem;
                
                Declare the SG level for each problem and dimension.
                
WARNING !  
=========
(w)-> Line 204 : Set a loop over trials. This is only needed when you run locally, and you want to do one or more than one trial. If you are running on slurm, this is already done in the PTF.sh files, so you should comment on this line in this last case.     

_____________________________________________
Line 205 - 238:  In case of training. Pass the parameters to the python code (locally) or the following sbatch file (slurm),e.g., "PTF.sh".           For each coefficient a(x,y) creates the training points.

Line 240 - 267:  In case of testing.  Pass the parameters to the python code (locally) or the following sbatch file (slurm),e.g., "PTeF.sh".          For each coefficient a(x,y) creates the testing points.

Line 205 - 238:  In case of DNN.      Pass the parameters to the python code (locally) or the following sbatch file (slurm),e.g., "AllTrials_DNN.sh". Trains and test a fully connected DNN.
                
__________
(The following scripts only can run on slurm):
____________________________________________________________________________________________________________________________________________________________________________________________________________
PTS.sh (PoissonTrainingScript)
It is one of the three scripts from L205-238 in Pbatch_run.sh. This script gets executed to create all the trials. It sets up the training using the group of commands at the top for sbatch and then iterates over groups of 4  "SLURM_ARRAY_TASK_ID" tasks which are run to create 12 trials in total.

That is, it creates an array of 4 IDs. Then iterates to send 4 jobs to each ID. Basically, to create 12 trials for the passion:

Line 28 : Trials from 0 to 3

Line 29 : Trials from 4 to 7

Line 30 : Trials from 8 to 11

It also passes the final command line arguments to the next file and creates logs in the “run_out" folder in the same directory as the scripts.
_____________________________________________________________________________________________________________________________________________________________________________________________________________
PTeS.sh (PoissonTestingScript)
____________
It is one of the three scripts from L240-267 in Pbatch_run.sh. Since you don't need different trials for testing, it only creates the testing points. 

It also passes the final command line arguments to the following file and creates logs in the “run_out" folder in the same directory as the scripts.
_____________________________________________________________________________________________________________________________________________________________________________________________________________
AllTrials_DNN.sh (Pass the trials to the DNNs)
____

It is one of the three scripts from L205-238 in Pbatch_run.sh. It sets up the run using the group of commands at the top for sbatch and then iterates over groups of 4 tasks, which are run concurrently on the 4 large GPUs on the compute nodes until 12 trials are run for the passion equation and 8 for the NSB. 

It also passes the final command line arguments to the following file and creates logs in the “run_out" folder in the same directory as the scripts.

_____________________________________________________________________________________________________________________________________________________________________________________________________________
DNN_trial.sh (Runs the DNN )
_____

It is the script that actually runs the python code.

It specifies which GPU to use (argument 28, an integer 0, 1, 2, or 3), and passes the command line arguments to the python code, which uses the argparser to create necessary variables.


PYTHON CODES
==============

-FEM_parametric_PDE_example: script to create training and testing points.

-PDE_data_NSB: script to obtain the solution via FEniCS of a single sample for NSB eq.

-PDE_data_poisson: script to obtain the solution via FEniCS of a single sample for Poisson eq.

-TF_parametric_PDE_2FUN: script to set the DNN architecture and use of parameters.

-callbacks: script with the instructions for the DNN and save the files.

-sympy2fenics: script used to write the differential equations in a simplified  way.


INSTRUCTIONS TO RUN LOCALLY
===============================
The code is ready to run locally "as it is"if you have the packages detailed below.

Packages:
---------------------------------------------
tensorflow                   2.14.0  or (2.12 CEDAR using CUDA)

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
