# (M)achine (L)earning _ (S)tochastic (PDE)
# Description
This README provides detailed instructions and explanations for running code related to parametric PDE (Partial Differential Equation) simulations, specifically focusing on the Poisson problem, Navier-Stokes-Brinkman (NSB) and Boussinesq equations. It includes information on the folder structure, file contents, functionality, and how to use the code for generating training sample points, testing points, training and testing DNNs (Deep Neural Networks), and generating plots using MATLAB.

It provides step-by-step instructions for running the code locally, changing parameters for different experiments, and generating plots for visualization. It also lists the required packages and their versions for running the code successfully.


 ![ezgif com-animated-gif-maker](https://github.com/Sebanthalas/MLSPDE/assets/21182719/7fc260bf-362b-487d-a4e4-8f3c12363990)


### General Prerequisites

Before using the MATLAB plotting scripts, make sure you have MATLAB installed on your system.
Before using the python scripts, make sure you have python installed on your system along with dependences.

### Mean plots and Shaded Plots

The scripts `plot_book_style.m`, `get_fig_param.m`,  from [https://github.com/simone-brugiapaglia/sparse-hd-book/tree/main/utils/graphics](https://github.com/simone-brugiapaglia/sparse-hd-book/tree/main/utils/graphics), slightly modified for the purpose of plotting these experiments, is used to generate mean  line plots along with shaded plots representing one standard deviation.

These should be placed in a directory of the form "/path_to_sparse-hd-book-main/sparse-hd-book-main/utils" and the pathdef.m file in plot_files should be modified accordingly on line 17: '/path_to_sparse-hd-book-main/sparse-hd-book-main/utils:', ...'


===========================================================================================================================================
## Folder: CODE_P\
### Contains:
- Folder      : meshes , run_out
-- meshes: contain physical discretization meshes.
-- run_out: folder that  will contain some outputs (FEM plots or slurm outputs for cluster use) of the code.
- Python files: callbacks.py, FEM_parametric_PDE_example.py, sympy2fenics.py, TF_parametric_PDE_2FUN.py, PDE_data_poisson.py
- Bash script : Pbatch.sh 

### Functionality
- Creates training sample points approximating u for the parametric Poisson problem with mixed formulation.
- Creates testing points.
- Trains points and tests a fully connected DNN approximating each desired solution of a stochastic PDE.
- Outputs: A folder containing the raw data results and plots.


## Fast Use (as it is) to approximate 12 trials of  u:

1. Ensure all necessary packages are installed and loaded.
2. Run on a terminal: `bash Pbatch.sh`.
   This generates a folder (if not exist) containing sample values for 500 training points, sample values for 1105 testing points for d=4 and 3937 for d=8, and the results of a 4,8-dimensional  4x40 or 10x100 elu,tanh,ReLU DNN architecture using m=[30,60,...,480] training points. 

## Generate plots   
1. Once the results for at least one trial are ready, go back to the PDE_DATA folder and run: `get_raw_data_P_u.py`.
   Make sure to edit any necessary variable depending on the choice of parameters 
   This generates the raw data needed to be processed by MATLAB to generate the plots.
   This code is optimal when using more than one trial and more than one activation function.
2. Run the MATLAB CODE -> `plot_P_u.m` on MATLAB.

3. If everything worked the plots should be in the folder: P_u.

 
===========================================================================================================================================
## Folder: CODE_NSB\


![ezgif com-animated-gif-maker(1)](https://github.com/Sebanthalas/MLSPDE/assets/21182719/8c1eaf75-19b4-4768-acc9-7fbbdc88c616)


### Contains:
- Folder      : meshes , run_out
-- meshes: contain physical discretization meshes.
-- run_out: folder that  will contain some outputs (FEM plots or slurm outputs for cluster use) of the code.
- Python files: callbacks.py, FEM_parametric_PDE_example.py, sympy2fenics.py, TF_parametric_PDE_2FUN.py, PDE_data_NSB.py
- Bash script : Nbatch_u.sh; Nbatch_p.sh

### Functionality
- Creates training sample points approximating (u,p) for the parametric Navier-Stokes-Brinkman (NSB) equations with mixed boundary conditions.
- Creates testing points.
- Trains points and tests a fully connected DNN approximating each desired solution of a stochastic PDE.
- Outputs: A folder containing the raw data results and plots.


## Fast Use (as it is) to approximate 12 trial of the velocity u:

1. Ensure all necessary packages are installed and loaded.
2. Run on a terminal: `bash Nbatch_u.sh`.
   This generates a folder (if not exist) containing sample values for 500 training points, sample values for 1105 testing points for d=4 and 3937 for d=8, and the results of a 4,8-dimensional  4x40 or 10x100 elu,tanh,ReLU DNN architecture using m=[30,60,...,480] training points. 

## Generate plots   
1. Once the results for at least one trial are ready, go back to the PDE_DATA folder and run: `get_raw_data_NSB_u.py`.
   Make sure to edit any necessary variable depending on the choice of parameters 
   This generates the raw data needed to be processed by MATLAB to generate the plots.
   This code is optimal when using more than one trial and more than one activation function.
2. Run the MATLAB CODE -> `plot_NSB_u.m` on MATLAB.

3. If everything worked the plots should be in the folder: NSB_u.

## Follow similar steps for the pressure "p".


===========================================================================================================================================
## Folder: CODE_Boussinesq\
### Contains:
- Folder      : meshes ; run_out
- Python files: callbacks.py, FEM_parametric_PDE_example.py, sympy2fenics.py, TF_parametric_PDE_2FUN.py, PDE_data_B.py
- Bash script : Bbatch_phi.sh, Bbatch_u.sh, Bbatch_p.sh 


### Functionality
- Creates training sample points approximating (u,phi,p) for the parametric Boussinesq equations.
- Creates testing points.
- Trains points and tests a fully connected DNN approximating each desired solution of a stochastic PDE.
- Outputs: A folder containing the raw data results.

## Fast Use (as it is) to approximate 12 trial of the velocity u:

1. Ensure all necessary packages are installed and loaded.
2. Run on a terminal: `bash Bbatch_u.sh`.
   This generates a folder (if not exist) containing sample values for 500 training points, sample values for 1105 testing points for d=4 and 849 for d=8, and the results of a 4,8-dimensional  4x40 or 10x100 elu,tanh,ReLU DNN architecture using m=[30,60,...,480] training points. 

## Generate plots   
1. Once the results for at least one trial are ready, go back to the PDE_DATA folder and run: `get_raw_data_Boussinesq_u.py`.
   This generates the raw data needed to be processed by MATLAB to generate the plots.
   This code is optimal when using more than one trial and more than one activation function.
2. Run the MATLAB CODE -> `plot_Boussinesq_u.m` on MATLAB.

3. If everything worked the plots should be in the folder: NSB_u.



## Fast Use to approximate the temperature phi:

1. Ensure all necessary packages are installed and loaded.
2. Run on a terminal: `bash Bbatch_phi.sh`.
   This generates a folder (if not exist) containing sample values for 500 training points, sample values for 1105 testing points for d=4 and 849 for d=8, and the results of a 4,8-dimensional  4x40 or 10x100 elu,tanh,ReLU DNN architecture using m=[30,60,...,480] training points. 

## Generate plots   
1. Once the results for at least one trial are ready, go back to the PDE_DATA folder and run: `get_raw_data_Boussinesq_phi.py`.
   This generates the raw data needed to be processed by MATLAB to generate the plots.
   This code is optimal when using more than one trial and more than one activation function.
2. Run the MATLAB CODE -> `plot_Boussinesq_phi.m` on MATLAB.

3. If everything worked the plots should be in the folder: NSB_phi.

## Follow similar steps for the  pressure "p".

===========================================================================================================================================
# Explanation of each file (NSB case only, Boussinesq and Poisson is essentially the same)

## Nbatch_u.sh (or Nbatch_u.sh)
### Description
This script is used to run locally/remotely and is the main code that sets parameters, passes them to other scripts, and executes a large loop that submits all the required jobs to the local machine or slurm batch system using "batch". It is structured as follows:

### Usage
- Lines 10-119: Set the parameters; choose which problem and function to approximate;
- Lines 120-163: Initiate the main loops over 3 different affine coefficients; loop over dimensions (d=4,8);
  - Declare how many total training points you will use (L-150) for the NSB;
  - Declare the SG level for each problem and dimension.
  
**WARNING!**
- Line 166: Set a loop over trials. This is only needed when you run locally, and you want to do one or more than one trial.

- Lines 205-238: In case of training, pass the parameters to the Python code (locally). For each coefficient a(x,y), create the training points.
- Lines 240-271: In case of testing, pass the parameters to the Python code (locally). For each coefficient a(x,y), create the testing points.
- Lines 273-320: In case of DNN, pass the parameters to the Python code (locally). Train and test a fully connected DNN.

##PDE_data_NSB.py
### Description
This script is used to generate the sample values, it solves the parametric NSB problem by using FEM for each sample.


### Usage
This is a script usable by FEM_parametric_PDE_example.py


##PDE_data_NSB.py
### Description
Set up the training and testing folders, as well as the parameters for all the training and testing points. Later calls  PDE_data_NSB.py to generate the finite element coefficients and save them in a .m file to be used by TF_parametric_PDE_2FUN.py later.


### Usage
This is a script usable by Nbatch_u.sh (or Nbatch_u.sh).

##TF_parametric_PDE_2FUN.py
### Description
Creates the result folder  and  load the sample values for training and testing. Set up the architecture of the DNN and run callbacks.py to generate the data by training the DNN.


### Usage
This is a script usable by Nbatch_u.sh (or Nbatch_u.sh).


```
===========================================================================================================================================
HOW TO GENERATE FILES FOR PARAVIEW PLOTS - DNN PLOTS Boussinesq (similarly for NSB and Poisson)
====================================================
- Open the file called "callbacks.py"
- Uncomment lines 132--137
- This generates a file "Sol_8000000.vtu" which Paraview can read.
- Then use Paraview to render the data as you like.

INSTRUCTIONS TO RUN LOCALLY (FOR TESTING THE CODE)
==================================================
The code is ready to run locally "as it is" if you have the packages detailed below. However, this runs the entire experiments! (weeks of experiments).

To run just a few to test the code, we suggest changing the following:
- For example, in "Bbatch_u.sh" change line:

91)  DNN_nb_trainingpts[0]="30"  to  DNN_nb_trainingpts[0]="5"
123) for i in {1..2..1}          to  for i in {1..1..1}
137) for d in {0..1..1}          to  for d in {0..0..1}
150) declare  train_pts=1000     to  declare  train_pts=5
152) declare  SG_level=5         to  declare  SG_level=1 
163) for trialss in {1..8..1}    to  for trialss in {1..1..1}
227) for l in {0..2..1}          to  for l in {0..0..1}
229) for k in {0..2..1}          to  for k in {0..0..1}
231) for j in {0..15..1}         to  for j in {0..0..1}

This will generate 1 single trial for: training an elu-4x40DNN using 5 training points and 9 testing points with d=4 dimensions.

Packages:
---------------------------------------------
- tensorflow                   2.12  
- fenics-dijitso               2019.2.0.dev0
- fenics-dolfin                2019.2.0.dev0
- fenics-ffc                   2019.2.0.dev0
- fenics-fiat                  2019.2.0.dev0
- TASMANIAN                    7.3
```

Once TASMANIAN is installed, the path in pathdef.m should be modified accordingly on line 16:      ''/path_to_TASMANIAN/TASMANIAN/InterfaceMATLAB:', ...'
