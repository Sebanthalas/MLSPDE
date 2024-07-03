#########################################################################
# ==============================================================
# CODE  TO GET THE RAW DATA IN A MATLAB FRIENDLY FILE
# THE OUTPUT IS A .m file containing
# x_data - x-values (they should be the same for all curves)
# y_data - 3D array containing the y-values defiing the curves. This array
# should be structured as follows: y_data(i, j, k) contains the y-value 
# corresponding to the i-th x-value (i.e., x(i)), to the j-th random trial, 
# and the k-th curve.
# ==============================================================
import sys
import scipy.io as sio
import numpy as np
import hdf5storage
import time
from numpy import linalg as la
import time, os, argparse, io, shutil, sys, math, socket
import random
import argparse
import threading
import time
import os

if __name__ == '__main__': 
    current_directory   = os.getcwd()
    # The following counter counts how many "failed" experiments you have
    count=0
    NUM_trials = 12  # Number of trials
    NUM_traini = 14 # Number of sets of training points
    NUM_activa = 4  # Number of activaton functions +1
    MMat = 'raw_B'
    for function in ['p']:      
        for problem in ['NSB']:#,'NSB']: Remember for NSB chsnge 12 -> 8 trials
            for d in [4,8]:
                for activation in ['relu','elu','tanh']:      
                    for example in['logKL','aff_S3']: #,'aff_S3','logKL']:
                        VAL =np.zeros((NUM_traini,NUM_trials,NUM_activa))
                        layer=0
                        for DNN_nb_layers in [4,5,10]:
                            layer=layer+1
                            i=0                         
                            for m in [10,20,30,40,50,60,70,80,90,100,200,300,400,500]:
                                i=i+1
                                j=0
                                for trial in [1,2,3,4,5,6,7,8,9,10,11,12]:
                                    j=j+1                                   
                                    key_DNN = (
                                    f"FUN_{function}_/"
                                    f"{str(m).zfill(6)}_pnts_e_tol_5e-08_"
                                    f"optimizer_adam_d_{d}_optimizer_l2_loss_"
                                    f"{activation}_{DNN_nb_layers}x{DNN_nb_layers}0_default")
                                    
                                    scratchdir_resul = os.path.join(
                                                                    current_directory,
                                                                    f"CODE_NSB/SCS_FEM_{problem}/result_data_{example}/{trial}/{key_DNN}",)
                                    DNN_results_filename = os.path.join(
                                        scratchdir_resul,
                                        f"data_m_{str(m).zfill(6)}_deg_1_trial_{trial}_d_{d}_{activation}{DNN_nb_layers}x{DNN_nb_layers}0_final.mat",)
                                    #print(DNN_results_filename)

                                    if not os.path.exists(DNN_results_filename):
                                        count=count+1
                                        print(DNN_results_filename)

                                    else:
                                        raw_data       =  hdf5storage.loadmat(DNN_results_filename)
                                        MinL2_err      = raw_data['minL2_err'][0][0]
                                        #print(MinL2_err)
                                        VAL[i-1,trial-1,0]        = m
                                        VAL[i-1,trial-1,layer]    = MinL2_err
                                        

                        run_data = {}
                        run_data['all_data'] = VAL
                        save_data  = current_directory+'/run_out/raw_'+str(function)+'_data'+str(activation)+'_'+str(problem)+'_'+str(example)+'_''.mat'
                        Mat_result = current_directory+'/plot_files/raw_NSB_p/raw_'+str(function)+'_data_'+str(activation)+'_d_'+str(d)+'_'+str(problem)+'_'+str(example)+'_''.mat'
                        sio.savemat(Mat_result, run_data)
                        print('Saved in:', Mat_result)

                                
print(count)
