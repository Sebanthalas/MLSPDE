import os, time
import tensorflow as tf
import tensorflow.keras.backend as KerasB
import numpy as np
from dolfin import *
from fenics import *
import scipy.io as sio
import matplotlib.pyplot as plt

class EarlyStoppingPredictHistory(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing. Also tests the model while training, using an update ratio to prevent testing too often. Note: the best weights for this checkpoint are held in memory, and restored after training if the loss for the checkpoint is better than the current loss.

  Arguments:
        run_data: dictionary of options for the callback and running
  """

    def __init__(self, run_data):
        super(EarlyStoppingPredictHistory, self).__init__()
        self.run_data      = run_data
        self.fenics_params = self.run_data['fenics_params']
        self.best_loss     = 1e12
        self.best_loss_epoch = -1
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs={}):
        self.predictions = []
        self.steps = []
        self.losses = []
        self.lrn_rates = np.array([])
        self.last_ckpt_loss = np.Inf
        self.stopped_epoch = -1
        self.L2_test_errs = []
        self.minL2_test_errs = 10
        #self.H1_test_errs = []
        self.time_intervals = []

        # variable holding percentage of testing points with error above 10^{-k} for various thresholds k
        self.percs = []
        self.num_perc = np.array([])
        self.Tosave_data={} #TO save the final data without the training and testing points (too expensive to save  run_data)

        # keep track of the minimum loss (corresponding to the last save)
        self.last_output_loss = 10
        self.last_output_epoch = -1

    def on_epoch_end(self, epoch, logs=None):

        current_loss = logs.get("loss")
        current_learning_rate = KerasB.eval(self.model.optimizer.lr.__abs__(epoch))

        self.losses    = np.append(self.losses, [logs["loss"]])
        self.lrn_rates = np.append(self.lrn_rates, [current_learning_rate])

        self.steps.append(epoch)

        if (current_loss < self.best_loss):

            # record the best loss
            self.best_loss       = current_loss
            self.best_loss_epoch = epoch

            # start the epoch timer for waiting if using patience
            self.wait = 0

            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()

            # check if the loss has decreased enough (or more than 10k epochs have gone by) 
            # to evaluate the model on the testing data (just for command line output purposes)
            if (current_loss/self.last_output_loss < self.run_data['update_ratio']) or (epoch - self.last_output_epoch > 1e4):

                # update the last output epoch for checking if 10k have gone by
                self.last_output_epoch = epoch

                # update the last output loss for checking if loss decreased enough
                if (current_loss/self.last_output_loss < self.run_data['update_ratio']):
                    self.last_output_loss = current_loss

                test_start_time = time.time()
                # predict on the x_test_data
                y_DNN_pred = self.model.predict(self.run_data['x_test_data'])

                # the L2 and H1 errors we compute
                L2_err = 0.0
                L2_norm = 0.0

                # compute the absolute difference between the trained model 
                # and the true data
                absdiff = np.abs(self.run_data['y_test_data'] - y_DNN_pred)

                solDNN = Function(self.fenics_params['V'])
                solTRU = Function(self.fenics_params['V'])
                


                for i in range(self.run_data['nb_test_points']):
                    solDNN.vector().set_local(absdiff[i,:])
                    solTRU.vector().set_local(self.run_data['y_test_data'][i,:])
                    if self.run_data['PROBLEM'] =="poisson":

                        u_sol, _  = solTRU.split()
                        uh   , _  = solDNN.split() 
                        norm_u     = sqrt(assemble((u_sol)**2*dx)) 
                        error_L2  = sqrt(assemble((uh)**2*dx)) 

                    if self.run_data['FUNCTION'] =='_u_' and self.run_data['PROBLEM'] =="NSB":
                        u_sol, _, _, _, _   = solTRU.split()
                        uh   , _, _, _, _  = solDNN.split() 
                        norm_u     = sqrt(sqrt(assemble( ((u_sol)**2)**2*dx)))
                        error_L2  = sqrt(sqrt(assemble( ((uh)**2)**2*dx)))
                    if self.run_data['FUNCTION'] =='_p_' and self.run_data['PROBLEM'] =="NSB":
                        u_sol  = solTRU
                        uh     = solDNN 
                        norm_u     = sqrt(assemble((u_sol)**2*dx)) 
                        error_L2   = sqrt(assemble((uh)**2*dx))
                    error_L2 = error_L2
                    norm_u   = norm_u
                    #error_H1 = norm(error, 'H1')

                    L2_err = L2_err   + np.abs(error_L2)**(2.0)*self.run_data['w_quadrature_weights_test'][i]
                    L2_norm = L2_norm + np.abs(norm_u)**(2.0)*self.run_data['w_quadrature_weights_test'][i]
                #=======================================
                # Generate plots in PNG
                #=======================================
                #U_dnn = u_sol-uh
                #plot(U_dnn)
                #filename = 'results/_'+str(epoch)+'_DNNu.png'
                #plt.savefig ( filename )
                #plt.close()   

                L2_err = np.sqrt(np.abs(L2_err*2**(-1.0*self.run_data['input_dim'])))
                L2_norm = np.sqrt(np.abs(L2_norm*2**(-1.0*self.run_data['input_dim'])))
                L2_err =L2_err/L2_norm
                self.L2_test_errs.append(L2_err)
                print(L2_err)
                #self.H1_test_errs.append(H1_err)
                self.time_intervals.append(time.time() - self.run_data['start_time'])
                test_time = time.time() - test_start_time
                #if not self.run_data['quiet']:
                print('======================================================================================================================')
                print('Epochs: ' + str(epoch) + '- Loss = %8.4e,' % (logs["loss"]),
                    'LR = %4.4e,' % (current_learning_rate), 'L2 err = %8.4e,' % (L2_err),
                    #'H1 err = %8.4e,' % (H1_err), 
                    'time to compute test error = %8.2f' % (test_time) )
                print('======================================================================================================================')
                    

        else:
            self.wait += 1

            # With patience large, this will never happen. However, if a small value of patience is used, 
            # then the model weights will be replaced with the best weights seen so far according to the loss
            if self.wait >= self.run_data['patience']:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

        # if the model has converge or run out of epochs of training, or if 1000 epochs have passed
        if (epoch == 0) or (current_loss <= self.run_data['error_tol']) or (epoch == self.run_data['nb_epochs'] - 1) or  self.model.stop_training:
            test_start_time = time.time()
            y_DNN_pred = self.model.predict(self.run_data['x_test_data'])

            # compute the absolute difference between the trained model 
            # and the true data
            absdiff = abs(self.run_data['y_test_data'] - y_DNN_pred)
            # the L2 and H1 errors we compute
            L2_err = 0.0
            L2_norm = 0.0

            # compute the absolute difference between the trained model 
            # and the true data
            absdiff = np.abs(self.run_data['y_test_data'] - y_DNN_pred)
            solDNN = Function(self.fenics_params['V'])
            solTRU = Function(self.fenics_params['V'])

            #error = Function(self.fenics_params['V'])

            for i in range(self.run_data['nb_test_points']):
                solDNN.vector().set_local(absdiff[i,:])
                solTRU.vector().set_local(self.run_data['y_test_data'][i,:])
                if self.run_data['PROBLEM'] =="poisson":

                    u_sol, _   = solTRU.split()
                    uh   , _   = solDNN.split() 
                    norm_u     = sqrt(assemble((u_sol)**2*dx)) 
                    error_L2   = sqrt(assemble((uh)**2*dx)) 

                if self.run_data['FUNCTION'] =='_u_' and self.run_data['PROBLEM'] =="NSB":
                    
                    u_sol, _, _, _, _   = solTRU.split()
                    uh   , _, _, _, _   = solDNN.split() 
                    norm_u     = sqrt(sqrt(sqrt(assemble( ((u_sol)**2)**2*dx))))
                    error_L2  = sqrt(sqrt(assemble( ((uh)**2)**2*dx)))
                if self.run_data['FUNCTION'] =='_p_' and self.run_data['PROBLEM'] =="NSB":
                    u_sol  = solTRU
                    uh     = solDNN 
                    norm_u     = sqrt(assemble((u_sol)**2*dx)) 
                    error_L2  = sqrt(assemble((u_sol-uh)**2*dx))

                

                L2_err  = L2_err  + np.abs(error_L2)**(2.0)*self.run_data['w_quadrature_weights_test'][i]
                L2_norm = L2_norm + np.abs(norm_u)**(2.0)*self.run_data['w_quadrature_weights_test'][i]

            #U_dnn = u_sol-uh
            #plot(U_dnn)
            #filename = 'results/_'+str(epoch)+'_DNNu.png'
            #plt.savefig ( filename )
            #plt.close()    
            L2_err = np.sqrt(np.abs(L2_err*2**(-1.0*self.run_data['input_dim'])))
            L2_norm = np.sqrt(np.abs(L2_norm*2**(-1.0*self.run_data['input_dim'])))
            L2_err =L2_err/L2_norm
            print(L2_err)
            
            self.L2_test_errs.append(L2_err)
            self.minL2_test_errs = min(self.L2_test_errs)
            #self.H1_test_errs.append(H1_err)
            self.time_intervals.append(time.time() - self.run_data['start_time'])
            test_time = time.time() - test_start_time

            
            #self.Tosave_data['lrn_rates_'+ self.run_data['activation'] +'_Npl'+str(self.run_data['nb_layers'])+'x'+str(self.run_data['nb_nodes_per_layer'])+'_m_'+str(self.run_data['nb_train_points'])+'_trial_'+str(self.run_data['trial'])+'_dim_'+str(self.run_data['input_dim'])+'_problem_'+self.run_data['PROBLEM'] +''+self.run_data['FUNCTION']+'']          = self.lrn_rates
            self.Tosave_data['run_time_' + self.run_data['activation'] +'_Npl'+str(self.run_data['nb_layers'])+'x'+str(self.run_data['nb_nodes_per_layer'])+'_m_'+str(self.run_data['nb_train_points'])+'_trial_'+str(self.run_data['trial'])+'_dim_'+str(self.run_data['input_dim'])+'_prob_'+self.run_data['PROBLEM'] +''+self.run_data['FUNCTION']+'']          = time.time() - self.run_data['start_time']
            self.Tosave_data['L2_err_'+ self.run_data['activation'] +'_Npl'+str(self.run_data['nb_layers'])+'x'+str(self.run_data['nb_nodes_per_layer'])+'_m_'+str(self.run_data['nb_train_points'])+'_trial_'+str(self.run_data['trial'])+'_dim_'+str(self.run_data['input_dim'])+'_prob_'+self.run_data['PROBLEM'] +''+self.run_data['FUNCTION']+'']         = self.L2_test_errs
            self.Tosave_data['mL2_err_'+ self.run_data['activation'] +'_Npl'+str(self.run_data['nb_layers'])+'x'+str(self.run_data['nb_nodes_per_layer'])+'_m_'+str(self.run_data['nb_train_points'])+'_trial_'+str(self.run_data['trial'])+'_dim_'+str(self.run_data['input_dim'])+'_prob_'+self.run_data['PROBLEM'] +''+self.run_data['FUNCTION']+'']        = self.minL2_test_errs
            self.Tosave_data['minL2_err']      = self.minL2_test_errs
            #self.run_data['H1_test_errs']       = self.H1_test_errs
            self.Tosave_data['iterations_']         = epoch
            self.Tosave_data['loss_per_iteration_'+ self.run_data['activation'] +'_Npl'+str(self.run_data['nb_layers'])+'x'+str(self.run_data['nb_nodes_per_layer'])+'_m_'+str(self.run_data['nb_train_points'])+'_trial_'+str(self.run_data['trial'])+'_dim_'+str(self.run_data['input_dim'])+'_problem_'+self.run_data['PROBLEM'] +''+self.run_data['FUNCTION']+''] = self.losses
            self.Tosave_data['time_intervals']     = self.time_intervals

            

            # save the resulting mat file with scipy.io
            sio.savemat(self.run_data['run_data_filename'], self.Tosave_data)

            # if we've converged to the error tolerance in the loss, or run
            # into the maximum number of epochs, stop training and save
            if (current_loss <= self.run_data['error_tol']) or (epoch == self.run_data['nb_epochs'] - 1) or self.model.stop_training:
                # output the final checkpoint loss and statistics
                print('=============================================x=x=========================================================================')
                print('Epochs: ' + str(epoch) + '- Loss = %8.4e,' % (logs["loss"]),
                    'LR = %4.4e,' % (current_learning_rate), 'L2 err = %8.4e,' % (L2_err),
                    #'H1 err = %8.4e,' % (H1_err), 
                    'time to compute test error = %8.2f' % (test_time) )
                print('=============================================x=x========================================================================')

                print('Current loss at epoch %s:   %8.12e' % (str(epoch).zfill(8), current_loss))
                print('Best loss at epoch    %s:   %8.12e' % (str(self.best_loss_epoch).zfill(8), self.best_loss))
                if current_loss <= self.best_loss:
                    print("Saving model with current loss.")
                    self.stopped_epoch = epoch
                else:
                    print("Restoring model weights from the end of the best epoch.")
                    self.stopped_epoch = self.best_loss_epoch
                    self.model.set_weights(self.best_weights)

                self.model.save(self.run_data['DNN_model_final_savedir'])

                self.Tosave_data['run_time'] = time.time() - self.run_data['start_time']
                self.Tosave_data['percentiles_at_save'] = self.percs
                self.Tosave_data['percentiles_save_iters'] = self.num_perc
                #self.Tosave_data['y_DNN_pred'] = y_DNN_pred
                self.Tosave_data['iterations'] = self.steps
                self.Tosave_data['loss_per_iteration'] = self.losses
                self.Tosave_data['lrn_rates'] = self.lrn_rates
                self.Tosave_data['stopped_epoch'] = self.stopped_epoch
                self.Tosave_data['best_loss'] = self.best_loss
                self.Tosave_data['iterations_']         = epoch
                self.Tosave_data['best_loss_epoch'] = self.best_loss_epoch

                # save the resulting mat file with scipy.io
                sio.savemat(self.run_data['run_data_filename'], self.Tosave_data)

                self.model.stop_training = True

        if (epoch % self.run_data['DNN_show_epoch'] == 0):
            print('epoch: ' + str(epoch).zfill(8) + ', loss: %8.4e, lrn_rate: %4.4e, seconds: %8.2f ' \
                % (logs["loss"], current_learning_rate, time.time() - self.run_data['start_time']))
            self.Tosave_data['iterations_']         = epoch


    def on_train_end(self, logs=None):
        if (self.stopped_epoch > 0) and (self.stopped_epoch < self.run_data['nb_epochs'] - 1):
            print("Epoch %05d: early stopping" % (self.stopped_epoch))

