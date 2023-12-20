import threading
import time

def function1():
    for epoch in range(10):
        print("Function 1:",epoch)
        time.sleep(3)  # Delay for 1 second
    print('over 1')
    return epoch

def function2():
    for epoch in range(11):
        print("                              Function 2",epoch)
        time.sleep(6)  # Delay for 1 second
    print('over 2')
    return epoch

# Create threads for each function
thread1 = threading.Thread(target=function1)
thread2 = threading.Thread(target=function2)


# Start the threads
_epochs1 = thread1.start()
_epochs2 = thread2.start()


# Keep the main thread running
print('over total',_epochs2)




def function_DNN_u(DNN_u,nb_epochs,BATCH_SIZE,m,x_train_data,u_train_data):
    DNN_u.summary()
    l2_error_data   = np.array([])
    for epoch in range(nb_epochs):
        #For each epoch we select random entries of training data
        I_ad       = np.array([])
        I          = np.array([])
        batch_size = int(BATCH_SIZE)
        I_i        = range(m)
        I_ad       = random.sample(I_i, batch_size)
        I          = np.append(I, I_ad).astype(int)
        # Data inside the domain
        y_in     = tf.cast(np.asarray(x_train_data)[I,:], tf.float32)
        u_in     = tf.cast(np.asarray(u_train_data)[I,:], tf.float32)
        # gradient descent
        GD_u(y_in,u_in)
        # compute loss
        res = get_mse(y_in,u_in)
        l2_error_data = np.append(l2_error_data, res)
        if (epoch % args.DNN_show_epoch ==0):
            print('============================================================')
            print('Epochs: ' + str(epoch) + ' | Error: ' + str("{:.4e}".format(res)) )
            print('============================================================')




def function_DNN_p(DNN_p,nb_epochs,BATCH_SIZE,m,x_train_data,p_train_data):
    DNN_p.summary()
    l2_error_data   = np.array([])
    for epoch in range(nb_epochs):
        #For each epoch we select random entries of training data
        I_ad       = np.array([])
        I          = np.array([])
        batch_size = int(BATCH_SIZE)
        I_i        = range(m)
        I_ad       = random.sample(I_i, batch_size)
        I          = np.append(I, I_ad).astype(int)
        # Data inside the domain
        y_in     = tf.cast(np.asarray(x_train_data)[I,:], tf.float32)
        p_in     = tf.cast(np.asarray(p_train_data)[I,:], tf.float32)
        # gradient descent
        GD_p(y_in,p_in)
        # compute loss
        res = get_mse(y_in,u_in)
        l2_error_data = np.append(l2_error_data, res)
        if (epoch % args.DNN_show_epoch ==0):
            print('                                                             ============================================================')
            print('                                                             Epochs: ' + str(epoch) + ' | Error: ' + str("{:.4e}".format(res)) )
            print('                                                             ============================================================')






for epoch in range(nb_epochs):
    #For each epoch we select random entries of training data
    I_ad       = np.array([])
    I          = np.array([])
    batch_size = int(BATCH_SIZE)
    I_i        = range(m)
    I_ad       = random.sample(I_i, batch_size)
    I          = np.append(I, I_ad).astype(int)
    # Data inside the domain
    y_in     = tf.cast(np.asarray(x_train_data)[I,:], tf.float32)
    u_in     = tf.cast(np.asarray(u_train_data)[I,:], tf.float32)