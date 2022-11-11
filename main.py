

import numpy as np
from DataProcessing import *
from dictionary import *



def spade4(X,dXdt,train_ratio_list,dimension,timesteps,dt = None,embedding_dim = None,tau = None, n = None, 
           tol = None, lambda_vec = None,val_window = None):
    """
    '''Description: Build the time delayed matrix for training input data and output vector 
    Input: input_train of size (num_train,); output_train of size (num_train,)
    Output: H_train matrix of size (num_train - tau*emb_dim + 1,emb_dim);
            y_train of size(num_train - tau*emb_dim + 1,)

   
    Input
    -----------
    X:    numpy array of size (#measurements,)
          Observed variable of the multidimensional system of ODEs
    dXdt: numpy array of size (#measurements,)
          Approximated values of dX/dt and then denoised
    
    dt = 1
    train_ratio_list: list of possible training-validation ratios
    dimension: (int) choice of possible true dimension of the dynamical system
    embedding_dim: (int, default: 2*dimension+1) choice of embedding dimension.  
    tau: (int, default:1) choice of time delay
    n: (int, default: 50) scale of the number of measurements which gives number of features i.e., #features = n*measurements
    tol: (float, default: 1e-5) tolerance for converence of optimization algorithm
    lambda_vec: (default: [5e-6, 1e-7, 5e-7, 1e-8, 5e-8, 1e-9]) list of possible regularization parameter. 
    timesteps: total number of days
    val_window: (default: 7)short-term forecasting window
    
    Output
    -----------
    out_mat: numpy array of size (val_window,len(train_ratio_list))
             Matrix of predicted values of X for val_window days for each training ratio."""
    if dt is None:
        dt = 1
    if embedding_dim is None:
        embedding_dim = 2*dimension+1
        
    if tau is None:
        tau = 1
    if n is None:
        n = 50
    if tol is None:
        tol = 1e-5
    if lambda_vec is None:
        lambda_vec = [5e-6, 1e-7, 5e-7, 1e-8, 5e-8, 1e-9]
    if val_window is None:
        val_window = 7
        
    out_mat = np.zeros([val_window ,len(train_ratio_list)])
    for k in np.arange(0,len(train_ratio_list)):
        train_ratio = train_ratio_list[k]
        val_ratio = 1-train_ratio
        num_train = int(train_ratio*(timesteps))
        num_val = int(val_ratio*(timesteps))
        input_train,input_val,output_train,_ = get_data(X,dXdt,train_ratio)
        
        ## Build the time delayed matrix
        #training time delayed Matrix
        H_train = np.zeros([int(train_ratio*timesteps)-tau*embedding_dim+1,embedding_dim])
        y_train = np.zeros(int(train_ratio*timesteps)-tau*embedding_dim+1)
        for i in range(num_train-tau*embedding_dim+1):
            H_train[i,:] = input_train[i:(i+tau*embedding_dim):tau]
            y_train[i] = output_train[(i+tau*embedding_dim-1)] # dx(t)/dt = x(t) + previous tim


        '''Description: Build dictionary matrix from H_train and y_train
        Input: time delayed matrix H_train of size (num_train - tau*emb_dim + 1,emb_dim) 
        Output: Dictionary matrix of size (num_train - tau*emb_dim + 1,k) where k depends on the type of dictionary
        Last updated: March 10, 2022; 10:30 am'''
        ## Build the dictionary matrix
        N = n*num_train
        Omega_SRF,bias_SRF = generate_omega_bias(rows = N,columns = embedding_dim,weight = 1,par1 = 0,par2 = 1,
                                           distribution = 'SRF',bool_bias = True,sparsity = embedding_dim)
        Dict_train = dictionary(H_train, Omega = Omega_SRF , bias = bias_SRF, activation = 'relu', dictType = 'SRF')
        
        #normalize each column of dictionary matrix
        scale = np.linalg.norm(Dict_train, axis = 0) 
        Dict_train /= scale

        '''Description: Solve the problem using optimization algorithm
        Input: Dictionary matrix and output vector y_train
        Output: Coefficient matrix of size (k,)'''
        for r in np.arange(0,len(lambda_vec)):
            alpha_lasso = lambda_vec[r]
            clfMu = linear_model.Lasso(alpha=alpha_lasso, fit_intercept=False, 
                                 normalize=False, precompute=False, max_iter=50000,
                                 tol=tol, selection='random')
            clfMu.fit(Dict_train, y_train.reshape(-1))
            cMu_est = clfMu.coef_

            y_train_rec = np.matmul(Dict_train,cMu_est.reshape(-1,1))
            loss = np.linalg.norm(y_train_rec.reshape(-1)-y_train)**2/num_train
            BIC_model = BIC(loss, np.count_nonzero(cMu_est),num_train)
            if 1e+10 > BIC_model:
                alpha_criteria = BIC_model
                alpha_final = alpha_lasso
                loss_final = loss
                cMu = cMu_est
        
        '''Description: (1) Recovering the output (training) vector from the learnt coefficients.
                        (2) Calculating the training error and plotting them. '''
        
        H_train_rec = np.matmul(Dict_train,cMu.reshape(-1,1))        
        '''Description: (1) Predict validation data points from learnt coefficients using euler method for solving ODE.
                        (2) Calculate validation error on (input) data and plot them. 
           Input: Dictionary matrix at previous time point and learnt coefficients.
           Output: I_val at each (future) time point.'''
        
        H_timeDelay = H_train
        I_learnt_val = np.zeros(timesteps-num_train)
        I_learnt_val[0] = input_train[num_train-1]
        xvec = np.zeros([embedding_dim,1])
        for i in range(embedding_dim): 
            xvec[i] = input_train[num_train-embedding_dim+i]


        xnew = xvec[-1] + dt*np.matmul(dictionary(xvec.T,Omega_SRF,bias_SRF,activation = 'relu' ,dictType = 'SRF')
                                   /scale, cMu.reshape(-1,1))

        I_learnt_val[0] = xnew

        for i in range(1,num_val):
            for j in range(embedding_dim-1): 
                xvec[j] = xvec[j+1]
            xvec[embedding_dim-1] = xnew

            xnew = xvec[-1] + dt*np.matmul(dictionary(xvec.T,Omega_SRF,bias_SRF,activation = 'relu' ,dictType = 'SRF')
                                       /scale, cMu.reshape(-1,1))
            I_learnt_val[i,] = xnew
        
        out_mat[:,k] =  I_learnt_val[0:val_window]
        
    return out_mat

