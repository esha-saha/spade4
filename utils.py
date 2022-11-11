import numpy as np
import pandas as pd
import condat_tv
from scipy.optimize import curve_fit
import csv
from dictionary import *
from compartment_models import *
import sys
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import pandas as pd

from DataProcessing import *
from dictionary import *

from spgl1 import spg_bpdn
import sklearn
from sklearn import linear_model
import csv

'Create output dataset'

def create_out_data(X,noise = None,avg = None,smooth_factor = None,der_type = None,dt = None):
    """Inputs
    -------------
    
    X: (type: array,) input data (active or cumulative) of size (days,dimension)
    noise: (type: bool, default: True) noise present in data or not
    avg: (type: int, default: 7) number of days for averaging if noise = True
    smooth_factor: (type: int, default: 15) factor used for convolution based smoothing of the derivative
    der_type: (type: int, default: 2) type of finite difference for getting derivative; 1 for forward difference, 2 for central difference
    dt: (type: float, default: 1) days corresponding to which data is available
    """
    if noise is None:
        noise = True
    if avg is None:
        avg = 7
    if smooth_factor is None:
        smooth_factor = 15
    if der_type is None:
        der_type = 2
    if dt is None:
        dt = 1
        
    X_avg =rolling_avg(X,avg)
    dXdt = time_derivative(X_avg.reshape(-1,1),dt,der_type)
    dXdt_smooth = smooth(dXdt.squeeze(), smooth_factor, noise)
    return dXdt_smooth
    
    
    
def get_Idata(population):
    """get the data from csv file"""
    data = pd.read_csv("canada-covid19.csv")
    I_data = np.array(data.iloc[0:,1])
    return I_data, I_data_norm


def rolling_avg(data,window):
    if window==1:
        data_avg = data
    else:
        data_avg = np.zeros(int(data.shape[0]))#-window)
        data_avg[0:window-1] = data[0:window-1]
        for i in range(int(data.shape[0])-window+1):
            data_avg[i+window-1] = np.sum(data[i:i+window])/window
    return data_avg


def time_derivative(X,dt,type):
    ''' X is of size (#measurements,dimension)'''
    m=int(np.size(X,0))
    n=int(np.size(X,1))
    if type==1:
        roc=np.zeros([m,n])
        roc[0:m-1,:]=(X[1:m,:]-X[0:m-1,:])/dt #forward euler
        roc[m-1,:]=(X[m-1,:]-X[m-2,:])/dt #backward
    elif type==2:
        roc=np.zeros([m,n])
        roc[0,:]=(X[1,:]-X[0,:])/dt #forward
        roc[1:m-1,:]=(X[2:m,0:n]-X[0:m-2,0:n])/(2*dt) 
        roc[m-1,:] = (X[m-1,:] - X[m-2,:])/dt; # backward
    return roc


'''smoothing function''' 
def smooth(y, smooth_par,noise):
    "denoise the derivative vector"
    if noise== False:
        s=1
    else:
        s = smooth_par
        box = np.ones(s)/s
        y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



'''get training and val data'''

def get_data(X,Y,train_ratio):
    """split data into training and validation set"""
    tot_points = X.shape[0]
    inp_train = X[0:int(train_ratio*tot_points)]
    inp_val = X[int(train_ratio*tot_points):tot_points]
    out_train = Y[0:int(train_ratio*tot_points)]
    out_val = Y[int(train_ratio*tot_points):tot_points]
    return inp_train,inp_val,out_train,out_val


def BIC(SS,par,N_train): # it is BIC
    return (N_train*np.log(SS) + np.log(N_train)*par)



        

def error_fun(fit,par,y0,tspan,num_train,timesteps,time_st,dt_fine):
    if fit == 'seir':
        beta_learnt,sigma_learnt,gamma_learnt = par
        fitted = odeint(seir_function, y0, tspan, args=(beta_learnt,sigma_learnt,gamma_learnt),tfirst = True)
        fitted1 = fitted[0::time_st,2]
        fitted_train = fitted1[0:num_train] #fitted[0:int(num_train/dt_fine):time_st,2]
        fitted_val = fitted1[num_train:] #fitted[int(num_train/dt_fine):int(timesteps/dt_fine):time_st,2]
    elif fit =='sueir':
        beta_learnt,sigma_learnt,gamma_learnt,mu_learnt = par
        fitted = odeint(seirMu_function, y0, tspan, args=(beta_learnt,sigma_learnt,gamma_learnt,mu_learnt),tfirst = True)
        fitted_train = fitted[0:int(num_train/dt_fine):time_st,2]
        fitted_val = fitted[int(num_train/dt_fine):int(timesteps/dt_fine):time_st,2]
        
    else:
        beta_learnt,gamma_learnt = par
        fitted = odeint(sir_function, y0, tspan, args=(beta_learnt,gamma_learnt),tfirst = True)
        fitted_train = fitted[0:int(num_train/dt_fine):time_st,1]
        fitted_val = fitted[int(num_train/dt_fine):int(timesteps/dt_fine):time_st,1]
    return fitted_train, fitted_val



def get_data_coi(X,Y,train_points):
    tot_points = X.shape[0]
    inp_train = X[0:train_points]
    inp_val = X[train_points:tot_points]
    out_train = Y[0:train_points]
    out_val = Y[train_points:tot_points]
    return inp_train,inp_val,out_train,out_val


def coi(a,b,d,Tfinal,val_window,I,dIdt,tstep):
    timesteps = int(Tfinal)
    train_points_vec = np.linspace(a,b,(b-a)+1)
    out_mat = np.zeros([val_window ,len(train_points_vec)])
    error_mat = np.zeros([val_window ,len(train_points_vec)])
    error_val = np.zeros(len(train_points_vec))
    error_train = np.zeros(len(train_points_vec))
    
    
    for k in np.arange(0,len(train_points_vec)):
#     print(train_points_vec[k],k,len(train_points_vec))
        train_ratio = train_points_vec[k]/Tfinal
        val_ratio = 1-train_ratio
        num_train = int(train_points_vec[k])
        num_val = int(Tfinal-train_points_vec[k])
    #     print(train_ratio,num_train)
        input_train,input_val,output_train,_ = get_data_coi(I,dIdt,num_train) 
    #     print('shape of input',input_train.shape)
        #first num_train points from avg I analysis window

        ## Build the time delayed matrix
        emb_dim = 2*int(d)+1  #embedding dimension
        tau = 1 #time delay
        # observed_var = 2 #observed variable

        #training time delayed Matrix
        '''delayed matrix is of size (#embedding_dim,train_points-(time_delay*embedding_dim)+1)'''
        H_train = np.zeros([emb_dim,int(train_ratio*timesteps)-tau*emb_dim+1])
        y_train = np.zeros(int(train_ratio*timesteps)-tau*emb_dim+1,)
        for i in range(int(train_ratio*timesteps)-tau*emb_dim+1):#0 to 145, not counting last one
            H_train[:,i] = input_train[i:(i+tau*emb_dim):tau]
            y_train[i] = output_train[(i+tau*emb_dim-1)] # dx(t)/dt = x(t) + previous time


        ## Build the dictionary matrix
        dictType = 'SRF'

        if dictType == 'SRF':
            N = 5000
            q = 2
            act_SRF = 'relu'
            Omega_SRF1,bias_SRF1 = generate_omega_bias(rows = 2*N,columns = emb_dim,weight = 1,par1 = 0,par2 = 1,
                                               distribution = 'SRF',bool_bias = True,sparsity = q)
            Dict_train1 = dictionary(H_train.T, Omega = Omega_SRF1 , bias = bias_SRF1, activation = act_SRF, dictType = dictType)
            Dict_train = Dict_train1[:,Dict_train1.any(0)][:,0:N]
            Omega_SRF = Omega_SRF1[Dict_train1.any(0),:][0:N,:]
            bias_SRF = bias_SRF1[Dict_train1.any(0)][0:N]

        else:
            Omega_SRF = None
            bias_SRF = None
            act_SRF = None
            Dict_train = dictionary(H_train.T, Omega = Omega_SRF , bias = bias_SRF, activation = act_SRF, dictType = dictType)



        # print(np.count_nonzero(Dict_train))

    #     print('train shape:',train_ratio,'shape of dictionary matrix is',Dict_train.shape)

        #normalize each column of dictionary matrix
        scale = np.linalg.norm(Dict_train, axis = 0) 
    #     print('shape of scale vector (used for SRF)',scale.shape,np.count_nonzero(scale))
        Dict_train /= scale

        # '''Run test using LASSO'''
        tol_lasso = 1e-4 #*np.linalg.norm(H_train_dxdt[EqnRec,:])
        alpha_lasso_vec = [1e-7] #, 5e-7, 1e-8, 5e-8] #[1e-5, 5e-5, 1e-6, 5e-6, 1e-7, 5e-7, 1e-8, 5e-8] #1e-9, 1e-10]
        alpha_criteria = 1e+10
        for r in np.arange(0,len(alpha_lasso_vec)):
            alpha_lasso = alpha_lasso_vec[r]
            clfMu = linear_model.Lasso(alpha=alpha_lasso, fit_intercept=False, 
                                     normalize=False, precompute=False, max_iter=50000,
                                     tol=tol_lasso,selection='random')
            clfMu.fit(Dict_train, y_train.reshape(-1))
            cMu_est = clfMu.coef_

            y_train_rec = np.matmul(Dict_train,cMu_est.reshape(-1,1))
            loss = np.linalg.norm(y_train_rec.reshape(-1)-y_train)**2/num_train
            BIC_model = BIC(loss, np.count_nonzero(cMu_est),num_train)
    #         print(BIC_model)

            if alpha_criteria > BIC_model:
                    alpha_criteria = BIC_model
                    alpha_final = alpha_lasso
                    loss_final = loss
                    cMu = cMu_est
    #     print(alpha_final)


        '''Description: Pruning the coefficient vector

        Last updated: March 10, 2022; 10:30 am'''


        H_train_rec = np.matmul(Dict_train,cMu.reshape(-1,1))
    #     train_mat[:,k] = H_train_rec.reshape(-1)
        # print(H_train_rec.reshape(-1).shape,y_train.shape)

        error_train[k] = np.linalg.norm(H_train_rec.reshape(-1)-y_train)/np.linalg.norm(y_train)
    #     print('train error:',np.linalg.norm(H_train_rec.reshape(-1)-y_train)/np.linalg.norm(y_train))

        H_timeDelay = H_train

        I_learnt_val = np.zeros(timesteps-num_train)
        I_learnt_val[0] = input_train[num_train-1]

        xvec = np.zeros([emb_dim,1])
        for i in range(emb_dim): 
            xvec[i] = input_train[num_train-emb_dim+i]


        xnew = xvec[-1] + (tstep[num_train]-tstep[num_train-1])*np.matmul(dictionary(xvec.T,Omega_SRF,bias_SRF,activation = act_SRF ,dictType
                                                                                     = dictType)/scale, cMu.reshape(-1,1))

        I_learnt_val[0] = xnew

        for i in range(1,num_val):
            for j in range(emb_dim-1 ): 
                xvec[j] = xvec[j+1]
            xvec[emb_dim-1] = xnew

            xnew = xvec[-1] + (tstep[num_train+i]-tstep[num_train+i-1])*np.matmul(dictionary(xvec.T,Omega_SRF,bias_SRF,activation = act_SRF
                                                                                             ,dictType = dictType)/scale, cMu.reshape(-1,1))
            I_learnt_val[i,] = xnew

        out_mat[:,k] =  I_learnt_val[0:val_window]
        error_mat[:,k] = I_learnt_val[0:val_window]-input_val[0:val_window]

        if k%5==0:
            error_val[k] = np.linalg.norm(I_learnt_val[0:val_window]-input_val[0:val_window])/np.linalg.norm(input_val[0:val_window])
            print('val error at',k,'is',np.linalg.norm(I_learnt_val[0:val_window]-input_val[0:val_window])
                  /np.linalg.norm(input_val[0:val_window]))
    
    sigma = np.sqrt((np.linalg.norm(error_mat,axis = 1)**2)/len(train_points_vec))
    print(sigma)
    coi_plus = out_mat[:,-1] + 1.96*sigma
    coi_min = out_mat[:,-1] - 1.96*sigma

    fig = plt.figure(figsize=(4,4))
    # plt.plot(I_norm[-20:-1]*pre_const,'k', label = 'True I values')
    # plt.legend()
    # plt.title('(normalized) I values for the wave being studied')
    plt.plot(I_learnt_val[:val_window],'b',label = 'Ours',linewidth = 2.5)
    plt.plot(input_val[:val_window],'k',label = 'True',linewidth = 2.5)
    plt.plot(coi_plus[:],'.')
    plt.plot(coi_min[:],'.')
    plt.xticks(np.arange(val_window),np.arange(1,val_window+1))
    # plt.title('Plot of I on val data')
    plt.tick_params(labelsize = 18)
    plt.xlabel('Days',fontsize=18)
    plt.ylabel('Cummulative Cases',fontsize=18)
    x = np.linspace(0,val_window-1,val_window)
    plt.fill_between(x,coi_min[0:val_window], coi_plus[0:val_window],
                     facecolor='orange', # The fill color
                     color='red',       # The outline color
                     alpha=0.2)          # Transparency of the fill

    plt.legend(loc = 'upper left')
    plt.show()