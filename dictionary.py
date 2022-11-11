import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib



#             phiX[:, int((k*(2*n-k+3)/2))]=(np.sqrt(5)/2.0)*(3*np.array([np.square(U[:,k-1])]).T-np.ones([m,1])).T
          
#     return phiX

def dictionarypoly(U,order,option=[]):
    #% Description: Construct the dictionary matrix phiX containing all multivariate monomials up to degree two for the Lorenz 96
    # % Input: U = [x1(t1) x2(t1) .... xn(t1)
    # %             x1(t2) x2(t2) .... xn(t2)
    # %                    ......
    # %             x1(tm) x2(tm) .... xn(tm)]
    # %        option = [] (monomial) or 'legendre'
    # % Output: the dictionary matrix phiX of size m by N, where m= #measurements and N = (n^2+3n+2)/2
    #if nargin ==1
    #if str(option) == 'mon' or str(option=='legendre):
    #end
    #U=np.array([[1,2,3],[4,5,6],[7,8,9]])
    if order == '2':
        m=int(np.size(U,0))
        n=int(np.size(U,1))
        phiX=np.zeros([m,int((n+1)*(n+2)/2)])
        phiX[:,0]=np.ones([1,m])
        phiX[:,1:n+1]=np.sqrt(3)*U
        for k in range(1,n+1):
            phiX[:,int(((k)*(2*(n)-k+3)/2)) : int(((k+1)*(n) -(k**2)/2 + k/2 +1 ))]=3*np.multiply(np.matlib.repmat(np.array([U[:,k
                                                                                                                    -1]]).T,1,n+1-k),U[:,k-1:n])
            if option=='legendre':
                phiX[:, int((k*(2*n-k+3)/2))]=(np.sqrt(5)/2.0)*(3*np.array([np.square(U[:,k-1])]).T-np.ones([m,1])).T
          
    else:
        m=int(np.size(U,0))
        n=int(np.size(U,1))
        phiX=np.zeros([m,int((3*n**2 + 3*n + 2)/2)])
        phiX[:,0]=np.ones([1,m])
        phiX[:,1:n+1]=np.sqrt(3)*U
        for k in range(1,n+1):
            phiX[:,int(((k)*(2*(n)-k+3)/2)) : int(((k+1)*(n) -(k**2)/2 + k/2 +1 ))]=3*np.multiply(np.matlib.repmat(np.array([U[:,k
                                                                                                                               -1]]).T,1,n+1-k),U[:,k-1:n])
            phiX[:,int(((n+1)*(n) -(n**2)/2 + n/2 +1 )+(k-1)*n):int(((n+1)*(n) -(n**2)/2 + n/2 +1 )+(k)*n)]=3*np.sqrt(3)*np.multiply(np.matlib.repmat(np.array([U[:,k-1]**2]).T,1,n),U)
            if option=='legendre3':
                phiX[:, int((k*(2*n-k+3)/2))]=(np.sqrt(5)/2.0)*(3*np.array([np.square(U[:,k-1])]).T-np.ones([m,1])).T
                phiX[:, int(((n+1)*(n) -(n**2)/2 + n/2 +1 )+(k-1)*(n+1))]=(np.sqrt(7)/2.0)*(5*np.array([(U[:,k-1])**3]).T
                                                                                        - 3*np.array([(U[:,k-1])]).T).T
    
    return phiX

def generate_omega_bias(rows,columns,weight,par1,par2,distribution,bool_bias,sparsity):
    mask = np.zeros(shape=(rows,columns)) #(N,d)
    for i in range(rows):
        idx = np.random.choice(columns,sparsity,replace=False)
        mask[i,idx] = 1.
    if distribution=='uniform':
        Omega = weight*np.random.uniform(par1,par2,(rows,columns))
        Omega *= mask
        if bool_bias == True:
            bias = np.random.uniform(par1,par2,rows)
        else: 
            bias = np.zeros(rows)
    elif distribution=='SRF':
        Omega = weight*np.random.normal(par1,par2,(rows,columns))
        Omega *= mask
        if bool_bias == True:
            bias = np.random.uniform(0,2*np.pi,rows)
        else:
            bias = np.zeros(rows)
    else:
        Omega = weight*np.random.normal(par1,par2,(rows,columns))
        Omega *= mask
        if bool_bias == True:
            bias = np.random.normal(par1,par2,rows)
        else:
            bias = np.zeros(rows)
    return Omega,bias


'''Define dictionary'''
def dictionary(X,Omega ,bias ,activation ,dictType):
    ''' Input: X of size(#measurements,dimension)
    - Omega: matrix of size (rows,columns) i.e., (#features,dimension)
    - bias: vector of size (#features,)
    - activation: 'sin' for sin; 'tanh' for tanh; else for 'relu'.
    -dictType: Type of dictionary - 'SRF' for SRF dictionary; 'poly' for polynomial dictiionary upto order 2;
    'SRFpoly' for SRF+polynomial dictionary
    -idty: add constant times identity; 'yes' or 'no'.'''
    
    if dictType == 'SRF':
        if activation=='sin':
            A_ext = np.sin(np.matmul(X,Omega.T) + bias)
            A = A_ext #[:,A_ext.any(0)][:,0:int(Omega.shape[0]/2)]
        elif activation=='tanh':
            A_ext = np.tanh(np.matmul(X,Omega.T) + bias)
            A = A_ext #[:,A_ext.any(0)][:,0:int(Omega.shape[0]/2)]
            
        else:
            A_ext = np.maximum(np.matmul(X,Omega.T) + bias,0)
            A = A_ext #[:,A_ext.any(0)][:,0:int(Omega.shape[0]/2)]
                            
#     elif dictType == 'Legpoly2':
#         A = dictionarypoly(X,'2','legendre')
#         Adict = A
#     elif dictType == 'Legpoly3':
#         A = dictionarypoly(X,'3','legendre')
#         Adict = A
#     elif dictType == 'SRFpoly2':
#         polydict = dictionarypoly(X,'2','legendre')
#         if activation=='sin':
#             Asrf = np.sin(np.matmul(X,Omega.T) + bias)
#         elif activation=='tanh':
#             Asrf = np.tanh(np.matmul(X,Omega.T) + bias)
#         else:
#             Asrf = np.maximum(np.matmul(X,Omega.T) + bias,0)
#         A = np.hstack((polydict,Asrf))
        
#     else:
#         polydict = dictionarypoly(X,'3','legendre')
#         if activation=='sin':
#             Asrf = np.sin(np.matmul(X,Omega.T) + bias)
#         elif activation=='tanh':
#             Asrf = np.tanh(np.matmul(X,Omega.T) + bias)
#         else:
#             Asrf = np.maximum(np.matmul(X,Omega.T) + bias,0)
#         A = np.hstack((polydict,Asrf))
#     return A
    return A #, Omega[A_ext.any(0),:][0:int(Omega.shape[0]/2),:], bias[A_ext.any(0)][0:int(Omega.shape[0]/2)]


def hard_thresh_pur_tik(A,b,c0,s,tot_iter,thresh,mu,lam,task):
    error = np.zeros(tot_iter)
    b = b.flatten()
    C = np.zeros((A.shape[1],tot_iter))
    C[:,0] = c0
    i = 0
    iter_req = i
    z1 = np.matmul(A.T,b)
    z2 = np.matmul(A.T,A)
    rel_err = np.linalg.norm(np.matmul(A,c0).flatten()-b)/np.linalg.norm(b)
    while rel_err>thresh:
#         c_tilde = C[:,i] + mu*(z1 - np.matmul(z2,C[:,i]))
        c_tilde = C[:,i] + mu*(z1 - np.matmul(z2,C[:,i])) - ((mu*lam)*C[:,i])
        c_tilde_sort = np.sort(abs(c_tilde))
        idx = c_tilde >= c_tilde_sort[-s]
        A_pruned = A[:,idx]
        z1_pruned = np.matmul(A_pruned.T,b)
        z2_pruned = np.matmul(A_pruned.T,A_pruned)
#         c_pruned = np.matmul(np.linalg.pinv(z2_pruned),z1_pruned)
        c_pruned = np.matmul(np.linalg.pinv((z2_pruned + lam*np.identity(z2_pruned.shape[1])),rcond=1e-15)
                                 ,z1_pruned)
#         c_pruned = np.matmul(np.linalg.pinv((z2[:,idx][idx,:] + lam*np.identity(z2[:,idx][idx,:].shape[1])),rcond=1e-15)
#                                  ,z1[idx])
    
        c_pruned = c_pruned.flatten()
        erlst = 'nd'
        C[idx,i+1] = c_pruned
        if task=='classify':
            error[i+1] = np.linalg.norm((1/(1 + np.exp(-np.matmul(A,C[:,i+1]).flatten())))-b)/np.linalg.norm(b)
        else:
            error[i+1] = np.linalg.norm(np.matmul(A,C[:,i+1]).flatten()-b)/np.linalg.norm(b)
        rel_err = error[i+1]
        i = i+1
        iter_req = i
        if i+1==tot_iter:
#             print('Warning: You might want to use more iterations as maximum number of iterations of',
#                   tot_iter,'reached with relative error of',rel_err)
            iter_req = i
            break
                
                
#     print('Finally the total number of iterations the algorithm ran for was',iter_req+1)
    return C[:,0:iter_req+1],error[0:iter_req+1],iter_req+1,erlst



def normalize(X):
    '''X is measurement matrix of size (#measurements,dimension)'''
    x_norm = np.zeros([X.shape[0],X.shape[1]])
    for i in range(X.shape[1]):
        mean = np.mean(X[:,i])
        std = np.std(X[:,i])
        x_norm[:,i] = (X[:,i]-mean)/std
    return x_norm


def normalize_data_minmax(X):
    '''X is measurement matrix of size (#measurements,dimension)'''
    x_norm = np.zeros([X.shape[0],X.shape[1]])
    for i in range(X.shape[1]):
        min_x = np.min(X[:,i])
        max_x = np.max(X[:,i])
        x_norm[:,i] = (X[:,i]-min_x)/(max_x-min_x)
    return x_norm




def hard_thresh_pur_tik(A,b,c0,s,tot_iter,thresh,mu,lam,task):
    error = np.zeros(tot_iter)
    b = b.flatten()
    C = np.zeros((A.shape[1],tot_iter))
    C[:,0] = c0
    i = 0
    iter_req = i
    z1 = np.matmul(A.T,b)
    z2 = np.matmul(A.T,A)
    rel_err = np.linalg.norm(np.matmul(A,c0).flatten()-b)/np.linalg.norm(b)
    while rel_err>thresh:
#         c_tilde = C[:,i] + mu*(z1 - np.matmul(z2,C[:,i]))
        c_tilde = C[:,i] + mu*(z1 - np.matmul(z2,C[:,i])) - ((mu*lam)*C[:,i])
        c_tilde_sort = np.sort(abs(c_tilde))
        idx = c_tilde >= c_tilde_sort[-s]
        A_pruned = A[:,idx]
        z1_pruned = np.matmul(A_pruned.T,b)
        z2_pruned = np.matmul(A_pruned.T,A_pruned)
#         c_pruned = np.matmul(np.linalg.pinv(z2_pruned),z1_pruned)
        c_pruned = np.matmul(np.linalg.pinv((z2_pruned + lam*np.identity(z2_pruned.shape[1])),rcond=1e-15)
                                 ,z1_pruned)
#         c_pruned = np.matmul(np.linalg.pinv((z2[:,idx][idx,:] + lam*np.identity(z2[:,idx][idx,:].shape[1])),rcond=1e-15)
#                                  ,z1[idx])
    
        c_pruned = c_pruned.flatten()
        erlst = 'nd'
        C[idx,i+1] = c_pruned
        if task=='classify':
            error[i+1] = np.linalg.norm((1/(1 + np.exp(-np.matmul(A,C[:,i+1]).flatten())))-b)/np.linalg.norm(b)
        else:
            error[i+1] = np.linalg.norm(np.matmul(A,C[:,i+1]).flatten()-b)/np.linalg.norm(b)
#         rel_err = error[i+1]
#         print(rel_err,C.shape)
        i = i+1
        iter_req = i
        if i+1==tot_iter:
#             print('Warning: You might want to use more iterations as maximum number of iterations of',
#                   tot_iter,'reached with relative error of',rel_err)
            iter_req = i
            break
                
                
#     print('Finally the total number of iterations the algorithm ran for was',iter_req+1)
    return C[:,0:iter_req+1],error[0:iter_req+1],iter_req+1,erlst

