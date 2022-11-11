import numpy as np

def seir_function(t, y,beta,sigma,gamma): 
    ''' y = [S_hat,E_hat,I_hat,R_hat] such that S_hat + E_hat + I_hat + R_hat = 1
    dS_hat / dt = -beta * S_hat * I_hat 
    dE_hat / dt = +beta * S_hat * I_hat - sigma * E_hat
    dI_hat / dt = +sigma * E_hat - gamma * I_hat 
    dR_hat / dt = gamma * I_hat '''

    ydot = np.zeros(4)
    ydot[0] = -beta * y[0] * y[2] 
    ydot[1] = +beta * y[0] * y[2]  - sigma * y[1]
    ydot[2] = +sigma * y[1] - gamma * y[2] 
    ydot[3] = gamma * y[2] 
    return ydot


def sir_function(t, y,beta,gamma): 
    ''' y = [S_hat,I_hat,R_hat] such that S_hat  + I_hat + R_hat = 1
    dS_hat / dt = -beta * S_hat * I_hat 
    dI_hat / dt = +beta * S_hat * I_hat - gamma * I_hat
    dR_hat / dt = gamma * I_hat '''


    ydot = np.zeros(3)
    ydot[0] = -beta * y[0] * y[1] 
    ydot[1] = +beta * y[0] * y[1]  - gamma * y[1]
    ydot[2] = gamma * y[1] 
    return ydot




def seirMu_function(t, y,beta,sigma,gamma,mu): 
    ''' y = [S_hat,E_hat,I_hat,R_hat] such that S_hat + E_hat + I_hat + R_hat = 1
    dS_hat / dt = -beta * S_hat*(E_hat+I_hat) 
    dE_hat / dt = +beta * S_hat *(E_hat+I_hat) - sigma * E_hat
    dI_hat / dt = +sigma * mu * E_hat - gamma * I_hat 
    dR_hat / dt = gamma * I_hat '''

    ydot = np.zeros(4)
    ydot[0] = -beta * y[0] *(y[1]+ y[2]) #S
    ydot[1] = +beta * y[0] *( y[1]+ y[2]) - sigma * y[1] #E
    ydot[2] = +sigma * mu * y[1] - gamma * y[2] #I
    ydot[3] = gamma * y[2] #R
    return ydot

