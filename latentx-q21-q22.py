import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import matplotlib.pyplot as plt
import math
import scipy as sp
import scipy.optimize as opt

## returns X from 0 to 4pi for N data points, A being a 10 x 2 matrix from a random normal distribution with mean 0 and sd 1
def getData():
    X = np.linspace(0,4*math.pi,N).reshape(1,N)
    A = np.random.normal(0, 1, size=(10,2))
    return X, A
## for q22, we instead generate 20 random numbers and not from a random normal dist 
def getData2():
    X = np.linspace(0,4*math.pi,N).reshape(1,N)
    A = np.random.randn(20)
    A = A.reshape((10,2))
    return X, A

## apply the non lin described in the coursework
def f_non_lin(X):
    Y = np.zeros((N,2))
    Y[:,0] = X*np.sin(X)
    Y[:,1] = X*np.cos(X) 
    return Y
## apply the linear described in the coursework
def f_linear(A, non_linear):
    return non_linear.dot(A.T)


def f(W):
    ## We define the dimensionality of our weight matrix W as (10, 2)
    ## the below is from eq 12.44 in Bishop
    D = 10
    L = 2
    W = W.reshape(D,L)
    C = np.eye(D) + np.dot(W, np.transpose(W))
    C_Inverse = inv(C)
    Det_C = det(C)
    R1 = 10*np.log(2*math.pi)
    R2 = np.log(Det_C)
    R3 = np.trace(np.dot(np.dot(Y, C_Inverse), np.transpose(Y)))
    return 0.5*N*(R1 + R2 + R3)


def dfx(W):
    ## We define the dimensionality of our weight matrix W as (10, 2)
    W = W.reshape(10,2)
    C = np.eye(10) + W.dot(W.T)
    C_Inverse = inv(C)
    S = np.diag(np.diag(Y.T.dot(Y)))
    
    ## create empty matricies for our result and also for our delta
    result = np.zeros(W.shape)
    delta = np.zeros((W.shape[0], W.shape[0]))

    ## calculate gradient for each element
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            M = np.zeros((W.shape[0], W.shape[1]))
            M[i,j] = 1
            delta = np.dot(M,np.transpose(W)) + np.dot(W,np.transpose(M))
            result[i,j] = N*0.5*(np.trace(np.dot(C_Inverse,delta)) + np.trace(np.dot(np.dot(np.dot(-C_Inverse,delta),C_Inverse),S)))
    return result.reshape(W.shape[0] * W.shape[1])


N = 100
X, A = getData()

## generate our output
Y = f_linear(A, f_non_lin(X))
print('y shape', Y.shape)

## generate a prior
W0 = np.reshape(np.random.rand(20), (10,2))

## attain the paramaters
Wstar = opt.fmin_cg(f,W0, fprime=dfx).reshape(10,2)
WTW_star = np.dot(np.transpose(Wstar), Wstar)

## produce our x inputs given the output Y and our parameters Wstar
X_input = np.dot(np.dot(Y, Wstar), inv(WTW_star))

X, A = getData2()

print('Y shape', Y.shape)
actual_x = f_non_lin(X)
plt.figure()
plt.plot(actual_x[:,0],actual_x[:,1], 'r.')
plt.title("the actual X in two-dimensional space")
plt.legend(["original X"])

## Question 22
Y_not_normal = f_linear(A, f_non_lin(X))


# ## plot graphs
fig = plt.figure(figsize=(25,25))
    
# ax1 = fig.add_subplot(221)
ax1 = fig.add_subplot(221)
ax1.title.set_text('Question 21')
ax1.scatter(X_input[:,0],X_input[:,1])

ax2 = fig.add_subplot(222)
ax2.title.set_text('Question 22')
ax2.scatter(Y_not_normal[:,0],Y_not_normal[:,1])

fig.savefig('graph-q21-22.png')
plt.show()


