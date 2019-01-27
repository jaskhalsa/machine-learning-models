import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
import math

## here we present the test data from -1 to 1 with a step of 0.1
X = np.linspace(-10,10,500).reshape(-1, 1)
variance_sq = np.var(X)**2

## create training data from -pi to pi with 7 points
X_train = np.linspace((-math.pi), math.pi, num=7).T.reshape(7,1)
print('x train shape', X_train.shape)

## create epsilon noise with normal dist - mean=0 and sigma^2=0.5
epsilon_noise = np.random.normal(0,0.5,7).reshape(7,1)
print('noise', epsilon_noise.shape)

## create points for the y training data with y=sin(x) + epsilon_noise
Y_train = np.sin(X_train) + epsilon_noise
print('y train',Y_train.shape)
print('np sin',np.sin(X_train).shape)



## this function takes two arrays, finds the sqr euclidian dist and returns a kernel aka cov
def kernel(a, b, length_scale):
    sqrdist = cdist(a, b, 'sqeuclidean')
    result = np.exp(-1 * (1/length_scale**2) * sqrdist)
    if(len(a) == len(b)):
        result += 0 * np.eye(result.shape[0])
    return result

## we just plug the formula from bishop here and return the mean and cov determined from the kernal operations
def form_posterior(X_test, X_train, Y_train, l=3):
    K_test_train = kernel(X_test, X_train, l).reshape(500, 7)

    ## we add noise (below) so that it has a level of uncertainty and doesn't fit exactly to the data points (overfit)
    K_train_train_inv = inv(kernel(X_train, X_train, l) + 0.3 * np.eye(X_train.shape[0]))
    
    K_test_test = kernel(X_test, X_test, l)
    K_train_test = kernel(X_train, X_test, l)

    mean = K_test_train.dot(K_train_train_inv).dot(Y_train)
    cov = np.subtract(K_test_test, (K_test_train.dot(K_train_train_inv).dot(K_train_test)))
    return mean, cov

mean, cov = form_posterior(X, X_train, Y_train)

print('cov shape', cov.shape)
print('mean shape', mean.shape)

## we plot for 20 samples
samples = np.random.multivariate_normal(mean.ravel(), cov, 20).T
print('samples', samples.shape)

# ## plot graphs
fig = plt.figure(figsize=(25,25))
    
# ax1 = fig.add_subplot(221)
ax1 = fig.add_subplot(221)
ax1.title.set_text('Length scale 3')
ax1.plot(X, samples)
ax1.scatter(X_train, Y_train, marker='o')
plt.show()

fig.savefig('graph-q14.png')