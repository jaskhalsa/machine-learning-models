import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist

## create test data from -1 to 1, with a step of 0.01
X_test_data = np.linspace(-1,1,201).reshape(-1, 1)
variance_sq = np.var(X_test_data)**2

## this function takes two arrays, finds the sqr euclidian dist and returns a kernel
def kernel(a, b, length_scale):
    sqrdist = cdist(a, b, 'sqeuclidean')
    return variance_sq*np.exp(-0.5 * (1/length_scale**2) * sqrdist)

## compute mean and covariance
mu = np.zeros(X_test_data.shape)
cov = kernel(X_test_data, X_test_data, 0.1)
samples = np.random.multivariate_normal(np.squeeze(np.asarray(mu)), cov, 10).T
print(samples)

## we plot the random multivariate normal for 4 different length scales beginning at 0.1 and 5 samples for each
## we just chuck it into a results array for laziness / simplicity for when we have to plot
results = []
for i in range(0, 4):
    cov = kernel(X_test_data, X_test_data, i+0.1)
    results.append(np.random.multivariate_normal(np.squeeze(np.asarray(mu)), cov, 5).T)

## plot graphs
fig = plt.figure(figsize=(15,15))
    
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.title.set_text('Length scale 0.1')
ax2.title.set_text('Length scale 1.1')
ax3.title.set_text('Length scale 2.1')
ax4.title.set_text('Length scale 3.1')
ax1.plot(X_test_data, results[0])
ax2.plot(X_test_data, results[1])
ax3.plot(X_test_data, results[2])
ax4.plot(X_test_data, results[3])
plt.show()

fig.savefig('graph-q13.png')
