import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

## visualise prior distribution with mean 0 and spherical covariance matrix (2x2 identity matrix)
x, y = np.mgrid[-1:1.01:.01, -1:1.01:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
random_var = multivariate_normal([0, 0], [[1, 0], [0, 1]])

## now we present a contour plot with probability density function
fig = plt.figure(figsize=(10,15))
ax = fig.add_subplot(321)
ax.contourf(x, y, random_var.pdf(pos))

## generate weights from a random multivariate normal with mean 0 and 2x2 covariance matrix with value 0.3 
## accross the diagonals for 20 samples
ax1 = fig.add_subplot(322)
x, y = np.random.multivariate_normal([0,0], [[0.3, 0], [0, 0.3]], 20).T

## generate data from -1 to 1 for 201 points in between (step of 0.01) rounded to 2 decimal places
# X_test_data = np.around(np.arange(-1,1.01,0.01), decimals=2)
X_test_data = np.linspace(-1,1,201)

## iterate from 0 to 201 creating straight lines from the weights and plotting the weights against each other
for i in range(0,x.shape[0]):
    y_data = X_test_data*x[i]+y[i]
    ax1.plot(X_test_data, y_data)
#     ax1.scatter(x, y)

    
w = [-1.3, 0.5]
epsilon_noise = np.random.normal(0,0.3,201)

## create our x data set from -1 to 1 with a step of 0.01 rounded to 2 decimal places
X = np.linspace(-1,1,201)

## as per the formula we compute the values for y, where yi = w0xi + w1 + epsilon
Y = np.array(w[0]*X+ w[1]  + epsilon_noise)
ax2 = fig.add_subplot(323)

## we then plot this data on a graph
ax2.plot(X, Y)


## here is where we calculate the posterior
beta = 0.3
phi_x = np.matrix([np.ones(201), X]).T
t_data = np.matrix([Y]).T
conj_prior_mean = np.matrix([0,0]).T
conj_prior_cov = np.matrix([[1,0], [0,1]])
# print(phi_x.shape)

# print(conj_prior_mean.shape)
# print(conj_prior_cov.shape)

covN = inv(inv(conj_prior_cov) + beta * phi_x.T * phi_x)
meanN = covN * (inv(conj_prior_cov) * conj_prior_mean + beta * phi_x.T * t_data)


x, y = np.mgrid[-1:1:.01, -2.0:1.01:0.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
# squeezed_mean = np.squeeze(np.asarray(meanN))
random_var = multivariate_normal(np.squeeze(np.asarray(meanN)), covN)

ax3 = fig.add_subplot(325)
ax3.contourf(x, y, random_var.pdf(pos), colors=['white', 'red'], levels=17, antialiased=True)

print("Shape of the mean",meanN.shape)
ax4 = fig.add_subplot(326)
x_a, y_a = np.random.multivariate_normal(np.squeeze(np.asarray(meanN)), covN, 20).T
test_data = np.linspace(-1,1,201)

for i in range(0,x_a.shape[0]):
    y_data = test_data*y_a[i]+x_a[i]
    ax4.plot(test_data, y_data)
    ax4.scatter(X, Y)

fig.savefig('graph.png')
