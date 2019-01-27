import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from numpy import random
from scipy.stats import multivariate_normal
from math import exp, sqrt, pi
from scipy.spatial import distance

## this function draws an o when the value is -1 and x otherwise
def draw_x_o(data, m, text):
    print(text)
    for i in range(3):
        for j in range(3):
            if data[i][j] == -1:
                print("o", end=" ")
            else:
                print("x", end=" ")
        print()
    print()

## we reorder the indexes, which we then use to plot the evidence
def create_index_set(evidence):
    E = evidence.sum(axis=1)
    dist = distance.squareform(distance.pdist(evidence, 'euclidean'))
    np.fill_diagonal(dist, np.inf)
    
    L = []
    D = list(range(E.shape[0]))
    L.append(E.argmin())
    D.remove(L[-1])
    
    while len(D) > 0:
        # add d if dist from d to all other points in D
        # is larger than dist from d to L[-1]
        N = [d for d in D if dist[d, D].min() > dist[d, L[-1]]]
        
        if len(N) == 0:
            L.append(D[dist[L[-1],D].argmin()])
        else:
            L.append(N[dist[L[-1],N].argmax()])
        
        D.remove(L[-1])
    
    # reverse the resulting index array
    return np.array(L)[::-1]

## we create 512 3x3 matrices with values -1 or 1
def getData():
    cartesian_product = list(it.product([-1, 1], repeat=9))
    data = []
    for p in cartesian_product:
        data.append(np.reshape(np.asarray(p), (3, 3)))
    return np.array(data)

## the following are the models given M0 to M3
def M0():
    return 1/512

def M1(data, theta):
    p = 1
    for i in range(3):
        for j in range(3):
            exponent = data[i, j] * theta[0] * (j-1)
            p*= 1/(1+ exp(-exponent))
    return p

def M2(data, theta):
    p = 1
    for i in range(3):
        for j in range(3):
            exponent = data[i, j] * ((theta[0] * (j-1)) + (theta[1] * (1-i)))
            p*= 1/(1+ exp(-exponent))
    return p

def M3(data, theta):
    p = 1
    for i in range(3):
        for j in range(3):
            exponent = data[i, j] * ((theta[0] * (j-1)) + (theta[1] * (1-i)) + theta[2])
            p*= 1/(1+ exp(-exponent))
    return p

## create prior given sigma and how many samples
def prior(sigma,model,sampleSize):
    if model == 0:
        return np.zeros((4,512))
    
    mean = np.zeros(model)
    covar = sigma*np.eye(model)
    samples = random.multivariate_normal(mean,covar,sampleSize)
    return samples

## q29 we give a mean of 5 for each model
def prior2(sigma,model,sampleSize):
    if model == 0:
        return np.zeros((4,512))
    
    mean = np.full((model), 5)
    covar = sigma*np.eye(model)
    samples = random.multivariate_normal(mean,covar,sampleSize)
    return samples

## q29 we change the covariance to no longer be diagonal
def prior3(sigma,model,sampleSize):
    if model == 0:
        return np.zeros((4,512))
    
    mean = np.full((model), 5)
    covar = sigma*np.full((model,model), 1)
    samples = random.multivariate_normal(mean,covar,sampleSize)
    return samples

## followed from given formula in coursework
def monte_carlo(data, sigma, model, sampleSize):
    samples = prior(sigma, model, sampleSize)
    evidence = 0
    for i in range(sampleSize):
        if model == 1:
            evidence += M1(data, samples[i,:])
        if model == 2:
            evidence+= M2(data, samples[i,:])
        if model == 3:
            evidence+= M3(data, samples[i,:])
    return evidence/sampleSize

def compute_evidence(data, sigma, sampleSize):
    evidence = np.zeros((4,512))
    evidence[0,:] = np.ones((512)) * M0()

    for i in range(512):
        evidence[1,i] = monte_carlo(data[i,:], sigma, 1, sampleSize)
        evidence[2,i] = monte_carlo(data[i,:], sigma, 2, sampleSize)
        evidence[3,i] = monte_carlo(data[i,:], sigma, 3, sampleSize)
    return np.transpose(evidence)

dataSet = getData()
print('dataset', dataSet.shape)
evidence = compute_evidence(dataSet, 1000, 1000)

print('evidence', evidence)
evsum = np.sum(evidence,axis=0)
print(evsum)

## re order indices for plotting a pretty looking graph
index = create_index_set(evidence)

print('index shape', index)

## we display the x and o representing the values that correspond to the maximal and minimal dataset for each model
for m, dat in enumerate(np.transpose(evidence)):
    draw_x_o(dataSet[dat.tolist().index(max(dat))], m, "Maximal dataset for M{}".format(m)) 
print('---------------------------')

for m, dat in enumerate(np.transpose(evidence)):
    draw_x_o(dataSet[dat.tolist().index(min(dat))], m, "Minimal dataset for M{}".format(m)) 


plt.figure()
plt.plot(evidence[:,3],'g', label= "P(D|M3)")
plt.plot(evidence[:,2],'r', label= "P(D|M2)")
plt.plot(evidence[:,1],'b', label= "P(D|M1)")
plt.plot(evidence[:,0],'m--', label = "P(D|M0)")
plt.xlim(0,520)
plt.ylim(0,0.12)
plt.xlabel('All dataSet without ordered indices')
plt.ylabel('evidence')
plt.title('evidence of all data sets')
plt.legend()
plt.show()

plt.figure()
plt.plot(evidence[index,3],'g', label= "P(D|M3)")
plt.plot(evidence[index,2],'r', label= "P(D|M2)")
plt.plot(evidence[index,1],'b', label= "P(D|M1)")
plt.plot(evidence[index,0],'m--', label = "P(D|M0)")
plt.xlim(0,520)
plt.ylim(0,0.12)
plt.xlabel('All dataSet')
plt.ylabel('evidence')
plt.title('evidence of all data sets with ordered indices')
plt.legend()
plt.show()

plt.figure()
plt.plot(evidence[index,3],'g', label= "P(D|M3)")
plt.plot(evidence[index,2],'r', label= "P(D|M2)")
plt.plot(evidence[index,1],'b', label= "P(D|M1)")
plt.plot(evidence[index,0],'m--', label = "P(D|M0)")
plt.xlim(0,80)
plt.ylabel('subset of possible dataSet')
plt.ylabel('evidence')
plt.ylim(0,0.12)
plt.title('evidence of subset of possible data sets')
plt.legend()
plt.show()


