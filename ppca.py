
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.model_selection import train_test_split
import os, pickle
import numpy as np
from numpy import log, sum, exp, prod
from numpy.linalg import det
from numpy.random import beta, binomial, dirichlet, uniform, gamma, seed, multinomial, gumbel, rand, multivariate_normal, normal
from scipy.stats import wishart #, norm, randint, bernoulli, beta, multinomial, gamma, dirichlet, uniform
from scipy.special import digamma
from imp import reload
from copy import deepcopy
#import seaborn as sns
import pandas as pd


#############################################
## Probabilistic PCA / see Bishop book
#############################################

N = 5000
D = 10
K = 3    # latent dimension M in book
MCsim = 50

#-----------------
# Generate data: #
#-----------------
#X = uniform(size = (N,D)); X.shape
X = normal(loc=0, scale=1, size = (N,D)); X.shape
#---------------------------------------------------

W = np.empty((D,K,MCsim)) 
sigma2 = np.empty((1,MCsim))           

#-------------
# Initialize:
#-------------
i = 0

W[:,:,i] = normal(loc=0, scale=1, size = (D,K))
sigma2[0,i] = gamma(1)


for i in range(1,MCsim):

    if i % 10 == 0: print("Iter: ",i)

    #--------
    # E-Step
    #--------
    M = (W[:,:,i-1].T) @ W[:,:,i-1] + (sigma2[0,i-1] * np.eye(K))
    M1 = np.linalg.inv(M) 

    x_mean = np.mean(X, axis=0)
    X_mean = np.tile(x_mean, (N,1))

    Ez = M1 @ (W[:,:,i-1].T) @ (X - X_mean).T       # 12.54
    
    z_moment2 = np.empty((N,K,K)) ; Ezz = 0
    for n in range(N) : z_moment2[n,:,:] = sigma2[0,i-1] * M1 + np.dot(Ez[:,n].reshape(K,1),Ez[:,n].reshape(1,K))     

    Ezz = z_moment2.sum(axis=0)       # 12.55

    #----------
    # M-step:
    #----------
    x_Ez = (X - X_mean).T @ Ez.T

    W[:,:,i] = x_Ez @ np.linalg.inv(Ezz)     # 12.56

    # Do summing in 12.57 term-wise:
    first_term = np.sum((X - X_mean)**2)

    # 2nd term...
    second_term = -2*Ez.T.sum(axis=0) @ W[:,:,i].T @ (X - X_mean).sum(axis=0)

    # third term
    third_term = np.trace(z_moment2[i,:,:] @ W[:,:,i].T @ W[:,:,i])

    sigma2[0,i] = (first_term + second_term + third_term)/(N*D)

print('Completed!')    
#-------------------------------------------------------------------------------

sigma2

W[:,:,:<ya]
