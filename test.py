
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


N = 5000
D = 4
K = 10    # latent dimension M in book
MCsim = 100

# Generate data:
#-------------------
#X = uniform(size = (N,D)); X.shape
X = normal(loc=0, scale=1, size = (N,D)); X.shape

x_mean = np.mean(X, axis=0)

W = np.empty((D,K,MCsim)) 
sigma2 = np.empty((1,MCsim))           

# Initialize:
#-------------
i = 0
W[:,:,i] = normal(loc=0, scale=1, size = (D,K))
sigma2[0,i] = gamma(1)

ww = (W[:,:,i].T) @ W[:,:,i]

# E-Step
M = ww + sigma2[0,i]*np.eye(K)
M1 = np.linalg.inv(M) 

X_mean = np.tile(x_mean, (N,1))
#x_centered[n,:].reshape(D,1) @ zm.T
X_cent = (X - X_mean).T
X_cent.shape

Ez = M1 @ (W[:,:,i].T) @ X_cent       # 12.54
Ez.T.shape

z_moment2 = np.empty((N,K,K))
#z_moment1, z_moment2 = np.empty((N,K)), np.empty((N,K,K))
#x_centered = np.empty((N,D))
#x_Ez = 0
Ezz = 0
#cross_pro = np.empty((N,D))
for n in range(N) : 
    #x_centered[n,:] = (X[n,:] - x_mean)
    #z_moment1[n,:] = M1 @ (W[:,:,i].T) @ x_centered[n,:]    #12.54
    #zm = z_moment1[n,:].reshape(K,-1)
    z_moment2[n,:,:] = sigma2[0,i] * M1 + np.dot(Ez[:,n].reshape(K,1),Ez[:,n].reshape(1,K))     # 12.55
    #x_Ez += x_centered[n,:].reshape(D,1) @ z_moment1[n,:].reshape(K,1).T
    #np.sum((X - X_mean)[n,:]**2)

#(X - X_mean).flatten().reshape(-1,N*D)w

x_Ez = (X - X_mean).T @ Ez.T
Ezz = z_moment2.sum(axis=0)

W_new = x_Ez @ np.linalg.inv(Ezz)     # 12.56
W_new.shape

# Do summing in 12.57 term-wise:
euclid = np.sum((X - X_mean)**2,axis=1)
euclid.shape

# third term
np.trace(z_moment2[0,:,:] @ W_new.T @ W_new)

# 2nd term...
Ez.T.shape

a = -2 * Ez.T @ W_new.T
a.shape

z_moment2.shape

z_moment2[0,:,:].shape





