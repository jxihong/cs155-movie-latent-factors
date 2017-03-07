import numpy as np
import math

def error(Y, U, V, a, b):
    """
    Computes the RMSE error.
    """
    # Exclude missing ratings
    W = Y.copy()
    W[W != 0] = 1
    W[W == 0] = 0
    
    M, N = Y.shape
    
    # Stack bias vectors into MxN matrices, so I don't have to iterate
    # over individual elements
    A = np.hstack([a.reshape(-1, 1)] * N)
    B = np.vstack([b] * M)
    
    error = np.sum(W * ((Y - np.dot(U, V.T) - A - B)**2))/len(W[W!=0])
    
    return error

def factor_matrix(Y, k, eta=0.01, reg=0):
    """
    Factors a MxN matrix Y into a product of a Mxk matrix U
    and Nxk matrix V using SGD.
    """
    M, N = Y.shape
    
    U = np.random.uniform(low=-0.5, high=0.5, size=(M, k))
    V = np.random.uniform(low=-0.5, high=0.5, size=(N, k))
    a = np.zeros(M)
    b = np.zeros(N)

    # Exclude missing ratings
    users, movies = Y.nonzero()

    iteration = 0

    while True:
        data = zip(users, movies) 
        for i,j in np.random.permutation(data):
            e = Y[i, j] - np.dot(U[i, :], V[j, :]) - a[i] - b[j]
            
            U[i, :] += eta * (e * V[j, :] - reg * U[i, :])
            V[j, :] += eta * (e * U[i, :] - reg * V[j, :])
            
            a[i] += eta * (e - reg * a[i])
            b[j] += eta * (e - reg * b[j])
            
        rmse = error(Y, U, V, a , b) # RMSE error
        
        if rmse < 0.1:
            break
        if iteration > 50:
            break

        print iteration, rmse
        iteration += 1

    return U, V, a, b


if __name__=='__main__':
    ratings = np.loadtxt('data/ratings.out',delimiter=',')

    mean_rating = np.mean(ratings[np.nonzero(ratings)])
    # Take out mean of all ratings
    ratings_normed = ratings.copy()
    ratings_normed[np.nonzero(ratings_normed)] -= mean_rating

    U, V, a, b = factor_matrix(ratings_normed, 20)
    
