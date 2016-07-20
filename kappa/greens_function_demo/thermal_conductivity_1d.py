# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:02:19 2016

@author: Alex Kerr

Demonstrating the Green's function calculation for a 1D system of atoms.
"""

import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

def main():
    
    k0 = 1.
    m0 = 1.
    gamma0 = 0.1
    N = 3
    
    #get neighbor lists
    nLists = find_neighbors(N)
    
    #get Kmatrix (N x N)
    KMatrix = calculate_K_matrix(N,k0,nLists)
    
    #get gamma matrix (N x N)
    gammaMatrix = calculate_gamma_matrix(N,gamma0)
    
    #mass matrix now (N x N)
    MMatrix = calculate_M_matrix(N,m0)
    
    #find eigenvalues and eigenvectors
    #2N eigenvalues: N lambdha and N lambdha*
    #2N eigenvectors of length 2N
    val,vec = calculate_evec(MMatrix,gammaMatrix,KMatrix)

    coeff = calculate_greens_function(val, vec, MMatrix, gammaMatrix)
    
    kap = calculate_thermal_conductivity(coeff, val, vec, gammaMatrix, KMatrix)
    
    print(kap)
    
def calculate_thermal_conductivity(coeff, val, vec, gMat, kMat):
    """Return the thermal conductivity coefficient from one particle to another."""
    
    i = 1
    j = 2
    
    #driven atom
    #assuming same drag constant as other driven atom
    driver = 0
    
    kap = 0.j
    
    for sigma in range(len(val)):
        
        for tau in range(len(val)):
            
            try:
                valTerm = (val[sigma] - val[tau])/(val[sigma] + val[tau])
            except ZeroDivisionError:
                print("Encountered the error")
                continue
            
#            print(coeff[sigma,driver]*coeff[tau,driver])
#            print(vec[i,sigma])
            a = coeff[sigma,driver]*coeff[tau,driver]*vec[i,sigma]*vec[j,tau]*valTerm
#            print(a)
            kap += a
    
#    return kap*2*gMat[driver,driver]*kMat[i,j]
    return kap
    
def calculate_greens_function(val, vec, massMat, gMat):
    """Return the 2N x N Green's function coefficient matrix."""
    
    N = len(vec)//2
    
    #need to determine coefficients in eigenfunction/vector expansion
    # need linear solver to solve equations from notes
    # AX = B where X is the matrix of expansion coefficients
    
    A = np.zeros((2*N, 2*N), dtype=complex)
    A[:N,:] = vec[:N,:]

    #adding mass and damping terms to A
    lamda = np.tile(val, (N,1))

    A[N:,:] = np.multiply(A[:N,:], np.dot(massMat,lamda) + np.dot(gMat,np.ones((N,2*N))))
    
    #now prep B
    B = np.concatenate((np.zeros((N,N)), np.identity(N)), axis=0)

    return np.linalg.solve(A,B)
    
def calculate_evec(M,G,K):
    
    N = len(M)
    
    a = np.zeros([N,N])
    a = np.concatenate((a,np.identity(N)),axis=1)
    b = np.concatenate((K,G),axis=1)
    c = np.concatenate((a,b),axis=0)
    
    x = np.identity(N)
    x = np.concatenate((x,np.zeros([N,N])),axis=1)
    y = np.concatenate((np.zeros([N,N]),-M),axis=1)
    z = np.concatenate((x,y),axis=0)
    
    w,vr = linalg.eig(c,b=z,right=True)
    
    return w,vr
    
def calculate_M_matrix(N,m):
    """Return the scaled identity matrix (Cartesian coords.)"""
    return m*np.identity(N)
    
def calculate_gamma_matrix(N,g0):
    """Return the damping matrix, assuming only the ends are damped."""
    
    gMatrix = np.zeros([N,N])
    gMatrix[0,0] = g0
    gMatrix[N-1,N-1] = g0
    
    return gMatrix
    
def calculate_K_matrix(N,k0,nLists):
    """Return the Hessian of a linear chain of atoms assuming only nearest neighbor interactions."""
    
    KMatrix = np.zeros([N,N])
    
    for i,nList in enumerate(nLists):
        KMatrix[i,i] = k0*len(nList)
        for neighbor in nList:
            KMatrix[i,neighbor] = -k0
    
    return KMatrix
    
def find_neighbors(N):
    """Return a list of neighbor lists indexed like corresponding atoms, calculated for a single line"""
    
    nLists = []
    for i in range(N):
        if i == 0:
            nLists.append([1])
        elif i == N-1:
            nLists.append([N-2])
        else:
            nLists.append([i-1,i+1])
            
    #return as numpy array
    return np.array(nLists)
    
def plot(y,x):
    
    for i in range(np.shape(y)[1]):
        plt.plot(x,y[:,i] + 2*i*np.ones(len(x)))
    plt.ylabel('displacement')
    plt.xlabel('time')
    plt.show()
    
    
main()
