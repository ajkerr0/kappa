# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:02:19 2016

@author: Alex Kerr

Demonstrating the Green's function calculation for a 1D system of atoms.
"""

import numpy as np
import scipy.linalg as linalg


def main():
    
    k0 = 1.
    m0 = 1.
    gamma0 = 0.1
    N = 6
    
    #get neighbor lists
    nLists = find_neighbors(N)
    
    #get Kmatrix
    KMatrix = calculate_K_matrix(N,k0,nLists)
    
    #get gamma matrix
    gammaMatrix = calculate_gamma_matrix(N,gamma0)
    
    #mass matrix now
    MMatrix = calculate_M_matrix(N,m0)
    
    #find eigenvalues and eigenvectors
    val,vec = calculate_evec(MMatrix,gammaMatrix,KMatrix)
    
    #testing val,vec
    testVec(val,vec)
    
    gFxn = calculate_green_function(vec)
    
#    print gFxn
    
def calculate_green_function(vec):
    """Return the Green function"""
    
    N = len(vec)/2
    print N
    gfxn = np.zeros(N,dtype=complex)
    
    
    return gfxn
    
def testVec(val,vec):
    
#    N = len(vec)/2
#    
#    ind = 0
#    
#    val=val[ind]
#    vec=vec[ind]
#    
#    print vec
#    
#    for entry in vec[-N:]:
#        print entry*val
    
    x = -.0008283 + .068519j
    val = -.00222166+1.93169j
    print x*val
    
    
def calculate_evec(M,G,K):
    
    N = len(M)
    
    a = np.zeros([N,N])
    a = np.concatenate((a,np.identity(N)),axis=1)
    b = np.concatenate((K,G),axis=1)
    c = np.matrix(np.concatenate((a,b),axis=0))
    
    x = np.identity(N)
    x = np.concatenate((x,np.zeros([N,N])),axis=1)
    y = np.concatenate((np.zeros([N,N]),-M),axis=1)
    z = np.matrix(np.concatenate((x,y),axis=0))   
    
    w,vr = linalg.eig(c,b=z,right=True)
    
    return w,vr
    
def calculate_M_matrix(N,m):
    """Return the scaled identity matrix (Cartesian coords.)"""
    return m*np.matrix(np.identity(N))
    
def calculate_gamma_matrix(N,g0):
    """Return the damping matrix, assuming only the ends are damped."""
    
    gMatrix = np.matrix(np.zeros([N,N]))
    gMatrix[0,0] = g0
    gMatrix[N-1,N-1] = g0
    
    return gMatrix
    
def calculate_K_matrix(N,k0,nLists):
    """Return the Hessian of a linear chain of atoms assuming only nearest neighbor interactions."""
    
    KMatrix = np.matrix(np.zeros([N,N]))
    
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
    
    
main()
