# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:52:49 2016

@author: Alex Kerr

Define a set of operations for use on Molecule objects that we don't want being methods.
"""

import os
import errno
from copy import deepcopy
import pickle

import numpy as np

#change in position for the finite difference equations
ds = 1e-5
dx = ds
dy = ds
dz = ds

vdx = np.array([dx,0.0,0.0])
vdy = np.array([0.0,dy,0.0])
vdz = np.array([0.0,0.0,dz])
           
save_dir = "./kappa_save/"

def _path_exists(path):
    """Make a path if it doesn't exist"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
            
def _file_exists(path):
    """Return True if file path exists, False otherwise"""
    return os.path.isfile(path)

def save(molecule):
    """Save a pickled molecule into /kerrNM/systems/"""
    path = save_dir + molecule.name
    _path_exists(path)
    pickle.dump(molecule, open( path + "/mol.p", "wb" ) )
    
def load(name):
    """Load a pickled molecule given a name"""
    return pickle.load(open(save_dir+name+"/mol.p", "rb"))
    
def _calculate_hessian(molecule, stapled_index, numgrad=False):
    """Return the Hessian matrix for the given molecule after calculation."""
    
    N = len(molecule)
    
    H = np.zeros([3*N,3*N])
    
    if numgrad:
        calculate_grad = molecule.define_gradient_routine_numerical()
    else:
        calculate_grad = molecule.define_gradient_routine_analytical()
    
    for i in range(N):
        
        ipos = molecule.posList[i]
        
        ipos += vdx
        plusXTestGrad,_,_ = calculate_grad()
        ipos += -2.0*vdx
        minusXTestGrad,_,_ = calculate_grad()
        ipos += vdx + vdy
        plusYTestGrad,_,_ = calculate_grad()
        ipos += -2.0*vdy
        minusYTestGrad,_,_ = calculate_grad()
        ipos += vdy + vdz
        plusZTestGrad,_,_ = calculate_grad()
        ipos += -2.0*vdz
        minusZTestGrad,_,_ = calculate_grad()
        ipos += vdz
        
        xiRow = (plusXTestGrad - minusXTestGrad)/2.0/dx
        yiRow = (plusYTestGrad - minusYTestGrad)/2.0/dy
        ziRow = (plusZTestGrad - minusZTestGrad)/2.0/dz
        
        H[3*i    ] = np.hstack(xiRow)
        H[3*i + 1] = np.hstack(yiRow)
        H[3*i + 2] = np.hstack(ziRow)
        
    if stapled_index is not None:
        dk = .1
        H[3*stapled_index  , 3*stapled_index  ] += dk
        H[3*stapled_index+1, 3*stapled_index+1] += dk
        H[3*stapled_index+2, 3*stapled_index+2] += dk
       
    return H
    
def hessian(molecule):
    """Return the Hessian for a molecule; if it doesn't exist calculate it otherwise load it
    from the numpy format."""
    path = save_dir + molecule.name
    #check if path exists
    _path_exists(path)
    #if Hessian file exists then load it; otherwise calculate it and save it
    if _file_exists(path + "/hessian.npy"):
        print("Loading Hessian matrix from file...")
        return np.load(path + "/hessian.npy")
    else:
        print("Calculating the Hessian matrix for " + molecule.name + "...")
        H = _calculate_hessian(molecule)
        print("Done!")
        np.save(path + "/hessian.npy", H)
        return H
    
def evecs(hessian):
    """Return the eigenvalues and eigenvectors associated with a given Hessian matrix."""
    w,vr = np.linalg.eig(hessian)
    return w,vr
        