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
    
def _calculate_hessian(molecule, stapled_index, numgrad=False, onsite=None):
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
        dk = 1.
        H[3*stapled_index  , 3*stapled_index  ] += dk
        H[3*stapled_index+1, 3*stapled_index+1] += dk
        H[3*stapled_index+2, 3*stapled_index+2] += dk
        
    if onsite is not None:
        H += np.diag(onsite*np.ones(H.shape[0]))
       
    return H

def hess_bond_stretching(pos, i, j, kb, b0):
    """
    Return the analytical Hessian matrix of the bond stretch energy.
    """
    # create stack of 3x3 blocks that will compose the hessian
    hess = np.zeros((pos.shape[0]**2,3,3))
    posij = pos[i] - pos[j]
    rij = np.linalg.norm(posij, axis=1)
    # create stack of subblocks where each layer is the outerproduct of pos diffs
    block  = b0[:,None,None]*np.einsum('ki,kj->kij', posij, posij)/rij[:,None,None]**3
    block += (1. - b0/rij)[:,None,None]*np.tile(np.eye(3), (i.shape[0],1,1))
    block  = 2.*kb[:,None,None]*block
    # add subblocks to the hessian
    # first add positives at the unraveled diagonal
    np.add.at(hess, (pos.shape[0]+1)*i, block)
    np.add.at(hess, (pos.shape[0]+1)*j, block)
    # now add off diagonal blocks
    np.add.at(hess, pos.shape[0]*i + j, -block)
    np.add.at(hess, pos.shape[0]*j + i, -block)
    return np.hstack(np.hstack(hess.reshape(pos.shape[0],pos.shape[0],3,3)))

def hess_bond_bending(pos, i, j, k, kt, t0):
    """
    Return the Hessian of the bond bend energy
    """
    # create stack of 3x3 blocks that will compose the hessian
    hess = np.zeros((pos.shape[0]**2,3,3))
    posij = pos[i] - pos[j]
    poskj = pos[k] - pos[j]
    rij, rkj = np.linalg.norm(posij, axis=1), np.linalg.norm(poskj, axis=1)
    cos_t = np.einsum('ij,ij->i',posij,poskj)/(rij*rkj)
    sin_t = np.sqrt(1.-(cos_t**2))
    theta = np.rad2deg(np.arccos(cos_t))
    
    dtdri = (posij*(cos_t/rij)[:,None] - poskj/(rkj[:,None]))/((rij*sin_t)[:,None])
    dtdrk = (poskj*(cos_t/rkj)[:,None] - posij/(rij[:,None]))/((rkj*sin_t)[:,None])
    
    ii  = (cos_t/rij)[:,None, None]*np.tile(np.eye(3), (i.shape[0],1,1))
    ii -= (cos_t/rij**3)[:,None,None]*np.einsum('ki,kj->kij', posij, posij)
    ii -= (sin_t/rij)[:,None,None]*(np.einsum('ki,kj->kij', posij, dtdri)
                                   +np.einsum('ki,kj->kij', dtdri, posij))
    ii -= (rij*cos_t)[:,None,None]*np.einsum('ki,kj->kij', dtdri, dtdri)
    ii *= ((theta-t0)/(rij*sin_t))[:,None,None]*180./np.pi
    ii += np.einsum('ki,kj->kij', dtdri, dtdri)*(180.*180./np.pi/np.pi)
    ii *= 2.*kt[:,None,None]
    # w.r.t. rk now
    kk  = (cos_t/rkj)[:,None, None]*np.tile(np.eye(3), (i.shape[0],1,1))
    kk -= (cos_t/rkj**3)[:,None,None]*np.einsum('ki,kj->kij', poskj, poskj)
    kk -= (sin_t/rkj)[:,None,None]*(np.einsum('ki,kj->kij', poskj, dtdrk)
                                   +np.einsum('ki,kj->kij', dtdrk, poskj))
    kk -= (rkj*cos_t)[:,None,None]*np.einsum('ki,kj->kij', dtdrk, dtdrk)
    kk *= ((theta-t0)/(rkj*sin_t))[:,None,None]*180./np.pi
    kk += np.einsum('ki,kj->kij', dtdrk, dtdrk)*(180.*180./np.pi/np.pi)
    kk *= 2.*kt[:,None,None]
    # now the mixed derivative
    ki  = -(1./rkj)[:,None,None]*np.tile(np.eye(3), (i.shape[0],1,1))
    ki +=  (1./rkj**3)[:,None,None]*np.einsum('ki,kj->kij', poskj, poskj)
    ki += -(sin_t/rij)[:,None,None]*np.einsum('ki,kj->kij', dtdrk, posij)
    ki += -(rij*cos_t)[:,None,None]*np.einsum('ki,kj->kij', dtdrk, dtdri)
    ki *=  ((theta-t0)/(rij*sin_t))[:,None,None]*180./np.pi
    ki += np.einsum('ki,kj->kij', dtdrk, dtdri)*(180.*180./np.pi/np.pi)
    ki *= 2.*kt[:,None,None]
    # add subblocks to hessian
    np.add.at(hess, (pos.shape[0]+1)*i, ii)
    np.add.at(hess, (pos.shape[0]+1)*k, kk)
    np.add.at(hess, (pos.shape[0]+1)*j, ii + kk + ki + np.transpose(ki, (0, 2, 1)))
    # add off-diagonal blocks
    np.add.at(hess, pos.shape[0]*i + k, np.transpose(ki, (0, 2, 1)))
    np.add.at(hess, pos.shape[0]*k + i, ki)
    np.add.at(hess, pos.shape[0]*i + j, -ii - np.transpose(ki, (0, 2, 1)))
    np.add.at(hess, pos.shape[0]*j + i, -ii - ki)
    np.add.at(hess, pos.shape[0]*k + j, -kk - ki)
    np.add.at(hess, pos.shape[0]*j + k, -kk - np.transpose(ki, (0, 2, 1)))
    return np.hstack(np.hstack(hess.reshape(pos.shape[0],pos.shape[0],3,3)))

def hess_dihedral(pos, i, j, k, l, vn, gn):
    """
    Return the Hessian of the dihedral interaction.
    """
    # create stack of 3x3 blocks that will compose the hessian
    hess = np.zeros((pos.shape[0]**2,3,3))
    posij = pos[i] - pos[j]
    poskj = pos[k] - pos[j]
    poslk = pos[l] - pos[k]
    posmj = np.cross(posij, poskj)
    posnk = np.cross(poslk, -poskj)
    rij = np.linalg.norm(posij, axis=1)
    rkj = np.linalg.norm(poskj, axis=1)
    rlk = np.linalg.norm(poslk, axis=1)
    rmj = np.linalg.norm(posmj, axis=1)
    rnk = np.linalg.norm(posnk, axis=1)
    
    cross12 = np.cross(-posij, poskj)
    cross23 = np.cross(poskj, poslk)
    n1 = cross12/np.linalg.norm(cross12, axis=1)[:,None]
    n2 = cross23/np.linalg.norm(cross23, axis=1)[:,None]
    m1 = np.cross(n1, poskj/rkj[:,None])
    x,y = np.einsum('ij,ij->i', n1, n2),  np.einsum('ij,ij->i', m1, n2)
    omega = np.rad2deg(np.arctan2(y,x))
    
    dwdri = -(rkj/rmj/rmj)[:,None]*posmj
    dwdrl = -(rkj/rnk/rnk)[:,None]*posnk
    dwdrj = (np.einsum('ki,ki->k', posij, poskj)/rkj/rkj - 1)[:,None]*dwdri
    dwdrj += (np.einsum('ki,ki->k', poslk, poskj)/rkj/rkj)[:,None]*dwdrl
    dwdrk = (np.einsum('ki,ki->k', poslk,-poskj)/rkj/rkj - 1)[:,None]*dwdrl
    dwdrk += -(np.einsum('ki,ki->k', posij, poskj)/rkj/rkj)[:,None]*dwdri
    
    # calculate separate Hessian blocks
    # ii
    ii = np.zeros((i.shape[0], 3,3))
    ii[:,0,1] = -poskj[:,2]
    ii[:,0,2] =  poskj[:,1]
    ii[:,1,0] =  poskj[:,2]
    ii[:,1,2] = -poskj[:,0]
    ii[:,2,0] = -poskj[:,1]
    ii[:,2,1] =  poskj[:,0]
    ii *= (rkj/rmj/rmj)[:,None,None]
    term1 = (rkj*rkj)[:,None]*posij - np.einsum("ki,ki->k",posij,poskj)[:,None]*poskj
    term1 *= -(2.*rkj/rmj**4)[:,None]
    term1 = np.einsum('ki,kj->kij', term1, posmj)
    ii = -ii - term1
    #STILL NEED TO TURN INTO ENERGY
    
    ki = np.zeros((i.shape[0], 3, 3))
    ki[:,0,1] =  posij[:,2]
    ki[:,0,2] = -posij[:,1]
    ki[:,1,0] = -posij[:,2]
    ki[:,1,2] =  posij[:,0]
    ki[:,2,0] =  posij[:,1]
    ki[:,2,1] = -posij[:,0]
    ki *= (rkj/rmj/rmj)[:,None,None]
    term1 = ((1./rkj/rmj/rmj) - (2.*rkj*rij*rij/rmj**4))[:,None]*poskj
    term1 += 2.*(rkj*np.einsum("ki,ki->k", posij, poskj)/rmj**4)[:,None]*posij
    term1 = np.einsum('ki,kj->kij', term1, posmj)
    ki = -ki - term1
    
    ji = -ki - ii
    
    li = np.zeros((i.shape[0], 3, 3))
    
    ll = np.zeros((i.shape[0], 3, 3))
    ll[:,0,1] =  poskj[:,2]
    ll[:,0,2] = -poskj[:,1]
    ll[:,1,0] = -poskj[:,2]
    ll[:,1,2] =  poskj[:,0]
    ll[:,2,0] =  poskj[:,1]
    ll[:,2,1] = -poskj[:,0]
    ll *= (rkj/rnk/rnk)[:,None,None]
    term1 = (rkj*rkj)[:,None]*poslk - np.einsum("ki,ki->k",poslk,poskj)[:,None]*poskj
    term1 *= -(2.*rkj/rnk**4)[:,None]
    term1 = np.einsum('ki,kj->kij', term1, posnk)
    ll = -ll - term1
    
    jl = np.zeros((i.shape[0], 3, 3))
    jl[:,0,1] =  poslk[:,2]
    jl[:,0,2] = -poslk[:,1]
    jl[:,1,0] = -poslk[:,2]
    jl[:,1,2] =  poslk[:,0]
    jl[:,2,0] =  poslk[:,1]
    jl[:,2,1] = -poslk[:,0]
    jl *= (rkj/rnk/rnk)[:,None,None]
    term1 = ((2.*rkj*rlk*rlk/rnk**4)-(1./rkj/rnk/rnk))[:,None]*poskj
    term1 += 2.*(rkj*np.einsum("ki,ki->k", poslk, -poskj)/rnk**4)[:,None]*poslk
    term1 = np.einsum('ki,kj->kij', term1, posnk)
    jl = -jl - term1
    
    kl = -ll - jl
    
    jj = (np.einsum('ki,ki->k', posij, poskj)/rkj/rkj - 1.)[:,None,None]*ji
    jj += (np.einsum('ki,ki->k', poslk, poskj)/rkj/rkj)[:,None,None]*jl
    term1 = 2.*(np.einsum("ki,ki->k", posij, poskj)/rkj**4)[:,None]*poskj
    term1 += -(1./rkj/rkj)[:,None]*(posij + poskj)
    jj += np.einsum('ki,kj->kij', term1, dwdri)
    term1 = 2.*(np.einsum("ki,ki->k", poslk, poskj)/rkj**4)[:,None]*poskj
    term1 += -(1./rkj/rkj)[:,None]*poslk
    jj += np.einsum('ki,kj->kij', term1, dwdrl)
    
    u2 = -.5*(  vn[:,0]*np.sin(np.radians(   omega - gn[:,0]))
           + 2.*vn[:,1]*np.sin(np.radians(2.*omega - gn[:,1]))
           + 3.*vn[:,2]*np.sin(np.radians(3.*omega - gn[:,2]))
           + 4.*vn[:,3]*np.sin(np.radians(4.*omega - gn[:,3])))
    
    u1 = -.5*(   vn[:,0]*np.cos(np.radians(   omega - gn[:,0]))
           +  4.*vn[:,1]*np.cos(np.radians(2.*omega - gn[:,1]))
           +  9.*vn[:,2]*np.cos(np.radians(3.*omega - gn[:,2]))
           + 16.*vn[:,3]*np.cos(np.radians(4.*omega - gn[:,3])))
    
    ii = u2[:,None,None]*ii + u1[:,None,None]*np.einsum('ki,kj->kij', dwdri, dwdri)
    ji = u2[:,None,None]*ji + u1[:,None,None]*np.einsum('ki,kj->kij', dwdrj, dwdri)
    ki = u2[:,None,None]*ki + u1[:,None,None]*np.einsum('ki,kj->kij', dwdrk, dwdri)
    li = u2[:,None,None]*li + u1[:,None,None]*np.einsum('ki,kj->kij', dwdrl, dwdri)
    ll = u2[:,None,None]*ll + u1[:,None,None]*np.einsum('ki,kj->kij', dwdrl, dwdrl)
    kl = u2[:,None,None]*kl + u1[:,None,None]*np.einsum('ki,kj->kij', dwdrk, dwdrl)
    jl = u2[:,None,None]*jl + u1[:,None,None]*np.einsum('ki,kj->kij', dwdrj, dwdrl)
    jj = u2[:,None,None]*jj + u1[:,None,None]*np.einsum('ki,kj->kij', dwdrj, dwdrj)
    
    
    np.add.at(hess, (pos.shape[0]+1)*i, ii)
    np.add.at(hess, (pos.shape[0]+1)*j, jj)
    np.add.at(hess, (pos.shape[0]+1)*k, -ki - kl + np.transpose(ji+jl, (0, 2, 1)) + jj)
    np.add.at(hess, (pos.shape[0]+1)*l, ll)
    
    # add off-diagonal blocks
    np.add.at(hess, pos.shape[0]*j + i, ji)
    np.add.at(hess, pos.shape[0]*k + i, ki)
    np.add.at(hess, pos.shape[0]*l + i, li)
    np.add.at(hess, pos.shape[0]*k + l, kl)
    np.add.at(hess, pos.shape[0]*j + l, jl)
    np.add.at(hess, pos.shape[0]*i + j, np.transpose(ji, (0, 2, 1)))
    np.add.at(hess, pos.shape[0]*i + k, np.transpose(ki, (0, 2, 1)))
    np.add.at(hess, pos.shape[0]*i + l, np.transpose(li, (0, 2, 1)))
    np.add.at(hess, pos.shape[0]*l + k, np.transpose(kl, (0, 2, 1)))
    np.add.at(hess, pos.shape[0]*l + j, np.transpose(jl, (0, 2, 1)))
    ###
    np.add.at(hess, pos.shape[0]*j + k, -ji - jj -jl)
    np.add.at(hess, pos.shape[0]*k + j, -np.transpose(ji + jl, (0, 2, 1)) - jj)
    
    return np.hstack(np.hstack(hess.reshape(pos.shape[0],pos.shape[0],3,3)))

def hess_lennard_jones(pos, i, j, rvdw0, epvdw):
    """
    Return the Hessian of the Lennard-Jones interaction.
    """
    # create stack of 3x3 blocks that will compose the hessian
    hess = np.zeros((pos.shape[0]**2,3,3))
    posij = pos[i] - pos[j]
    rij = np.linalg.norm(posij, axis=1)
    ep = np.sqrt(epvdw[i]*epvdw[j])
    r0 = rvdw0[i] + rvdw0[j]
    # create stack of subblocks where each layer is the outerproduct of pos diffs
    block  = (24.*ep*(r0**6/rij**10))[:,None,None]*np.einsum('ki,kj->kij', posij, posij)
    block  = (4. - 7.*(r0**6/rij**6))[:,None,None]*block
    block += -(12.*ep*(r0**6/rij**8)*(1. - (r0**6/rij**6)))[:,None,None]*np.tile(np.eye(3), (i.shape[0],1,1))
    # add subblocks to the hessian
    np.add.at(hess, (pos.shape[0]+1)*i, -block)
    np.add.at(hess, (pos.shape[0]+1)*j, -block)
    # now add off diagonal blocks
    np.add.at(hess, pos.shape[0]*i + j,  block)
    np.add.at(hess, pos.shape[0]*j + i,  block)
    return np.hstack(np.hstack(hess.reshape(pos.shape[0],pos.shape[0],3,3)))

def hess_constrain(pos, i, j, kc):
    """
    Return the Hessian of the -k_c log r_ij interaction
    """
    # create stack of 3x3 blocks that will compose the hessian
    hess = np.zeros((pos.shape[0]**2,3,3))
    posij = pos[i] - pos[j]
    rij = np.linalg.norm(posij, axis=1)
    # create stack of subblocks where each layer is the outerproduct of pos diffs
    block  = (2./rij**4)[:,None,None]*np.einsum('ki,kj->kij', posij, posij)
    block += (1./rij**2)[:,None,None]*np.tile(np.eye(3), (i.shape[0],1,1))
    block  = kc[:,None,None]*block
    # add subblocks to the hessian
    np.add.at(hess, (pos.shape[0]+1)*i, -block)
    np.add.at(hess, (pos.shape[0]+1)*j, -block)
    # now add off diagonal blocks
    np.add.at(hess, pos.shape[0]*i + j,  block)
    np.add.at(hess, pos.shape[0]*j + i,  block)
    return np.hstack(np.hstack(hess.reshape(pos.shape[0],pos.shape[0],3,3)))
    
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
        