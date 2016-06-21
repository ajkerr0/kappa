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
ds = 1e-7
dx = ds
dy = ds
dz = ds

vdx = np.array([dx,0.0,0.0])
vdy = np.array([0.0,dy,0.0])
vdz = np.array([0.0,0.0,dz])

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
    path = "./save_file/" + molecule.name
    _path_exists(path)
    pickle.dump(molecule, open( path + "/mol.p", "wb" ) )
    
def load(name):
    """Load a pickled molecule given a name"""
    return pickle.load(open("./save_file/"+name+"/mol.p", "rb"))
    
def _calculate_hessian(molecule):
    """Return the Hessian matrix for the given molecule after calculation."""
    
    N = len(molecule)
    
    H = np.zeros([3*N,3*N])
    
    calculate_energy = molecule.ff.define_energy_routine(molecule)
    
    for i in range(N):
        
        ipos = molecule.posList[i]
        
        ipos += vdx
        _,plusXTestGrad,_,_ = calculate_energy()
        ipos += -2.0*vdx
        _,minusXTestGrad,_,_ = calculate_energy()
        ipos += vdx + vdy
        _,plusYTestGrad,_,_ = calculate_energy()
        ipos += -2.0*vdy
        _,minusYTestGrad,_,_ = calculate_energy()
        ipos += vdy + vdz
        _,plusZTestGrad,_,_ = calculate_energy()
        ipos += -2.0*vdz
        _,minusZTestGrad,_,_ = calculate_energy()
        ipos += vdz
        
        xiRow = (plusXTestGrad - minusXTestGrad)/2.0/dx
        yiRow = (plusYTestGrad - minusYTestGrad)/2.0/dy
        ziRow = (plusZTestGrad - minusZTestGrad)/2.0/dz
        
        H[3*i    ] = np.hstack(xiRow)
        H[3*i + 1] = np.hstack(yiRow)
        H[3*i + 2] = np.hstack(ziRow)
        
    return H
    
def hessian(molecule):
    """Return the Hessian for a molecule; if it doesn't exist calculate it otherwise load it
    from the numpy format."""
    path = "./save_file/" + molecule.name + "/hessian.npy"
    if _file_exists("./save_file/" + molecule.name + "/hessian.npy"):
        print("Loading Hessian matrix from file...")
        return np.load(path)
    else:
        print("Calculating the Hessian matrix for " + molecule.name + "...")
        H = _calculate_hessian(molecule)
        print("Done!")
        np.save(path, H)
        return H
    
def evecs(hessian):
    """Return the eigenvalues and eigenvectors associated with a given Hessian matrix."""
    w,vr = np.linalg.eig(hessian)   
    return w,vr
    
def _combine(oldMolecule1,oldMolecule2,index1,index2, nextIndex1):
    """Return a single molecule which is the combination of 2 inputed molcules where indices
    1 and 2 are the same atom effectively."""
    
    molecule1 = deepcopy(oldMolecule1)
    molecule2 = deepcopy(oldMolecule2)
    
    size1 = len(molecule1)
    
    #anticipate new index1
    if nextIndex1 == index2:
        nextIndex1 = index1
    elif nextIndex1 > index2:
        nextIndex1 = nextIndex1 + size1 - 1
    else:
        nextIndex1 = nextIndex1 + size1
    
    #rotate molecule2
    axis = np.cross(molecule2.orientation, molecule1.orientation)
    mag = np.linalg.norm(axis)
    if mag > 1e-10:
        axis = axis/mag
        angle = np.degrees(np.arcsin(mag))
        molecule2.rotate(axis,angle)
    
    #shift molecule 2 into position
    pos1, pos2 = molecule1.posList, molecule2.posList
    z1, z2 = molecule1.zList, molecule2.zList
    displaceVec = pos1[index1] - pos2[index2]
    molecule2.translate(displaceVec)
    
    #adjust for new molecules
    
    #adjust neighbor lists
    #add neighbors to atom1
    for neighbor in molecule2.nList[index2]:
        if neighbor > index2:
            molecule1.nList[index1].append(neighbor + size1 - 1)
        elif neighbor < index2:
            molecule1.nList[index1].append(neighbor + size1)
        else:
            print("An atom can't neighbor itself!")
    for index,nList in enumerate(molecule2.nList):
        if index != index2:
            newNList = []
            for neighbor in nList:
                if neighbor == index2:
                    newNList.append(index1)
                elif neighbor > index2:
                    newNList.append(neighbor + size1 - 1)
                else:
                    newNList.append(neighbor + size1)
            molecule2.nList[index] = newNList
            
    #add atoms
    pos2 = np.delete(pos2, index2, 0)
    z2 = np.delete(z2, index2, 0)
    del molecule2.nList[index2]
    
    pos1 = np.concatenate((pos1,pos2), axis=0)
    z1 = np.append(z1,z2)
    molecule1.nList.extend(molecule2.nList)
    
    molecule1.posList = pos1
    molecule1.zList = z1
    
    return molecule1, nextIndex1
    
def chain(molList, indexList, name=""):
    """Return a molecule as a chain of inputted molecules."""
    
    #check validity of args
    ##
    
    molChain = molList[0]
    i,j = indexList[0]
    
    for molNum, mol in enumerate(molList[1:]):
        
        if molNum+1 < len(indexList):
            nextI = indexList[molNum+1][0]
        else:
            nextI = 0
        j = indexList[molNum][1]
        molChain, i = _combine(molChain, mol, i, j, nextI)
        
    if not name:
        name = molList[0].name
    molChain.name = name
        
    return molChain
    
    
        
        
        