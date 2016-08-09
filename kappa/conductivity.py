# -*- coding: utf-8 -*-
"""

@author: Alex Kerr

"""

import itertools

import numpy as np
import scipy.linalg as linalg

amuDict = {1:1.008, 6:12.01, 7:14.01, 8:16.00, 9:19.00,
           15:30.79, 16:32.065, 17:35.45}

class Calculation:
    
    def __init__(self, base):
        if len(base.faces) == 2:
            self.base = base
        else:
            raise ValueError("A base molecule with 2 interfaces is needed!")
        self.trialList = []
            
    def add(self, molList, indexList):
        """Append a trial molecule to self.trialList with enhancements 
        from molList attached to atoms in indexList"""
        
        from .operation import _combine
        for mol, index in zip(molList,indexList):
            newTrial = _combine(self.base, mol, index, 0, copy=True)
        newTrial._configure()
        self.trialList.append(newTrial)
        return newTrial
        
def calculate_thermal_conductivity(mol, d1, d2):
    
    gamma = 0.1
    
    #driven atoms
    drive1, drive2 = d1, d2
#    drive1, drive2 = 1,2
    
    from .operation import hessian
    kMatrix = hessian(mol)
#    kMatrix = _calculate_fake_K_matrix(len(mol), 1., mol.nList)
    
    gMatrix = _calculate_gamma_mat(len(mol),gamma, drive1, drive2)
    
    mMatrix = _calculate_mass_mat(mol.zList)
    
    val, vec = _calculate_thermal_evec(kMatrix, gMatrix, mMatrix)
    
    coeff = _calculate_coeff(val, vec, mMatrix, gMatrix)
    
    
    def _calculate_power(i,j):
        
        #driven atom
        #assuming same drag constant as other driven atom
        driver = drive1
        
        n = len(val)
        
        kappa = 0.
        
        val_sigma = np.tile(val, (n,1))
        val_tau = np.transpose(val_sigma)
        
        for idim in [0,1,2]:
            for jdim in [0,1,2]:
        
                term1 = np.tile(coeff[:, 3*driver], (n,1)) + np.tile(coeff[:, 3*driver + 1], (n,1)) \
                        + np.tile(coeff[:, 3*driver + 2], (n,1))
                term3 = np.tile(vec[3*i + idim,:], (n,1))
                term4 = np.transpose(np.tile(vec[3*j + jdim,:], (n,1))) 
        
                kappa += kMatrix[3*i + idim, 3*j + jdim]*np.sum(term1*term3*term4*((val_sigma-val_tau)/(val_sigma+val_tau)))
                
        return kappa
    
    #for each interaction that goes through the interface,
    #add it to the running total kappa
    
    #determine interfaces
    for count,face in enumerate(mol.faces):
        if drive1 in face.atoms:
            face1 = count
        if drive2 in face.atoms:
            face2 = count
            
    #determine interface path
    for face in mol.faces:
        #look for a path that contains both indices
        if face1 in face.path and face2 in face.path:
            path = face.path
        else:
            #the interfaces are on the 'root' mol
            pass
            #goig to assume every molecule i test uses meet this condition, for now
            
    #find all face paths that start at our interfaces
    face1paths, face2paths = [], []
    for face in mol.faces:
        if face1 in face.path:
            face1paths.append(face.path)
        if face2 in face.path:
            face2paths.append(face.path)
            
    #find all facetracking numbers used
    tracknums1, tracknums2 = [], []
    for path in face1paths:
        tracknums1.append(path[-1])
    for path in face2paths:
        tracknums2.append(path[-1])
            
    #find all dihedral interactions that contain an enhancement atom and interface atom
    #add them to pairings list
    interactions = []
#    atoms = np.where(mol.facetrack in tracknums)
#    print('check2')
    atoms1 = [i for i in range(len(mol)) if mol.facetrack[i] in tracknums1]
    atoms2 = [i for i in range(len(mol)) if mol.facetrack[i] in tracknums2]
#    print(tracknums)
#    print(atoms)
#    print('check3')
    for dih in mol.dihList:
        for atom in atoms1:
            if atom in dih:
                #find the elements of facetrack -1 that are also in dih
                #if there are any, then add them to interactions
                elements = [x for x in dih if mol.facetrack[x] == -1]
                for element in elements:
                    interactions.append([atom, element])
        for atom in atoms2:
            if atom in dih:
                elements = [x for x in dih if mol.facetrack[x] == -1]
                for element in elements:
                    interactions.append([element, atom])
                    
    #add nonbonded interactions
                    
    #remove duplicate interactions
    interactions.sort()
    interactions = list(k for k,_ in itertools.groupby(interactions))    
                    
    print('check4')
    print(interactions)
                
    kappa = 0.             
                
    for interaction in interactions:
        i,j = interaction
        kappa += _calculate_power(i,j)
    
    print(kappa)
    
def _calculate_coeff(val, vec, massMat, gMat):
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
    
def _calculate_thermal_evec(K,G,M):
    
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
    
def _calculate_mass_mat(zList):
    
    massList = []
    
    for z in zList:
        massList.append(amuDict[z])
        
    diagonal = np.repeat(np.array(massList), 3)
    
    return np.diag(diagonal)
    
def _calculate_gamma_mat(N,gamma, drive1, drive2):
    
    gmat = np.zeros((3*N, 3*N))
    driveList = [drive1, drive2]
    
    for drive_atom in driveList:
        gmat[3*drive_atom  , 3*drive_atom  ] = gamma
        gmat[3*drive_atom+1, 3*drive_atom+1] = gamma
        gmat[3*drive_atom+2, 3*drive_atom+2] = gamma
        
    return gmat
    
def _calculate_fake_K_matrix(N,k0,nLists):
    """Return the Hessian of a linear chain of atoms assuming only nearest neighbor interactions."""
    
    KMatrix = np.zeros([3*N,3*N])
    
    for i,nList in enumerate(nLists):
        KMatrix[3*i  ,3*i  ] = k0*len(nList)
        KMatrix[3*i+1,3*i+1] = k0*len(nList)
        KMatrix[3*i+2,3*i+2] = k0*len(nList)
        for neighbor in nList:
            KMatrix[3*i  ,3*neighbor] = -k0
            KMatrix[3*i+1,3*neighbor+1] = -k0
            KMatrix[3*i+2,3*neighbor+2] = -k0
    
    return KMatrix