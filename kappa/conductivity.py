# -*- coding: utf-8 -*-
"""

@author: Alex Kerr

"""

import itertools
from copy import deepcopy

import numpy as np
import scipy.linalg as linalg

amuDict = {1:1.008, 6:12.01, 7:14.01, 8:16.00, 9:19.00,
           15:30.79, 16:32.065, 17:35.45}
           
class Calculation:
    
    def __init__(self, base, n, descent="cg", search="backtrack", numgrad=False,
                 eprec=1e-2, fprec=1e-1, efreq=10, nbnfreq=15):
        if len(base.faces) == 2:
            self.base = base
        else:
            raise ValueError("A base molecule with 2 interfaces is needed!")
        #minimize the base molecule
        from ._minimize import minimize
        minimize(self.base, n, descent, search, numgrad, eprec, fprec, efreq, nbnfreq)
        #assign minimization attributes
        self.n = n
        self.descent = descent
        self.search = search
        self.numgrad = numgrad
        self.eprec = eprec
        self.fprec = fprec
        self.efreq = efreq
        self.nbnfreq = nbnfreq
        #other attributes
        self.trialcount = 0
        self.trialList = []
        self.driverList = []
            
    def add(self, molList, indexList):
        """Append a trial molecule to self.trialList with enhancements 
        from molList attached to atoms in indexList"""
        
        from .molecule import _combine
        newTrial = deepcopy(self.base)
        dList = [[],[]]
        for mol, index in zip(molList,indexList):
            #find faces of index
            for count, face in enumerate(self.base.faces):
                if index in face.atoms:
                    face1 = count
            sizetrial = len(newTrial)
            newTrial = _combine(newTrial, mol, index, 0, copy=False)
            #do minus 2 because 1 atom gets lost in combination
            #and another to account to the 'start at zero' indexing
            dList[face1].append(mol.driver + sizetrial - 1)
        newTrial._configure()
        self.driverList.append(dList)
        self.trialList.append(newTrial)
        from ._minimize import minimize
        minimize(newTrial, self.n, self.descent, self.search, self.numgrad,
                 self.eprec, self.fprec, self.efreq, self.nbnfreq)
        from .plot import bonds
        newTrial.name = "%s_trial%s" % (newTrial.name, str(self.trialcount))
        self.trialcount += 1
        bonds(newTrial)
        return newTrial
        
    def calculate_kappa(self, trial):
        calculate_thermal_conductivity(self.trialList[trial], self.driverList[trial], len(self.base))
        
def calculate_thermal_conductivity(mol, driverList, baseSize):
    
    #give each driver the same drag constant
    gamma = 0.1
    
    #standardize the driverList
    driverList = np.array(driverList)
    
    from .operation import hessian, _calculate_hessian
#    kMatrix = _calculate_hessian(mol)
#    print(kMatrix)
#    kMatrix = hessian(mol)
    kMatrix = _calculate_ballandspring_k_mat(len(mol), 1., mol.nList)
    ballandspring = True
    print(kMatrix)
    
    gMatrix = _calculate_gamma_mat(len(mol), gamma, driverList)
    
    mMatrix = _calculate_mass_mat(mol.zList)
    
    val, vec = _calculate_thermal_evec(kMatrix, gMatrix, mMatrix)
    
    coeff = _calculate_coeff(val, vec, mMatrix, gMatrix)
    
    def _calculate_power(i,j):
        
        #assuming same drag constant as other driven atom
#        driver0 = driverList[0]
        driver1 = driverList[1]
        
        n = len(val)
        
        kappa = 0.
        
        val_sigma = np.tile(val, (n,1))
        val_tau = np.transpose(val_sigma)
        
        for idim in [0,1,2]:
            for jdim in [0,1,2]:
                
                term3 = np.tile(vec[3*i + idim,:], (n,1))
#                print(np.shape(term3))
                term4 = np.transpose(np.tile(vec[3*j + jdim,:], (n,1)))
                
#                print("term3 %s" % term3)
#                print("term4 %s" % term4)
#                print(np.amax(term4))
                for driver in driver1:
        
                    term1 = np.tile(coeff[:, 3*driver], (n,1)) + np.tile(coeff[:, 3*driver + 1], (n,1)) \
                            + np.tile(coeff[:, 3*driver + 2], (n,1)) 
                    
                    term = kMatrix[3*i + idim, 3*j + jdim]*np.sum(term1*term3*term4*((val_sigma-val_tau)/(val_sigma+val_tau)))
#                    print(term)
                    kappa += term          
                    
        return kappa
    
    #for each interaction that goes through the interface,
    #add it to the running total kappa
            
    #find all dihedral interactions that contain an enhancement atom and interface atom
    #dihedral interactions exhaust all possible bond related interactions although if dihedrals are turned off 
            #many terms will be zero
    #add them to pairings list
    crossings = []
    atoms0 = mol.faces[0].attached
    atoms1 = mol.faces[1].attached
    
    if mol.ff.dihs:
        interactions = mol.dihList
    elif mol.ff.angles:
        interactions = mol.angleList
    elif mol.ff.lengths:
        interactions = mol.bondList
        
    if ballandspring:
        interactions = mol.bondList

    for it in interactions:
        for atom in atoms0:
            if atom in it:
                #find elements that are part of the base molecule
                #if there are any, then add them to interactions
                elements = [x for x in it if x < baseSize]
                for element in elements:
                    crossings.append([atom, element])
        for atom in atoms1:
            if atom in it:
                elements = [x for x in it if x < baseSize]
                for element in elements:
                    crossings.append([element, atom])
                    
    #add nonbonded interactions
                    
    #remove duplicate interactions
    crossings.sort()
    crossings = list(k for k,_ in itertools.groupby(interactions))    
                
    kappa = 0.      
                
    for crossing in crossings:
        i,j = crossing
        kappa += _calculate_power(i,j)
    
    print(kappa)
    return kappa
    
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
    
def _calculate_gamma_mat(N,gamma, driverList):
    
    gmat = np.zeros((3*N, 3*N))
    driveList = np.hstack(driverList)
    
    for drive_atom in driveList:
        gmat[3*drive_atom  , 3*drive_atom  ] = gamma
        gmat[3*drive_atom+1, 3*drive_atom+1] = gamma
        gmat[3*drive_atom+2, 3*drive_atom+2] = gamma
        
    return gmat
    
def _calculate_ballandspring_k_mat(N,k0,nLists):
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