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
        newTrial.name = "%s_trial%s" % (newTrial.name, str(self.trialcount))
        self.trialcount += 1
        return newTrial
        
    def calculate_kappa(self, trial):
        from .plot import bonds
        bonds(self.trialList[trial])
        calculate_thermal_conductivity(self.trialList[trial], self.driverList[trial], len(self.base))
        
def calculate_thermal_conductivity(mol, driverList, baseSize):
    
    #give each driver the same drag constant
    gamma = 0.1
    
    #standardize the driverList
    driverList = np.array(driverList)
    
    from .operation import _calculate_hessian
    kMatrix = _calculate_hessian(mol)
    ballandspring=False
#    print(kMatrix)
#    kMatrix = hessian(mol)
#    kMatrix, ballandspring = _calculate_ballandspring_k_mat(len(mol), 1., mol.nList), True
    
    gMatrix = _calculate_gamma_mat(len(mol), gamma, driverList)
    
    mMatrix = _calculate_mass_mat(mol.zList)
    
    val, vec = _calculate_thermal_evec(kMatrix, gMatrix, mMatrix)
    
#    np.set_printoptions(threshold=np.inf)
#    print(val)
    
    coeff = _calculate_coeff(val, vec, mMatrix, gMatrix)
    
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
    crossings = list(k for k,_ in itertools.groupby(crossings))    
                
    kappa = 0.
    
    print(crossings)
    
    mullenTable = []
    
    print_spring_constants(crossings, kMatrix)
                
    for crossing in crossings:
        i,j = crossing
        kappa += _calculate_power(i,j,val, vec, coeff, kMatrix, driverList, mullenTable)
#        kappa += _calculate_power_loop(i,j,val, vec, coeff, kMatrix, driverList, mullenTable)
    
#    pprint.pprint(mullenTable)
    print(kappa)
    return kappa
    
def _calculate_power_loop(i,j, val, vec, coeff, kMatrix, driverList, mullenTable):
    
    driver1 = driverList[1]    
    
    n = len(val)//2
    
    kappa = 0.
    
    for idim in [0,1,2]:
        for jdim in [0,1,2]:
            for driver in driver1:
                term = 0.
                for sigma in range(2*n):
                    cosigma = coeff[sigma, 3*driver + 1] + coeff[sigma, 3*driver +2] + coeff[sigma, 3*driver]
                    for tau in range(2*n):
                        cotau = coeff[tau, 3*driver] + coeff[tau, 3*driver+1] + coeff[tau, 3*driver+2]
                        try:
                            term += cosigma*cotau*(vec[:n,:][3*i + idim ,sigma])*(vec[:n,:][3*j + jdim,tau])*((val[sigma]-val[tau])/(val[sigma]+val[tau]))
                        except FloatingPointError:
                            print("Divergent term")
                            print(cosigma*cotau*(vec[:n,:][3*i + idim ,sigma])*(vec[:n,:][3*j + jdim,tau]))
                            print(tau)
                            print(sigma)
                            print(val[tau])
                            print(val[sigma])
                term *= kMatrix[3*i + idim, 3*j + jdim]
                mullenTable.append([3*i+idim,3*j+jdim,kMatrix[3*i + idim, 3*j + jdim],term])
                kappa += term
            
    return kappa
    
def _calculate_power2(i, j, val, vec, coeff, kMatrix, driverList, mullenTable):
    
    #assuming same drag constant as other driven atom
    driver1 = driverList[1]
    d = len(driver1)
    
    n = len(val)
    
    kappaterms = np.zeros((3,3,d,n,n))
    
    val_sigma = np.tile(val, (n,1))
    val_tau = np.transpose(val_sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        valterm = np.true_divide(val_sigma-val_tau,val_sigma+val_tau)
    valterm[~np.isfinite(valterm)] = 0.
    
    aix = vec[3*i  ,:]
    aiy = vec[3*i+1,:]
    aiz = vec[3*i+2,:]
    ajx = vec[3*j  ,:]
    ajy = vec[3*j+1,:]
    ajz = vec[3*j+2,:]
    
    return 0.
    
def _calculate_power(i,j, val, vec, coeff, kMatrix, driverList, mullenTable):
    
    #assuming same drag constant as other driven atom
    driver1 = driverList[1]
    
    n = len(val)
    
    kappa = 0.
    
    val_sigma = np.tile(val, (n,1))
    val_tau = np.transpose(val_sigma)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        valterm = np.true_divide(val_sigma-val_tau,val_sigma+val_tau)
    valterm[~np.isfinite(valterm)] = 0.
    
    for idim in [0,1,2]:
        for jdim in [0,1,2]:
            
            term3 = np.tile(vec[3*i + idim,:], (n,1))
            term4 = np.transpose(np.tile(vec[3*j + jdim,:], (n,1)))
            
            for driver in driver1:
    
                term1 = np.tile(coeff[:, 3*driver] + coeff[:, 3*driver+1] + coeff[:, 3*driver+2], (n,1))
                term2 = np.transpose(term1)
                
                term = kMatrix[3*i + idim, 3*j + jdim]*np.sum(term1*term2*term3*term4*valterm)

                mullenTable.append([3*i+idim,3*j+jdim,kMatrix[3*i + idim, 3*j + jdim],term])
                kappa += term          
                
    return kappa
    
def print_spring_constants(interactions, kmat):
    
    acts = np.array(interactions)
    
    for act in acts:
        i,j = act
        print(act)
        print(kmat[3*i:3*i+3,3*j:3*j+3])
    
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
    
def write_to_txt(twodlist, name):
    with open(name,'w') as f:
        for x in zip(*twodlist):
            f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(*x))