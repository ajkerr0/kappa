# -*- coding: utf-8 -*-
"""

@author: Alex Kerr

"""

import itertools
from copy import deepcopy
import csv
import time

import numpy as np
import matplotlib.pyplot as plt

from .molecule import build, chains
from .operation import _calculate_hessian
import ballnspring

amuDict = {1:1.008, 6:12.01, 7:14.01, 8:16.00, 9:19.00,
           15:30.79, 16:32.065, 17:35.45}
           
stapled_index = 30
           
class Calculation:
    
    def __init__(self, base, gamma=10., **minkwargs):
        if len(base.faces) == 2:
            self.base = base
        else:
            raise ValueError("A base molecule with 2 interfaces is needed!")
        self.gamma = gamma
        #minimize the base molecule
        from ._minimize import minimize
        minimize(self.base, **minkwargs)
        #assign minimization attributes
        self.minkwargs = minkwargs
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
#            dList[face1].append(mol.driver + sizetrial - 1)
#            #drive every hydrogen
            for hindex in np.where(mol.zList==1)[0]:
                dList[face1].append(hindex + sizetrial - 1)
        newTrial._configure()
        self.driverList.append(dList)
        self.trialList.append(newTrial)
        from ._minimize import minimize
        minimize(newTrial, **self.minkwargs)
        newTrial.name = "%s_trial%s" % (newTrial.name, str(self.trialcount))
        self.trialcount += 1
        return newTrial
        
    def calculate_kappa(self, trial):
        from .plot import bonds
        bonds(self.trialList[trial])
        return calculate_thermal_conductivity(self.trialList[trial], self.driverList[trial], len(self.base), self.gamma)
        
class ParamSpaceExplorer(Calculation):
    
    def __init__(self, base, cnum, clen=[1], cid=["polyeth"], gamma=10., **minkwargs):
        super().__init__(base, gamma=gamma, **minkwargs)
        self.clen = clen
        self.cnum = cnum
        self.cid = cid
        #make zero value array based on dim of parameters
        self.params = [cid, clen, cnum]
        self.values = np.zeros([len(x) for x in self.params])
        
    def explore(self):
        trial = 0
        for idcount, _id in enumerate(self.cid):
            for lencount, _len in enumerate(self.clen):
                chain = build(self.base.ff, _id, count=_len)
                for numcount in range(len(self.cnum)):
                    #find indices of attachment points
                    indices = [index for subindices in self.cnum[0:numcount+1] for index in subindices]
                    self.add([chain]*(numcount+1)*2, indices)
                    kappa = self.calculate_kappa(trial)
                    vals = [kappa.real, chains.index(_id), _len, numcount+1,
                            self.gamma, self.base.ff.name, indices]
                    self.write(self.base.name, vals)
                    self.values[idcount,lencount,numcount] = kappa
                    trial += 1
                    
    @staticmethod               
    def write(filename, vals):
        kappa, cid, clen, cnum, gamma, ff, indices = vals
        with open('{0}'.format(filename), 'a', newline='') as file:
            line_writer = csv.writer(file, delimiter=';')
            line_writer.writerow([kappa, cid, clen, cnum,0,0,0,gamma, ff, indices, time.strftime("%H:%M/%d/%m/%Y")])
            
class ModeInspector(Calculation):
    """A class designed to inspect quantities related to the thermal conductivity
    calculation.  Inherits from Calculation, but is intended to have only a single 
    trial molecule."""
    
    def __init__(self, base, molList, indices, gamma, **minkwargs):
        super().__init__(base, gamma=gamma, **minkwargs)
        super().add(molList, indices)
        self.mol = self.trialList[0]
        self.k = _calculate_hessian(self.mol, stapled_index, numgrad=False)
        self.dim = len(self.k)//len(self.mol.mass)
        
    @property
    def g(self):
        return ballnspring.calculate_gamma_mat(self.dim, len(self.mol), self.gamma, self.driverList[0])
        
    @property
    def m(self):
        return np.diag(np.repeat(self.mol.mass,self.dim))
        
    @property
    def evec(self):
        return ballnspring.calculate_thermal_evec(self.k, self.g, self.m)
        
    @property
    def coeff(self):
        print('coeff')
        val, vec = self.evec
        return ballnspring.calculate_coeff(val, vec, self.m, self.g), val, vec
        
    @property
    def tcond(self):
        
        coeff, val, vec = self.coeff
        
        crossings = find_interface_crossings(self.mol, len(self.base))
        
        kappaList = []
        kappa = 0.
        for crossing in crossings:
            i,j = crossing
            kappa += ballnspring.calculate_power_list(i,j, self.dim, val, vec, coeff, self.k, self.driverList[0], kappaList)
            
        return kappa, kappaList, val, vec
        
    def plot_ppation(self):
        
        kappa, kappaList, val, vec = self.tcond
        
        N = len(self.mol.posList)
        
        vec = vec[:N,:]
        
        num = np.sum((vec**2), axis=0)**2
        den = len(vec)*np.sum(vec**4, axis=0)
        
        fig = plt.figure()        
        
        plt.scatter(val, num/den, c='b')
        
        #plot points corresponding to the highest values
        max_indices = []
        for entry in kappaList:
            #get the sigma, tau indices
            max_indices.extend(entry[1:3])
            
        max_indices = np.array(max_indices)
        
        plt.scatter(val[max_indices], num[max_indices]/den[max_indices], c='y', zorder=-2)
        
        fig.suptitle("Val vs p-ratio")
        
        plt.axis([-.1,.1, 0., 1.])        
        plt.show()
        
def find_interface_crossings(mol, baseSize):
    """Return the interfactions that cross the molecular interfaces."""
    
    crossings = []
    atoms0 = mol.faces[0].attached
    atoms1 = mol.faces[1].attached
    
    if mol.ff.dihs:
        interactions = mol.dihList
    elif mol.ff.angles:
        interactions = mol.angleList
    elif mol.ff.lengths:
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
    ''' to 
        be 
        completed '''
                    
    #remove duplicate interactions
    crossings.sort()
    crossings = list(k for k,_ in itertools.groupby(crossings))
    print(crossings)
    
    return crossings

def calculate_thermal_conductivity(mol, driverList, baseSize, gamma):
    
    crossings = find_interface_crossings(mol, baseSize)
    
    kmat = _calculate_hessian(mol, stapled_index, numgrad=False)
    
    return ballnspring.kappa(mol.mass, kmat, driverList, crossings, gamma)
        