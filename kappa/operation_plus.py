# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:52:47 2016

@author: Alex
"""

from copy import deepcopy

import numpy as np

def _combine(mol1, mol2, index1, index2, copy=True, nextIndex1=None):
    """Return a single molecule which is the combination of input molecules.  If nextIndex1 is not
    None, also return the next index1 in the chain process in a tuple.
    
    Args:
        mol1 (Molecule): Base molecule to be combined whose indexing will be
            carried over into the new molecule.
        mol2 (Molecule): Molecule to be effectively attached to mol1.
        index1 (int): The atomic index of mol1 in which mol2 will be attached.
        index2 (int): The atomic index of mol2 that will become index1 of the new molecule.
        
    Keywords:
        copy (bool): True if the new molecule is to be a new Molecule instance, False if
            the new olecule is to be an altered mol1.  Default is True.
        nextIndex1 (int): For use in chain function.  Used to find the next index1 in chaining
            situations in which the next index is not yet determined.  If None only a new molecule
            is returned."""
            
    #check validity of parameters
    if mol1.zList[index1] != mol2.zList[index2]:
        raise ValueError("Molecules must be joined at atoms of the same atomic number")
    
    #create new instance if copy is true        
    if copy is True:
        mol1 = deepcopy(mol1)
    mol2 = deepcopy(mol2)
    
    #regularly referenced quantities
    size1 = len(mol1)
    pos1, pos2 = mol1.posList, mol2.posList
    z1, z2 = mol1.zList, mol2.zList
            
    #find faces of indices
    for count, face in enumerate(mol1.faces):
        if index1 in face.atoms:
            face1 = count
            norm1 = face.norm
    for count, face in enumerate(mol2.faces):
        if index2 in face.atoms:
            face2 = count
            norm2 = face.norm
            
    #change position of mol2 to put it in place
    #rotate mol2
    axis = np.cross(norm1, norm2)
    mag = np.linalg.norm(axis)
    if mag < 1e-10:
        #check for parallel/anti-parallel
        dot = np.dot(norm1, norm2)
        if dot > 0.:
            #flip the molecule
            mol2.invert()
    else:
        angle = np.degrees(np.arcsin(mag))
        mol2.rotate(axis,angle)
    #shift mol2 into position via translation
    mol2.translate(pos1[index1] - pos2[index2])
    
    #adjust molecule attributes
    #neighbors
    for neighbor in mol2.nList[index2]:
        if neighbor > index2:
            mol1.nList[index1].append(neighbor + size1 - 1)
        elif neighbor < index2:
            mol1.nList[index1].append(neighbor + size1)
        else:
            raise ValueError("An atom can't neighbor itself")
    for index, nList in enumerate(mol2.nList):
        if index != index2:
            newNList = []
            for neighbor in nList:
                if neighbor == index2:
                    newNList.append(index1)
                elif neighbor > index2:
                    newNList.append(neighbor + size1 - 1)
                else:
                    newNList.append(neighbor + size1)
            mol2.nList[index] = newNList
            
    #delete single atom interfaces
    if len(face1.atoms) == 1:
        del mol1.faces[face1]
    if len(face2.atoms) == 1:
        del mol2.faces[face2]
    
    #complete new molecule
    #delete the merging atom of mol2
    pos2 = np.delete(pos2, index2, 0)
    z2 = np.delete(z2, index2, 0)
    del mol2.nList[index2]
    #add atoms to mol1
    mol1.posList = np.concatenate((pos1,pos2), axis=0)
    mol1.zList = np.concatenate((z1,z2), axis=0)
    mol1.nList.extend(mol2.nList)
    
    #anticipate new index1
    if nextIndex1 is not None:
        if nextIndex1 == index2:
            nextIndex1 = index1
        elif nextIndex1 > index2:
            nextIndex1 = nextIndex1 + size1 - 1
        else:
            nextIndex1 = nextIndex1 + size1
        return mol1, nextIndex1
    
    return mol1
    
    