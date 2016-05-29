# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:57:04 2016

@author: alex
"""

import numpy as np
pos = []
bondList = []
angleList = []
dihList = []

kbList, l0List = [],[]
kaList, t0List = [],[]
vnList, nnList, gnList = [], [], []

e = 0

def _bonds(whereBonds):
    bonds = bondList[whereBonds]
    kb,l0 = kbList[whereBonds], l0List[whereBonds]
    i,j = bonds[:,0], bonds[:1]
    rij = pos[i] - pos[j]
    rij = np.linalg.norm(rij, axis=1)
    eList = kb*(rij - l0)*(rij - l0)
    return np.sum(eList)
    
def _add_bonds(energy_func):
    
    def bond_wrapper():
        energy_func()
        _bonds(whereBonds)
        
    return bond_wrapper
    
def _add_angles(energy_func):
    
    def angle_wrapper():
        energy_func()
        _angles(whereAngles)
        
    return angle_wrapper()
    
def _add_dihedrals(energy_func):
    
    def dihedral_wrapper():
        energy_func()
        _dihedrals(whereDihs)
        
    return dihedral_wrapper()
    
def _angles(whereAngles):
    angles = angleList[whereAngles]
    ka,t0 = kaList[whereAngles], t0List[whereAngles]
    i,j,k = angles[:,0],angles[:,1],angles[:,2]
    posij = pos[i] - pos[j]
    poskj = pos[k] - pos[j]
    rij = np.linalg.norm(posij,axis=1)
    rkj = np.linalg.norm(poskj,axis=1)
    cosTheta = np.einsum('ij,ij->i',posij,poskj)/rij/rkj
    theta = np.degrees(np.arccos(cosTheta))
    eList = ka*(theta - t0)*(theta - t0)
    return np.sum(eList)
    
def _dihedrals(whereDihs):
    dihs = dihList[whereDihs]
    vn,nn,gn = vnList[whereDihs],nnList[whereDihs],gnList[whereDihs]
    i,j,k,l = dihs[:,0],dihs[:,1],dihs[:,2],dihs[:3]
    posji = pos[j] - pos[i]
    poskj = pos[k] - pos[j]
    poslk = pos[l] - pos[k]
    rkj = np.linalg.norm(poskj,axis=1)
    cross12 = np.cross(posji,poskj)
    cross23 = np.cross(poskj,poslk)
    
    norm = np.einsum('ij,ij->i', cross12,cross12)
    