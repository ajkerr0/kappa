# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:54:05 2016

@author: alex
"""
import numpy as np

def main():
    
    posList = generate_benzene_sites()
    
    nList,zList = get_neighbors()
    
    return posList, nList, zList

        
def generate_benzene_sites():
    """Generate the locations of a benzene ring, one without hydrogen for attachment to something else."""
    
    #starting distance
    a = 1.45
    angle = 120 #degrees
    angle = angle*np.pi/180.0/2.0
    
    C1pos = np.array([0.,0.,0.])
    C2pos = C1pos + np.array([a*np.cos(angle),a*np.sin(angle),0.])
    C3pos = C2pos + np.array([a,0.,0.])
    C4pos = C1pos + np.array([a*np.cos(angle),-a*np.sin(angle),0.])
    C5pos = C4pos + np.array([a,0.,0.])
    C6pos = C5pos + np.array([a*np.cos(angle),a*np.sin(angle),0.])
    
    a = 1.15
    
    H2pos = C2pos + np.array([-a*np.cos(angle),a*np.sin(angle),0.])
    H3pos = C3pos + np.array([a*np.cos(angle),a*np.sin(angle),0.])
#    H6pos = C6pos + np.array([a,0.,0.])
    H5pos = C5pos + np.array([a*np.cos(angle),-a*np.sin(angle),0.])
    H4pos = C4pos + np.array([-a*np.cos(angle),-a*np.sin(angle),0.])
    
    CList = [C1pos,C2pos,C3pos,C4pos,C5pos,C6pos]
    HList = [H2pos,H3pos,H4pos,H5pos]
    
    CList.extend(HList)    
    
    return CList
    
def get_neighbors():
    
    nList = []
    
    #six carbons, four hydrogens
    nList.append([1,3])
    nList.append([0,2,6])
    nList.append([1,5,7])
    nList.append([0,4,8])
    nList.append([3,5,9])
    nList.append([2,4])
    
    nList.append([1])
    nList.append([2])
    nList.append([3])
    nList.append([4])
    
    zList = np.concatenate((np.full(6,6, dtype=int), np.full(4,1, dtype=int)))
    
    return nList, zList
    
    