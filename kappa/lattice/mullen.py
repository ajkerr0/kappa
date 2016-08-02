# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:47:41 2016

@author: Alex Kerr
"""

import numpy as np

def main():
    
    posList = generate_sites()
    nList,zList = get_neighbors()
    
    return posList, nList, zList
    
def generate_sites():
    """Return the locations of the carbon structure in Mullen's structure."""
    
    #starting distance
    a = 1.40
    
    a1 = a*np.array([1.,0.,0.])
    a2 = a*np.array([-.5,np.sqrt(3.)*.5,0.])
    a3 = a*np.array([.5,np.sqrt(3.)*.5,0.])
    
    pos1 = np.array([0.,0.,0.])
    pos2 = pos1 + a1
    pos3 = pos1 + a2
    pos4 = pos2 + a3
    pos5 = pos3 + a3
    pos6 = pos5 + a1
    pos7 = pos5 + a2
    pos8 = pos6 + a3
    pos9 = pos7 + a3
    pos10 = pos9 + a1
    pos11 = pos4 + a1
    pos12 = pos8 + a1
    pos13 = pos3 - a1
    pos14 = pos7 - a1
    
    return np.array([pos1,pos2,pos3,pos4,pos5,
                     pos6,pos7,pos8,pos9,pos10,
                     pos11,pos12, pos13,pos14])
                     
def get_neighbors():
    
    nList = [[1,2],[0,3],[0,4,12],[1,5,10],[2,5,6],[3,4,7],
             [4,8,13],[5,9,11],[6,9],[7,8],[3],[7],[2],[6]]
             
    return nList, np.concatenate((np.full(10,6, dtype=int), np.array([16,16,16,16])))
    
    