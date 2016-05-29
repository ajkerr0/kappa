# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:05:49 2016

@author: alex
"""

import numpy as np
import copy

from create import nm

def main():
    """Main program execution"""
    
    posList,nList = create_dingus_lattice()
    
#    dingus = build_system_dingus(posList,nList)
    
    return posList,nList
    
def create_dingus_lattice():
    """Return the lattice points for the dingus molecule, testing bond angle forces 
    for a chain of 'CA' atoms, initially at 150 degrees"""
    
    #lattice constant
    a = 1.4   
    
    #number of atoms in the chain
    n = 6
    #initial bond angles
    theta = 160
    theta = theta*np.pi/180.0
    
    startPos = np.array([0.,0.,0.])
    
    posList = np.zeros([n,3])
    posList[0] = startPos
    
    for i in range(n-1):
        newt = (np.pi - theta)*i
        posList[i+1] = copy.copy(posList[i]) + a*np.array([np.cos(newt),np.sin(newt),0.])
        
    #also need neighbor lists
    nLists = []
    if n > 1:
        nLists.append([1])
        for i in range(1,n-1):
            nLists.append([i-1,i+1])
        nLists.append([n-2])
    else:
        nLists.append([])
        
    return posList,nLists
    
#if __name__ == "__main__":
#    main()
