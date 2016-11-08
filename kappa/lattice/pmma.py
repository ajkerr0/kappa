# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:03:18 2016

@author: Alex Kerr
"""

import numpy as np

def main():
    """Main program execution."""
    
    #atomic distance
    a = 1.40
    
    posList = generate_pmma(a)
    
    nList, zList = find_neighbors(a, posList)
    
    return posList,nList,zList
    
def generate_pmma(a):
    """Return the starting positions of the atoms in the poly methyl methacrylate chain."""
    
    Tpos = np.array([0.,0.,0.])
    
    C1pos = Tpos + np.array([a,0.,0.])
    H1pos = C1pos + np.array([0.,a,0.])
    H2pos = C1pos + np.array([0.,-a,0.])
    
    C2pos = C1pos + np.array([a,0.,0.])
    
    C3pos = C2pos + np.array([0.,-a,0.])
    H3pos = C3pos + np.array([-a,-a,0.])/np.sqrt(2.)
    H4pos = C3pos + np.array([0.,-a,0.])
    H5pos = C3pos + np.array([a,-a,0.])/np.sqrt(2.)
    
    C4pos = C2pos + np.array([0.,a,0.])
    O1pos = C4pos + np.array([a,a,0.])/np.sqrt(2.)
    O2pos = C4pos + np.array([-a,a,0.])/np.sqrt(2.)
    
    C5pos = O2pos + np.array([0.,a,0.])
    H6pos = C5pos + np.array([a,a,0.])/np.sqrt(2.)
    H7pos = C5pos + np.array([-a,a,0.])/np.sqrt(2.)
    H8pos = C5pos + np.array([0.,a,0.])
    
    return [Tpos, C1pos, H1pos, H2pos, C3pos, H3pos, H4pos, H5pos,
            C4pos, O1pos, O2pos, C5pos, H6pos, H7pos, H8pos, C2pos]
            
def find_neighbors(a, posList):
    """Return the neighbor and atomic number lists"""
    
    #16 atoms in molecule
    #do by hand
    
    nLists = [[1],[0,2,3,15],[1],[1],[5,6,7,15],[4],[4],[4],[9,10,15],
              [8],[8,11],[10,12,13,14],[11],[11],[11],[1,4,8]]
    
    return nLists, np.array([6,6,1,1,6,1,1,1,6,8,8,6,1,1,1,6], dtype=int)
    
if __name__ == "__main__":
    main()
            