# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:55:52 2016

@author: Alex Kerr
"""

import numpy as np

def main(circum,length):
    
    #lattice constant
    a = 1.
    
    #graphene is a triangular lattice with a 2 atom basis
    a1 = a*np.array([1.,0.,0.])
    a2 = a*np.array([.5,.5*np.sqrt(3.),0.])
    r1 = a*np.array([0.,0.,0.])
    r2 = a*np.array([np.sqrt(3.)/6.,-.5,0.])
    
    posList = build_xtal((a1,a2), (r1,r2), (circum,length))
    
def build_xtal(aList, rList, lengths):
    """Return the positions of the atoms in a crystal given the
    real lattice vectors, the atom basis, and the max size of the crystal
    in the directions of the lattice vectors."""
    
    posList = []
    
    startPos = np.array([0.,0.,0.])
    
#    for i in range(lengths[0]):
#        
#        for j in range(lengths[1]):
#            
            
    