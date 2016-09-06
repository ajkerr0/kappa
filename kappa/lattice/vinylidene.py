"""


@author: Alex Kerr
"""

import numpy as np

def main(x1,x2):
    """Return polyvinylidene fluoride positions, neighbors, and atomic numbers."""
    
    #atomic distances
    ac = 1.40
    ah = 1.10
    
    posList = generate_vinyl(ac,ah)
    
    nList, zList = find_neighbors(x1,x2)
    
    return posList, nList, zList
    
def generate_vinyl(ac, ah):
    """Return the starting positions of the atoms in polyvinylidene fluoride."""
    
    xmove = np.array([ac, 0., 0.])
    ymove = np.array([0., ah, 0.])
    
    Tpos = np.array([0.,0.,0.])
    
    C1pos = Tpos + xmove
    C2pos = C1pos + xmove
    
    H1pos = C1pos + ymove
    H2pos = C1pos - ymove
    
    X1pos = C2pos - ymove
    X2pos = C2pos + ymove
    
    return [Tpos, C1pos, C2pos, H1pos, H2pos, X1pos, X2pos]
            
def find_neighbors(x1,x2):
    """Return neighbors and atomic numbers"""
    
    nLists = [[1], [0,2,3,4], [1,5,6], [1], [1],
              [2], [2]]
              
    zList = np.array([6,6,6,1,1,x1,x2], dtype=int)
    
    return nLists, zList