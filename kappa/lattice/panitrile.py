"""


@author: Alex Kerr
"""

import numpy as np

def main():
    """Return polyacrylonitrile positions, neighbors, and atomic numbers."""
    
    #atomic distances
    ac = 1.40
    ah = 1.10
    
    posList = generate_pan(ac,ah)
    
    nList, zList = find_neighbors(ac, posList)
    
    return posList, nList, zList
    
def generate_pan(ac, ah):
    """Return the starting positions of the atoms in polyacrylonitrile."""
    
    xmove = np.array([ac, 0., 0.])
    ymove = np.array([0., ah, 0.])
    
    Tpos = np.array([0.,0.,0.])
    
    C1pos = Tpos + xmove
    C2pos = C1pos + xmove
    
    H1pos = C1pos + ymove
    H2pos = C1pos - ymove
    H3pos = C2pos - ymove
    
    C3pos = C2pos + 1.3*ymove
    Npos = C3pos + ymove
    
    return [Tpos, C1pos, C2pos, H1pos, H2pos, H3pos,
            C3pos, Npos]
            
def find_neighbors(ac, posList):
    """Return neighbors and atomic numbers"""
    
    nLists = [[1], [0,2,3,4], [1,5,6], [1], [1],
              [2], [2, 7], [6]]
              
    zList = np.array([6,6,6,1,1,1,6,7], dtype=int)
    
    return nLists, zList