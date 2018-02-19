"""


@author: Alex Kerr
"""

import numpy as np

def main(side_z):
    """Return positions, neighbors, and atomic numbers of a mixture of atoms
    on a carbon backbone."""
    
    # atomic distances
    ac = 1.40
    ah = 1.23
    
    posList = generate_vinyl(ac,ah, side_z.shape[1])
    
    nList, zList = find_neighbors(side_z)
    
    return np.array(posList), nList, zList
    
def generate_vinyl(ac, ah, num):
    """Return the starting positions of the polymer"""
    
    xmove = np.array([ac, 0., 0.])
    ymove = np.array([0., ah, 0.])
    
    pos = [np.array([0.,0.,0.])]
    
    pos.extend([i*xmove for i in np.arange(1, num+1)])
    pos.extend([ymove + i*xmove for i in np.arange(1, num+1)])
    pos.extend([-ymove + i*xmove for i in np.arange(1, num+1)])
    pos.append(xmove*(num+1))
    
    return pos
            
def find_neighbors(side_z):
    """Return neighbors and atomic numbers"""
    
    bb = side_z.shape[1]
    
    nLists = [[1]]
    nLists.extend([[i-1, i+1, i+bb, i+2*bb] for i in np.arange(1, bb)])
    nLists.append([bb-1, 2*bb, 3*bb, 3*bb+1])
    nLists.extend([[i] for i in np.arange(1, bb+1)]*2)
    nLists.append([bb])
    
    zList = [6]*(bb+1)
    zList.extend(side_z[0,:])
    zList.extend(side_z[1,:])
    zList.append(1)
    
    return nLists, np.array(zList)