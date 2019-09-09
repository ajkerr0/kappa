"""


@author: Alex Kerr
"""

import numpy as np

def main(idList):
    """Return positions, neighbors, and atomic numbers of a line of 'T' atoms
    used in the amberedit forcefield."""
    
    # atomic distances
    ac = 1.40
    
    posList = generate_vinyl(ac, len(idList))
    
    nList, zList = find_neighbors(idList)
    
    return np.array(posList), nList, zList
    
def generate_vinyl(ac, num):
    """Return the starting positions of the chain"""
    
    xmove = np.array([ac, 0., 0.])
    zmove = 0.4
    
    pos = [np.array([0.,0.,0.])]
    
    pos.extend([i*xmove for i in np.arange(1, num+1)])
    pos.append(xmove*(num+1))
    
    # adjust the z-position of the backbones
    factor = zmove*np.ones(num)
    factor[::2] *= -1
    
    pos = np.array(pos)
    pos[1:num+1,2] += factor
    
    return pos
            
def find_neighbors(side_z):
    """Return neighbors and atomic numbers"""
    
    bb = len(side_z)
    
    nLists = [[1]]
    nLists.extend([[i-1, i+1] for i in np.arange(1, bb+1)])
    nLists.append([bb])
    
    zList = [6]
    zList.extend(side_z)
    zList.append(1)
    
    return nLists, np.array(zList)