"""


@author: Alex Kerr
"""

import numpy as np

def main(side_z):
    """Return positions, neighbors, and atomic numbers of a mixture of atoms
    on a carbon backbone."""
    
    # atomic distances
    ac = 1.526
    r0 = np.zeros(36)
    r0[1] = 1.09
    r0[9] = 1.38
    r0[17] = 1.766
    r0[35] = 1.944
    
    ah = r0[side_z]
    
    posList = generate_vinyl(ac,ah, side_z.shape[1])
    
    nList, zList = find_neighbors(side_z)
    
    return np.array(posList), nList, zList
    
def generate_vinyl(ac, ah, num):
    """Return the starting positions of the polymer"""
    
    a = ac
    
    # carbon backbone
    xmove = np.array([a*np.sqrt(2./3.), 0., 0.])
    ymove = np.array([0., ah, 0.])
    zmove = a/np.sqrt(3)/2.
    
    pos = [np.array([0.,0.,0.])]
    
    pos.extend([i*xmove for i in np.arange(1, num+1)])
    pos.extend([i*xmove for i in np.arange(1, num+1)])
    pos.extend([i*xmove for i in np.arange(1, num+1)])
    pos.append(xmove*(num+1))
    
    # adjust the z-position of the backbones
    factor = zmove*np.ones(num)
    factor[::2] *= -1
    
    pos = np.array(pos)
    pos[1:num+1,2] += factor
    
    # place off-backbone atoms
    yshift = np.zeros((num, 3))
    yshift[:,1] = ah[0,:]*np.sqrt(2/3)
    pos[num+1:2*num+1] += yshift
    
    yshift[:,1] = ah[1,:]*np.sqrt(2/3)
    pos[2*num+1:3*num+1] -= yshift
    
    pos[num+1:2*num+1, 2] += factor
    pos[2*num+1:3*num+1, 2] += factor
    
    zshift = np.zeros((num, 3))
    zshift[:,2] = ah[0,:]*np.sqrt(1/3)
    pos[num+1:2*num+1] += np.sign(pos[1:num+1,2])[:,None]*zshift
    
    zshift[:,2] = ah[1,:]*np.sqrt(1/3)
    pos[2*num+1:3*num+1] += np.sign(pos[1:num+1,2])[:,None]*zshift    
    
    # adjust the first carbon and the final hydrogen positions
    a0 = 1.51
    ah0 = 1.09 
    
    pos[0] = pos[1] + a0*np.array([-np.sqrt(2/3), 0., np.sqrt(1/3)])
    pos[3*num+1] = pos[num] + ah0*np.array([np.sqrt(2/3), 0., -np.sign(pos[num,2])*np.sqrt(1/3)])
    
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