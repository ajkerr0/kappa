"""


@author: Alex Kerr
"""

import numpy as np

def main():
    
    a = 1.40
    
    posList, zList =  get_sites(a)
    
    nList = [[1],[0,2,3],[1],[1,4],[3]]
    
    return posList,nList,zList
    
def get_sites(a):
    
    c0 = np.array([0.,0.,0.])
    c1 = c0 + a*np.array([1.,0.,0.])
    
    o2 = c1 + a*np.array([1.,1.,0.])/np.sqrt(2.)
    o3 = o2 - a*np.array([0.,1.,0.])
    
    h4 = o3 + 1.1*np.array([1.,0.,0.])
    
    z = np.array([6,6,8,8,1], dtype=int)
    
    return np.array([c0,c1,o2,o3,h4]), z
    
    