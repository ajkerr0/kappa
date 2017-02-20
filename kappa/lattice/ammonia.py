# -*- coding: utf-8 -*-
"""

@author: alex
"""

import numpy as np

def main():
    """Main program execution."""
    
    n,h1,h2,h3 = generate_ammonia_sites()
    nList = [[1,2,3],[0],[0],[0]]
    
    return [n,h1,h2,h3], nList
    
def generate_ammonia_sites():
    """Generate the locations for the atoms in the ammonia molecule"""
    
    x,y = np.array([1.,0.,0.]), np.array([0.,1.,0.])
    
    #atomic distance (angstroms)
    a = 1.40
    
    n = np.array([0.,0.,0.])
    
    h1 = n + a*y
    h2 = n - a*y/2. + a*x*(np.sqrt(3)/2)
    h3 = h2 - a*x*np.sqrt(3)
    
    return n,h1,h2,h3
        
        
    
    