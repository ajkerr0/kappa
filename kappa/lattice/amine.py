# -*- coding: utf-8 -*-
"""

@author: alex
"""

import numpy as np

def main():
    """Main program execution."""
    
    Cpos,Npos,H1pos,H2pos, = generate_amine_sites()
    nList = [[1],[0,2,3],[1],[1]]
    
    return [Cpos,Npos,H1pos,H2pos], nList
    
def generate_amine_sites():
    """Generate the locations for the atoms in the amine group"""
    
    #atomic distance (angstroms)
    a = 1.45
    
    Cpos = np.array([0.,0.,0.])
    
    #positions of N, H, H needed
    Npos = Cpos + np.array([0., a, 0.])
    
    H1pos = Npos + np.array([-a/np.sqrt(2.),a/np.sqrt(2.),0.])
    H2pos = Npos + np.array([a/np.sqrt(2.), a/np.sqrt(2.),0.])
    
    return Cpos,Npos,H1pos,H2pos
        
        
    
    