# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:24:17 2016

@author: alex
"""

import numpy as np


def main():
    """Main program execution."""
    
    posList = generate_imine_sites()
    
    nList,zList = get_neighbors()
    
    return posList,nList,zList
    
def generate_imine_sites():
    """Return the locations for the atoms in the imine group"""
    
    #atomic distance (angstroms)
    a = 1.45
    
    Tpos = np.array([0.,0.,0.])
    
    #positions of N,C,H
    Npos = Tpos + np.array([a,0.,0.])
    
    Cpos = Npos + np.array([a/np.sqrt(2),a/np.sqrt(2),0.0])
    Hpos = Cpos + np.array([-a/np.sqrt(2),a/np.sqrt(2),0.0])
    
    T2pos = Cpos + np.array([a,0.,0.])
    
    return [Tpos, T2pos, Npos, Cpos, Hpos]
    
def get_neighbors():
    
    nList = [[2],[3],[0,3],[1,2,4],[3]]
    zList = np.array([6,6,7,6,1])
    
    return nList,zList
    

    
    
    
    
