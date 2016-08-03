# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 22:31:39 2016

@author: Alex
"""

import numpy as np

def main(count):
    
    posList = get_sites(count)
    
    nList = find_neighbors(posList)
    
def get_sites(count):
    
    a = 1.40
    a1 = a1 = a*np.array([1.,0.,0.])
    a2 = a*np.array([-.5,np.sqrt(3.)*.5,0.])
    angle = 2.*np.pi/count
    
    pos0 = a*np.array([0.,0.,0.])
    pos1 = pos0 + a1
    pos2 = pos0 + a2
    pos3 = pos1 + 2.*a1
    pos5 = pos3 + a2
    pos4 = pos5 - a1
    pos6 = pos4 + a2
    pos7 = pos6 + 2.*a1
    pos9 = pos7 + a2
    pos8 = pos9 - a1
    
    cUnit = np.array([pos0,pos1,pos2,pos3,pos4,pos5,
                      pos6,pos7,pos8])