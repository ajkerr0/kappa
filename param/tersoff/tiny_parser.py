# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:57:50 2016

@author: alex
"""

import numpy as np

def parse_tersoff():
    """Return required empty lists because the Tersoff potential uses static parameters defined
    in the potential function."""
    
    length = 1
    arr = np.zeros([length])
    
    vdwR = arr
    vdwEp = arr
    kL0 = arr
    rL0 = arr
    kA0 = arr
    theta0 = arr
    Vtors = arr
    gammators = arr
    ntors = arr
    tersoffRef = arr
    
    #save matrices
    
    #vdw arrays
    np.save("vdwR", vdwR)
    np.save("vdwE", vdwEp)
    
    #bond length arrays
    np.save("kb", kL0)
    np.save("rb0", rL0)
    
    #bond angle arrays
    np.save("ka", kA0)
    np.save("theta0", theta0)
    
    #torsion arrays    
    np.save("vtors", Vtors)
    np.save("gammators", gammators)
    np.save("ntors", ntors)
    
    #refList
    np.save("refArr",tersoffRef)
    
parse_tersoff()