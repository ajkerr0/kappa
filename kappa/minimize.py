# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:52:49 2016

@author: Alex Kerr

Contains class definition(s) used to minimize the energy of Molecules
"""

class Minimizer:
    """A energy minimizer for Molecules
    
    Args:
        n (int): Max number of iterations in minimization run(s).
    Keywords:
        descent (str): Method in which the local energy minimum will be approached;
            must be key in descentDict
        search (str): Line search method; must be key in searchDict
        numgrad (bool): True if the gradients are to be calculated numerically;
            default is False
        eprec (float): Precision for energy changes when an iteration is made 
            signalling convergence has occurred
        fprec (float): Precision for force magnitude signaling
            convergence has occurred.
        efreq (int): Iteration period in which information is printed to the user;
            called frequency despite being inverse frequency."""
    
    def __init__(self, n, descent="cg", search="backtrack", 
                 numgrad=False, eprec=1e-5, fprec=1e-3,
                 efreq=1):
        self.n = n
        self.descent = descent
        self.search = search
        self.numgrad = numgrad
        self.eprec = eprec
        self.fprec = fprec
        self.efreq =  efreq
        
def steepest_descent():
    pass
    
def conjugate_gradient():
    pass

def line_search_backtrack():
    pass
        
descentDict = {"sd":steepest_descent, "cg":conjugate_gradient}
searchDict = {"backtrack":line_search_backtrack}