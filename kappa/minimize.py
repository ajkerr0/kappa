# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:52:49 2016

@author: Alex Kerr

Contains class definition(s) used to minimize the energy of Molecules
"""

class Minimizer:
    
    def __init__(self, method="cg", search="backtrack", numgrad=False):
        self.method = method
        self.search = search
        self.numgrad = numgrad
        
def steepest_descent():
    pass
    
def conjugate_gradient():
    pass

def line_search_backtrack():
    pass
        
methodDict = {"sd":steepest_descent, "cg":conjugate_gradient}
searchDict = {"backtrack":line_search_backtrack}