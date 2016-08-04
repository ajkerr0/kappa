# -*- coding: utf-8 -*-
"""

@author: Alex Kerr

"""

class Calculation:
    
    def __init__(self, base):
        if len(base.faces) == 2:
            self.base = base
        else:
            raise ValueError("A base molecule with 2 interfaces is needed!")