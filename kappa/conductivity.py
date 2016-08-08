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
        self.trialList = []
            
    def add(self, molList, indexList):
        """Append a trial molecule to self.trialList with enhancements 
        from molList attached to atoms in indexList"""
        
        from .operation import _combine
        for mol, index in zip(molList,indexList):
            for count, face in enumerate(self.faces):
                if index in face.atoms:
                    iface = count
            newTrial,_ = _combine(self, mol, index, 0, 0, iface, 0, copy=False)
        newTrial._configure()
        self.trialList.append(newTrial)
        return newTrial