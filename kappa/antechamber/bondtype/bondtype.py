"""
This module is designed to perceive bond types of kappa.Molecule instances.
Design from: http://www.sciencedirect.com/science/article/pii/S1093326305001737

@author: Alex Kerr
"""

import csv

import numpy as np

from . import package_dir

def main(mol):
    """Return the perceived bond types list for `mol`,
    indexed like `mol.bondList`."""
    
    max_valence_state = 2000
    state_num = 0
    
    tps = 0
    
    vstates = np.array([])
    
    while state_num < max_valence_state:
        
        #find states of given tps
        newstates = bondtype(tps, mol)
        
        #increment tps and state number
        tps += 1
        state_num += len(newstates)
        
    for vstate in vstates:
        
        match, b_order = boaf(vstate)
        
        if match:
            break
        
    #develop bond types from bond order
    b_types = b_order
        
    return b_types
    
def bondtype(tps, mol):
    """Return all the combinations of valence states for the given tps."""
    
    file_ = "APS_kerr_edit.DAT"
    
    reader = csv.reader(open("%s/antechamber/bondtype/%s" % (package_dir, file_)), delimiter=" ")
    lines = []
    for line in reader:
        #populate lineList
        #this is because reader object cycles only once; was a surprising bug!
        #there must be a better way to do this
        lines.append(line)
    
    vstates = np.array([])
    def dfs(index, vstate):
        """A recursive function to fill `vstates` will all valence states
        of total penalty score `tps`"""
        atomicnum, con = mol.zList[index], len(mol.nList[index])
        
    
    return vstates
    
def boaf(vstate):
    """Return True if bond order assignment of the valence state is successful,
    otherwise return False."""
    
    return False, None
    
def parse_line(line,):
    """Return True if"""
    pass