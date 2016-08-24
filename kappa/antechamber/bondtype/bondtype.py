"""
This module is designed to perceive bond types of kappa.Molecule instances.
Design from: http://www.sciencedirect.com/science/article/pii/S1093326305001737

@author: Alex Kerr
"""

import csv

import numpy as np

from ... import package_dir

def main(mol):
    """Return the perceived bond types list for `mol`,
    indexed like `mol.bondList`."""
    
    max_valence_state = 2000
    state_num = 0
    
    tps = 0
    
    #first we need to numerate the possible valences and respective penalty scores for each atom
    av = find_atomic_valences(mol)
    
    vstates = np.array([])
    
    while state_num < max_valence_state:
        
        #find states of given tps
        newstates = bondtype(tps, av)
        
        vstates = np.concatenate((vstates, newstates),axis=0)
        
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
    
def bondtype(tps, av):
    """Return all the combinations of valence states for the given tps."""
    
    vstates = []
    
    def dfs(index, vstate, runsum):
        """A recursive function to fill `vstates` will all valence states
        of total penalty score `tps`"""
        newstate = vstate[:]
        try:
            for valence, ps in av[index]:
                if ps+runsum <= tps:
                    dfs(index+1, newstate, runsum+ps)
        except IndexError:
            vstates.append(newstate)
    
    return np.array(vstates)
    
def boaf(vstate):
    """Return True if bond order assignment of the valence state is successful,
    otherwise return False."""
    
    return False, None
    
def find_atomic_valences(mol):
    
    file_ = "APS_kerr_edit.DAT"
    
    reader = csv.reader(open("%s/antechamber/bondtype/%s" % (package_dir, file_)), delimiter=" ", skipinitialspace=True)
    lines = []
    for line in reader:
        #populate lineList
        #this is because reader object cycles only once; was a surprising bug!
        #there must be a better way to do this
        lines.append(line)
    
    print(lines)
    av = []
    
    #for each atom in the molecule search through the APS file and add all possible valence values
    for z, nList in zip(mol.zList, mol.nList):
        for line in [line for line in lines if line[0]=="APS"]:
            if int(line[2]) == z:
                #check to see if con is specified
                if line[3] == "*":
                    #then we have a match
                    av.append(parse_line(line))
                    break
                elif int(line[3]) == len(nList):
                    #then we have a match
                    av.append(parse_line(line))
                    break
            else:
                #go to next line
                continue
        else:
            #no aps values were found for this atom
            raise ValueError("Atom type could not be assigned for bond type perception")
            
    return av
    
def parse_line(line,):
    """Return a list of 2-tuples of the possible atomic valences for a given line from
    the APS defining sheet."""
    
    possap = []
    
    for valence, entry in enumerate(line[4:]):
        if entry != "*":
            possap.append((valence, int(entry)))
            
    return possap