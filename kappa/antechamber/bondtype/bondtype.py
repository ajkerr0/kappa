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
    
    max_valence_state = 50
    state_num = 0
    
    tps = 0
    
    #first we need to numerate the possible valences and respective penalty scores for each atom
    av = find_atomic_valences(mol)
    
    vstates = np.array([np.zeros(len(mol), dtype=int)])
    
    while state_num < max_valence_state:
        
        #find states of given tps
        newstates = bondtype(tps, av)
        
        print(tps)
        print(newstates)
        if newstates is not None:
            vstates = np.concatenate((vstates, newstates),axis=0)
            num = len(newstates)
        else:
            num = 0
        
        #increment tps and state number
        tps += 1
        state_num += num
        
    print(vstates)
        
    for vstate in vstates[1:]:
        
        match, b_order = boaf(vstate, mol.bondList)
        
        if match:
            break
    else:
        raise ValueError("Valid bond order assignments were not found. \
                          Consider increasing the max valance state count")
        
    #develop bond types from bond order
    b_types = b_order
        
    return b_types
    
def bondtype(tps, av):
    """Return all the combinations of valence states for the given tps."""
    
    vstates = []
    
    def dfs(index, vstate, runsum):
        """A recursive function to fill `vstates` will all valence states
        of total penalty score `tps`"""
        for valence, ps in av[index]:
            newstate = vstate[:]
            newsum = runsum
            if ps+runsum <= tps:
                newstate.append(valence)
                newsum += ps
#                print("{0},{1}".format(newstate, newsum))
                if len(newstate) < len(av):
                    dfs(index+1, newstate, newsum)
                elif len(newstate) == len(av):
                    if newsum == tps:
                        vstates.append(newstate)
            
    dfs(0, [], 0)
    
    if len(vstates) == 0:
        return None
    else:
        return np.array(vstates)
    
def boaf(vstate, bondList):
    """Return True if bond order assignment of the valence state is successful,
    otherwise return False."""
    
    #connectivity and bond order lists
    con0 = np.bincount(bondList[:,0])
    con1 = np.bincount(bondList[:,1])
    l0, l1 = len(con0), len(con1)
    if l0 > l1:
        con1 = np.concatenate((con1, np.zeros(l0-l1, dtype=int)))
    elif l1 > l0:
        con0 = np.concatenate((con0, np.zeros(l1-l0, dtype=int))) 
    conList = con0 + con1
    boList = np.zeros(len(bondList), dtype=int)   #zero order means unassigned
            
    #first run helper function
    fail = apply_rules123(vstate, conList, bondList, boList)
    
    if fail:
        return False, None
    
    #if there are unassigned bonds still...
    zeronum = 0
    while len(np.where(boList==0)[0]) > 0:
        #...perform trial and error on the first unassigned bond
        zeroindex = np.where(boList==0)[0][zeronum]
        for trialorder in [1,2,3]:
            testbo = np.copy(boList)
            testvs = np.copy(vstate)
            testcon = np.copy(conList)
            testbo[zeroindex] = trialorder
            #apply rule 1
            i,j = bondList[zeroindex]
            testcon[i] += -1
            testcon[j] += -1
            testvs[i]  += -trialorder
            testvs[j]  += -trialorder
            fail = apply_rules123(testvs, testcon, bondList, testbo)
            if fail:
                continue
            else:
                if check_match(testvs, testcon):
                    return True, testbo
        if fail:
            return False, None
#        zeronum += 1
    return False, None
    
def check_match(vstate, conList):
    if len(np.nonzero(vstate)[0]) == 0 and len(np.nonzero(conList)[0]) == 0:
        return True
    else:
        return False
    
def apply_rules123(vstate, conList, bondList, boList):
    """A helper function to enforce rules 1,2, and 3."""
    
    atom = 0
    while atom < len(vstate):
        #check for rules 2 and 3; if True apply rule 1
    
        #2: set the orders of remaining bonds to 1 if con == av
        if conList[atom] == vstate[atom] and conList[atom] != 0:
            #set bond order to 1 for all remaining bonds
            bonds1 = np.where(bondList==atom)[0]
            bonds2 = np.where(boList==0)[0]
            bonds = np.intersect1d(bonds1,bonds2)
            boList[bonds] = np.ones(len(bonds), dtype=int)
            #1: if order is determined reduced the valence and connectivity
            for i,j in bondList[bonds]:
                conList[i] += -1
                conList[j] += -1
                vstate[i]  += -1
                vstate[j]  += -1
            
            #reset
            atom = 0
            continue
        
        #3: set the orders to av if con == 1
        elif conList[atom] == 1:
            #set remaining bond to order of the remaining valence
            bonds1 = np.where(bondList==atom)[0]
            bonds2 = np.where(boList==0)[0]
            bonds = np.intersect1d(bonds1,bonds2)
            boList[bonds[0]] = vstate[atom]
            #1: if order is determined reduced the valence and connectivity
            for i,j in bondList[bonds]:
                av = vstate[atom]
                conList[i] += -1
                conList[j] += -1
                vstate[i]  += -av
                vstate[j]  += -av
            
            #reset
            atom = 0
            continue
        
        elif (conList[atom] == 0 and vstate[atom] != 0) or \
             (conList[atom] != 0 and vstate[atom] == 0):
            #boaf exits with false match
            return True
        
        atom += 1
        
    return False
    
def find_atomic_valences(mol):
    
    file_ = "APS_kerr_edit.DAT"
    
    reader = csv.reader(open("%s/antechamber/bondtype/%s" % (package_dir, file_)), delimiter=" ", skipinitialspace=True)
    lines = []
    for line in reader:
        #populate lineList
        #this is because reader object cycles only once; was a surprising bug!
        #there must be a better way to do this
        lines.append(line)
    
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