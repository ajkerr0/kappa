"""
This module is designed to perceive bond types of kappa.Molecule instances.
Design from: http://www.sciencedirect.com/science/article/pii/S1093326305001737

@author: Alex Kerr
"""

import csv

import numpy as np

from ... import package_dir

def main(mol, nstates=2000, mintps=0, maxtps=None):
    """Return the perceived bond types list for `mol`,
    indexed like `mol.bondList`."""
    
    if maxtps is None:
        maxtps = 64*len(mol)
    
    # first numerate the possible valences and respective penalty scores 
    # for each atom
    av = find_atomic_valences(mol)
    
    # instantiate valence states
    vstates = np.array([np.zeros(len(mol), dtype=int)])
    
    # loop through each possible total penalty score
    for tps in np.arange(mintps, maxtps):
        
        # find states of given tps up to the max number of valance states
        newstates = bondtype(tps, av, nstates)
        
        if newstates is not None:
            vstates = np.concatenate((vstates, newstates),axis=0)
            num = len(newstates)
        else:
            num = 0
        
        # adjust number of valence states to be found
        nstates -= num
        
        # break out of loop if max number of valance states are found
        if nstates < 1:
            break
        
    # attempt to assign a bond order to each of the     
    for vstate in vstates[1:]:
        
        match, b_order = boaf(vstate, mol.bondList)
        
        if match:
            break
    else:
        raise ValueError("Valid bond order assignments were not found. \
                          Consider increasing the max valance state count")
        
    # develop bond types from bond order
    b_types = b_order
        
    return b_order, b_types
    
def bondtype(tps, av, maxadd):
    """Return all the combinations of valence states for the given
    total penalty score (tps) up to a limit."""
    
    vstates = []
    
    def dfs(index, vstate, runsum):
        """A recursive function to fill `vstates` will all valence states
        of total penalty score `tps`."""
        for valence, ps in av[index]:
            newstate = vstate[:]
            newsum = runsum
            if ps+runsum <= tps:
                newstate.append(valence)
                newsum += ps
                if len(newstate) < len(av):
                    dfs(index+1, newstate, newsum)
                elif len(newstate) == len(av):
                    if newsum == tps:
                        vstates.append(newstate)
            
    dfs(0, [], 0)
    
    length = len(vstates)
    
    if length < 1:
        return None
    else:
        return np.array(vstates)[:maxadd]
    
def boaf(vstate, bondList):
    """Return True & the bond order if bond order assignment of the 
    valence state is successful, otherwise return False & None."""
    
    # connectivity and bond order lists
    conList = bonds2connectivity(bondList)
    boList = np.zeros(len(bondList), dtype=int)   # zero order means unassigned
            
    # first run helper function that applies rules
    fail = apply_rules123_old(vstate, conList, bondList, boList)
    
    if fail:
        return False, None
        
    elif check_match(vstate, conList):
        return True, boList
    
    # if there are unassigned bonds, perform trial and error
    while 0 in boList:
        firstzero = np.where(boList==0)[0][0]
        for trialorder in [1,2,3]:
            testbo = np.copy(boList)
            testvs = np.copy(vstate)
            testcon = np.copy(conList)
            testbo[firstzero] = trialorder
            # apply rule 1
            i,j = bondList[firstzero]
            testcon[i] += -1
            testcon[j] += -1
            testvs[i]  += -trialorder
            testvs[j]  += -trialorder
            fail = apply_rules123_old(testvs, testcon, bondList, testbo)
            if fail:
                continue
            elif check_match(testvs, testcon):
                return True, testbo
            else:
                boList = testbo
                vstate = testvs
                conList = testcon
                break
        if fail:
            return False, None

    raise ValueError('This valence state slipped through our conditions!')
    
def check_match(vstate, conList):
    if len(np.nonzero(vstate)[0]) == 0 and len(np.nonzero(conList)[0]) == 0:
        return True
    else:
        return False
    
def apply_rules123(vstate, cons, bonds, bos):
    """ """
    
    # Rule 1: For each atom in a bond, if the bond order bo is determined,
    #   con is deducted by 1 and av is deducted by bo
    #######
    # Since no bond orders are known at the start we don't start looping through
    #   bonds
    
    # Rule 2: For one atom, if its con equals to av, the bond orders 
    # of its unassigned bonds are set to 1
    
    # Rule 3: For one atom, if its con equals to 1, the bond order 
    # of the last bond is set to av
    
    while True:
        
        # apply rule 2
        # where unassigned bonds (bo == 0) of atoms of con == av are set to single valence
        where_con_is_av = np.where(np.intersect1d(vstate == cons, vstate != 0))
        bos[np.intersect(find_bonds(where_con_is_av), bos==0)] = 1
        
        # apply rule 3
        # where unassigned bonds (bo == 0) of atoms of con == 1 are set to av
        where_con_is_one = np.where(cons == 1)
        bonds_where_con1 = find_bonds(where_con_is_one)
        bos[bonds_where_con1] = vstate[bonds_where_con1]
        
        # apply rule 1
        # where bo is determined (!= 0), substract av by bo
        # then subtract con by 1 at this atoms
        where_bo_known = np.where(bos != 0)[0]
        sub_bos_by_av, sub_con_by_1 = find_atoms(vstate.shape[0], 
                                                 bonds[where_bo_known], 
                                                 bos[where_bo_known])
        vstate -= sub_bos_by_av
        cons -= sub_con_by_1
        
        
    
def apply_rules123_old(vstate, conList, bondList, boList):
    """A helper function to enforce rules 1, 2, and 3. This is an old version of the function"""
    
    atom = 0
    while atom < len(vstate):
        # check for rules 2 and 3; if True apply rule 1
    
        # 2: set the orders of remaining bonds to 1 if con == av
        if conList[atom] == vstate[atom] and conList[atom] > 0:
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
        
        elif conList[atom] < 0 or vstate[atom] < 0:
            #boaf exits
            return True
        
        elif (conList[atom] == 0 and vstate[atom] != 0) or \
             (conList[atom] != 0 and vstate[atom] == 0):
            #boaf exits with false match
            return True
        
        atom += 1
        
    return False
    
def find_atomic_valences(mol):
    """Return a list of list of tuples containing all possible valences and 
    their respective penalty scores."""
    
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

def find_bonds(atoms, bonds):
    """Return the indices of the BONDS that contain the input atomic indices."""
    return np.where(bonds==atoms)[0]

def find_atoms(mol_length, bonds, bos):
    """Return the total known bond orders indexed by the atoms making up the
    respective bonds, the bond orders are added if known atoms are in multiple
    known bonds."""
    
    known_bos = np.zeros(mol_length)
    sub_con = np.zeros(mol_length)
    
    for (i,j), bo in zip(bonds, bos):
        known_bos[i] += bo
        known_bos[j] += bo
        sub_con[i] += 1
        sub_con[j] += 1
        
    return known_bos, sub_con
    
def bonds2connectivity(bondList):
    """Return the connectivities for each atomic index given a 2d list of
    bonds."""
    
    con0 = np.bincount(bondList[:,0])
    con1 = np.bincount(bondList[:,1])
    l0, l1 = len(con0), len(con1)
    if l0 > l1:
        con1 = np.concatenate((con1, np.zeros(l0-l1, dtype=int)))
    elif l1 > l0:
        con0 = np.concatenate((con0, np.zeros(l1-l0, dtype=int))) 
    return con0 + con1
    
def parse_line(line,):
    """Return a list of 2-tuples of the possible atomic valences for a given line from
    the APS defining sheet."""
    
    possap = []
    
    for valence, entry in enumerate(line[4:]):
        if entry != "*":
            possap.append((valence, int(entry)))
            
    return possap