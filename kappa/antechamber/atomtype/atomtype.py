# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:38:20 2016

@author: Alex Kerr

Define functions used to parse Forcefield atomtype files.
Rules are defined by 'ANTECHAMBER, AN ACCESSORY SOFTWARE PACKAGE FOR MOLECULE MECHANICAL CALCULATIONS' by Wang et al. (2000)
"""

import csv
import re
from ... import package_dir

nums = ["1", "2", "3", "4"]
atomicSymDict = {"H":[1], "C":[6], "N":[7], "O":[8], "F":[9], "P":[15], "S":[16], "Cl":[17], 
                 "Br":[35], "I":[53], "XX":[6,7,8,15,16], "XA":[8,16], "XB":[7,15], "XD":[15,16]}

def main(molecule):
    """Main module execution."""
    
    file_ = molecule.ff.atomtypeFile
    
    reader = csv.reader(open("%s/antechamber/atomtype/%s" % (package_dir,file_)), delimiter=" ")
    lines = []
    for line in reader:
        #populate lineList
        #this is because reader object cycles only once; was a surprising bug!
        lines.append(line)
    
    typeList = []
    for atom in range(len(molecule)):
        
        for line in lines:
            
            check,type_ = parse_line(line, atom, molecule)
            
            if check:
                break
            else:
                continue
        
        typeList.append(type_)
        
    return typeList
        
def atomic_num(entry, num):
    """Return True if two given integers match, False otherwise"""
    entry = int(entry)
    if entry == num:
        return True
    else:
        return False

def neighbors(entry, nList):
    """Return True if the number of neighbors matches the entry, False otherwise."""
    entry = int(entry)
    if entry == len(nList):
        return True
    else:
        return False

def hneighbors(entry, hcount):
    """Return True if two given integers match, False otherwise"""
    entry = int(entry)
    if entry == hcount:
        return True
    else:
        return False

def wneighbors(entry, wcount):
    """Return True if two given integers match, False otherwise"""
    entry = int(entry)
    if entry == wcount:
        return True
    else:
        return False

def atomic_prop(entry, molTuple):
    """Return true if atom fits F5 (atomic property) entry, False otherwise"""
    molecule, atomIndex = molTuple
    
    #determine atom properties
    propList = []
    nonRing = True
    for count, ring in enumerate(molecule.ringList):
        if atomIndex in ring:
            nonRing = False
            propList.append('RG%s' % (str(len(ring))))
            propList.append(molecule.aromaticList[count])
    
    if nonRing:
        propList.append('NR')
    
    #separate properties by comma
    entry = entry[1:-1].split(',')
    propList = propList[:]
    for index in range(len(entry)):
        entry[index] = entry[index].split('.')
    matchCount = 0
    for condition in entry:
        for count, prop in enumerate(propList):
            if prop in condition:
                matchCount += 1
                del propList[count]
                break
    if matchCount == len(entry):
        return True
    else:
        return False
        
def chem_env(entry, molTuple):
    """Return True if the atom matches the F6 (subtle chemical environment) entry, False otherwise"""
    
    mol, atomIndex = molTuple
    
    #turn text entry into list of paths
    pathList = []
    path_parser(entry[1:-1], pathList, [])
    
    #try to confirm all matches, starting with bigger ones (more 'specific' paths)
    #so as not to get false negatives
    pathList.sort(key=len)
    pathList.reverse()
    #find all paths of the molecule with lengths specified in pathList
    pathLengths = set([len(x) for x in pathList])
    
    truePathList = []
    def dfs(index, innerList, size):
        newList = innerList[:]
        newList.append(index)
        if len(newList) < size:
            for neighbor in mol.nList[index]:
                dfs(neighbor, newList, size)
        elif len(newList) == size:
            truePathList.append(newList)
    
    for length in pathLengths:
        #find all the paths of said length, starting from atom
        #recursive algorithm ala Molecule._configure_ring_lists
        for neighbor in mol.nList[atomIndex]:
            dfs(neighbor, [], length)
        
    #find conformance with pathList from entry and the true path list
    #need to "unconsider" subpaths of longer paths that are chosen
        
    pathCount = 0
    for path in pathList:
        #search through each actual path to find matches
        for truePath in truePathList:
            
            if len(truePath) != len(path):
                continue
            
            #go through each path entry
            entryCount = 0
            for count, pathEntry in enumerate(path):
                
                if "[" in pathEntry:
                    if atomic_prop(pathEntry[pathEntry.find("[")+1:pathEntry.find("]")], (mol, truePath[count])):
                        #if atomic property match, go on to check atomic number and # neighbors
                        pass
                    else:
                        #we already know it isn't a match, go to next path
                        break
                    charLoc = path.find("[")-1
                elif "<" in pathEntry:
                    charLoc = path.find("<")-1
                else:
                    charLoc = len(pathEntry)-1
                    
                #if the last character (before special characters) is 1 2 3 or 4, check number of neighbors
                if pathEntry[charLoc] in nums:
                    
                    if len(mol.nList[truePath[count]]) == int(pathEntry[charLoc]):
                        #go on to next check
                        pass
                    else:
                        #we know this true path doesn't match the proposed match, so go to next true path
                        break
                    
                    charLoc += -1   
                
                #finally check atomic number
                atomicNum = atomicSymDict[pathEntry[0:charLoc+1]]
                
                if mol.zList[truePath[count]] in atomicNum:
                    entryCount += 1
                else:
                    break
                
            if entryCount == len(path):
                #then we have a match for truePath and path
                #delete truePath and all of its subpaths from truePathList
                pathCount += 1
                break
        
        truePathList = eliminate_subpaths(truePathList,truePath)
                
    if pathCount == len(pathList):
        #then our atom matches the environment
        return True
    else:
        return False

def path_parser(pathString, masterList, pathList):
    """A recursive function to parse F6 path entries."""
    #split string by commas that are not within parentheses
    #Avinash Raj on StackOverflow: http://stackoverflow.com/a/26634150
    pathString = re.split(r',\s*(?![^()]*\))', pathString)
    
    #for each path, check if there are parentheses
    for path in pathString:
        if "(" in path:
            #add next item in path (comes before parentheses)
            start, stop = path.find("("), path.rfind(")")
            newList = pathList[:]
            newList.append(path[:start])
            #recursively call the function with the contents of the parentheses
            path_parser(path[start+1:stop], masterList, newList)
        else:
            newList = pathList[:]
            newList.append(path)
            masterList.append(newList)


funcList = [atomic_num, neighbors, hneighbors, wneighbors, atomic_prop, chem_env]
        
def parse_line(line, atom, molecule):
    """Return True and atomtype if atom matches a line entry, False otherwise."""

    for entryIndex, entry in enumerate(line[2:]):
        
        type_ = 'dummy'
        
        if entry == '&':
            match = True
            type_ = line[1]
            break
        elif entry == '*':
            continue
        else:
            input_ = find_input(molecule, atom, entryIndex)
            entryMatch = funcList[entryIndex](entry, input_)
            if entryMatch:
                continue
            else:
                match = False
                break
            
    return match, type_

def find_input(molecule, atomIndex, funcIndex):
    """Return the corresponding input for given entry checking function index."""
    
    if funcIndex == 0:
        #return the atomic number
        return molecule.zList[atomIndex]
    elif funcIndex == 1:
        #return the neighbor list
        return molecule.nList[atomIndex]
    elif funcIndex == 2:
        #find how many of these neighbors are hydrogens and return that number
        neighbors = molecule.nList[atomIndex]
        hcount = 0
        for neighbor in neighbors:
            z = molecule.zList[neighbor]
            if z == 1:
                hcount += 1
        return hcount
    elif funcIndex == 3:
        #find how many next nearest neighbors are `electron withdrawl` (N, O, F, Cl, Br), should be a single direct neighbor
        neighbors = molecule.nList[atomIndex]
        wcount = 0
        for neighbor in neighbors:
            nneighbors = [x for x in molecule.nList[neighbor] if x != atomIndex]
            for nneighbor in nneighbors:
                z = molecule.zList[nneighbor]
                if z in [7,8,9,17,35]:
                    wcount += 1
        return wcount
    elif funcIndex == 4:
        #just return the whole molecule and atom we're working with
        return (molecule, atomIndex)
    elif funcIndex == 5:
        #just return the whole molecule
        return (molecule, atomIndex)
    else:
        #have yet to implement F7
        return None
        
def eliminate_subpaths(masterList, path):
    """Return a path list with the sub paths removed."""
    
    subPaths = [path[0:i] for i in range(len(path)+1)]
    listCopy = masterList[:]
    
    for list_ in masterList:
        for subPath in subPaths:
            if list_ == subPath:
                listCopy.remove(subPath)
                
    return listCopy     
        
if __name__ == "__main__":
    main(amberFile)