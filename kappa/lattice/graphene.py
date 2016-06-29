
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:27:56 2015

@author: alex

This program creates a finite honeycomb sheet to be used to model graphene.  The size must be specified in the main program execution.
The spacing between the atoms will be unphysical.  Need to resize system based on potential used.

"""

import numpy as np

#create lattice points

def main(radius):
    """Main program execution."""
    
    #lattice constant
    a = 1
    
    #approximate radius for triangular lattice foundation 
#    triRadius = nm.input_int("Input an integer for approximate radius of graphene system")
    
    #return triangular lattice position list
    triList = triangular_lattice(a, radius)

    #return hexagonal lattice position list, using triangular lattice base
    hexList = hexagonal_lattice(a, triList)
    
    #max radius for final lattice sheet, before resizing   
    maxRadius = (radius - 0.5)*a + 0.1
    
    #return the honeycomb lattice which is a trimmed version of the above hexagonal lattice
    honeyLattice = honeycomb_lattice(a, hexList, maxRadius)
#    plot_annotate(honeyLattice, "testpic")
    
    #return list of nearest neighbors, to be used to calculate TB potential
    nList = find_neighbors(a, honeyLattice)
    
    #position for minimum energy for 2-body problem
#    rmin = 1.4472
    rmin = 1.45
    
    #find the interfaces
    faceList = find_interfaces(honeyLattice)
    
    #return final, resized position list
    finalSheet = resize(rmin, honeyLattice)

    return finalSheet, nList, faceList
    
def find_interfaces(posList):
    
    posList = np.array(posList)
    
    #find maximum, minimum x-positions
    xmax = np.amax(posList[:,0])
    xmin = np.amin(posList[:,0])
    
    #add atoms on right, left edges
    dx = 0.5
    rface = []
    lface = []
    for count,pos in enumerate(posList):
        if pos[0] > xmax-dx:
            rface.append(count)
        if pos[0] < xmin+dx:
            lface.append(count)
    
    return [rface,lface]
    

def triangular_lattice(a, triRadius):
    """Create a triangular lattice in a rectangular shape with length specified by triRadius"""
    
    #spacing quantities
    xMove = np.sqrt(3)*a/2.0
    yMove = a/2.0
    
    posList = []
    startPos = [-triRadius*xMove*2,-triRadius]
    xLength = triRadius*2 + 1
    yLength = xLength*2 - 1
    m = -1
    
    for j in range(yLength):
               
        newRowStart = startPos
        startingX = newRowStart[0]
        startingY = newRowStart[1]
        
        for i in range(xLength):
            
            newX = startingX + i*xMove*2
            pos = [newX, startingY]
            posList.append(pos)
            
        x = startingX + m*xMove
        y = startingY + yMove
        startPos = [x,y]
        m = -m
        
    return posList
            

def hexagonal_lattice(a, posList):
    """Create a hexagonal lattice with two instances of a triangular lattice, displaced appropriately."""
    
    listA = []
    listB = []
    
    #spacing quantities to move entire 
    xMove = np.sqrt(3)*a/6.0
    yMove = -a/2.0
    
    for i in range(len(posList)):
        
        pos = posList[i]
        x = pos[0]
        y = pos[1]
        
        xA = x + xMove
        xB = x - xMove
        y = y + yMove
        
        posA = [xA,y]
        posB = [xB,y]
        listA.append(posA)  
        listB.append(posB)
        
    hexList = listA + listB

    return hexList
    
def honeycomb_lattice(a, hexList, maxRadius):
    """Create a honeycomb pattern which is a circular portion of the hexagonal lattice. Approximate radius specified by maxRadius."""
    
    finalList = []    
    
    for i in hexList:
        x = i[0]
        y = i[1]
        r = np.sqrt(x*x + y*y)
        
        if r < maxRadius:
            finalList.append(i)
        else:
            pass
        
    return finalList
    
def find_neighbors(a, honeyLattice):
    """Create a list of lists which contain the index in the position list of each site's nearest neighbors."""
    
    neighList = []    
    
    icounter = 0      
    
    for i in honeyLattice:
        
        xi = i[0]
        yi = i[1]        
        
        jcounter = 0  
        neighbors = []
        
        for j in honeyLattice:
            
            if icounter != jcounter:
                
                xj = j[0]
                yj = j[1]
                rij = np.sqrt((xj-xi)*(xj-xi) + (yj-yi)*(yj-yi))
            
                if rij < (np.sqrt(3)*a/2 + 0.1):   
                    neighbors.append(jcounter)
                else:
                    pass
                
            else:
                pass

            jcounter = jcounter + 1
        
        neighList.append(neighbors)
        icounter = icounter + 1

    return neighList
    
def resize(rmin, honeyLattice):
    '''Reposition the lattice sites of the system so that they are at the minimum of a hypothetical 2-body Tersoff potential.  Also give 3rd dimension and set z = 0.0
    '''
    
    finalSheet = []    
     
    for i in honeyLattice:
        x = i[0]*rmin*np.sqrt(3)
        y = i[1]*rmin*np.sqrt(3)
        z = 0.0
        rescaledPoint = [x,y,z]
        finalSheet.append(rescaledPoint)
        
    return finalSheet
        