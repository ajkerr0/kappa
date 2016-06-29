
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:27:56 2015

@author: alex

This program creates a 'rolled-up' finite honeycomb sheet to be used to model carbon nanotubes  The size must be specified in the main program execution.
The spacing between the lattice sites are made to be at the minimum potential of the 2-body Tersoff potential.

"""

#create lattice points
from __future__ import division
import numpy as np

def main(radius, length):
    '''Main program execution.'''
    
    #lattice constant
    a = 1
    
    #approximate radius for triangular lattice foundation 
#    triRadius = 3
#    length = 10
    
    #return triangular lattice position list
    triList = triangular_lattice(a, radius, length)

    #return hexagonal lattice position list, using triangular lattice base
    hexList = hexagonal_lattice(a, triList)
    
    #find interfaces
    faceList = find_interfaces(hexList)
    
    #position for minimum energy for 2-body problem
    rmin = 1.40
    
    #return resized position list
    resizedSheet = resize(rmin, hexList)
    
    #return curled up sheet
    finalSheet = curl(resizedSheet, a, rmin)
    
    #return list of nearest neighbors, to be used to calculate TB potential
    nList = find_neighbors(a, rmin, finalSheet)
    
    return finalSheet, nList, faceList
    
def find_interfaces(posList):
    
    posList = np.array(posList)
    
    #find max, min y-positions
    ymax = np.amax(posList[:,1])
    ymin = np.amin(posList[:,1])
    
    #add atoms on top, bottom edges
    dx = 0.5
    tface = []
    bface = []
    for count,pos in enumerate(posList):
        if pos[1] > ymax-dx:
            tface.append(count)
        if pos[1] < ymin+dx:
            bface.append(count)
            
    return [tface,bface]

def triangular_lattice(a, triRadius, length):
    '''Create a triangular lattice in a rectangular shape with length specified by triRadius
    '''    
    
    #spacing quantities
    xMove = np.sqrt(3)*a/2.0
    yMove = a/2.0
    
    posList = []
    startPos = [-triRadius*xMove*2,-length/4]
    xLength = triRadius*2 + 1
    yLength = length
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
    '''Create a hexagonal lattice with two instances of a triangular lattice, displaced appropriately.
    '''
    
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
    
def curl(posList, a, rmin):
    
    xList = []
    curlPosList = []
    
    for i in posList:
        x = i[0]
        xList.append(x)
        
    maxX = max(xList)
    minX = min(xList)
    
    circum = maxX - minX + (np.sqrt(3)*a*rmin/2)
    radius = circum/2/np.pi
    
    for i in posList:
        x = i[0]
        y = i[1]
        theta = 2*np.pi*(x-minX)/circum
        
        newX = radius*np.cos(theta)
        newZ = radius*np.sin(theta)
        
        curlPos = [newX, y, newZ]
        
        curlPosList.append(curlPos)
    
    return curlPosList
    
def find_neighbors(a, rmin, honeyLattice):
    '''Create a list of lists which contain the index in the position list of each site's nearest neighbors.
    '''
    
    neighList = []    
    
    icounter = 0      
    
    for i in honeyLattice:
        
        xi = i[0]
        yi = i[1]
        zi = i[2]
        
        jcounter = 0  
        neighbors = []
        
        for j in honeyLattice:
            
            if icounter != jcounter:
                
                xj = j[0]
                yj = j[1]
                zj = j[2]
                rij = np.sqrt((xj-xi)*(xj-xi) + (yj-yi)*(yj-yi) + (zj-zi)*(zj-zi))
            
                if rij < (3*a*rmin/2.0 + 0.2):   
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
    
