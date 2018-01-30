# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 22:31:39 2016

@author: Alex
"""

import numpy as np

def main(count, length):
    
    a = 1.52
    
    posList, zList = get_sites(a, count, length)
    
    nList = find_neighbors(a, posList)
    
    return posList,nList,zList
    
def get_sites(a, count, length):
    
    a1 = a*np.array([1.,0.,0.])
    a2 = a*np.array([-.5,np.sqrt(3.)*.5,0.])
    
    pos0 = a*np.array([0.,0.,0.])
    pos1 = pos0 + a1
    pos2 = pos0 + a2
    pos3 = pos2 + 2.*a1
    pos5 = pos3 + a2
    pos4 = pos5 - a1
    pos6 = pos4 + a2
    pos7 = pos6 + 2.*a1
    pos9 = pos7 + a2
    pos8 = pos9 - a1
    
    pos10 = pos2 - a1
    pos11 = pos6 - a1
    
    unitpos = np.array([pos0,pos1,pos2,pos3,pos4,pos5,
                      pos6,pos7,pos8,pos9,
                      pos10,pos11])
    unitz = np.array([6,6,6,6,6,6,6,6,6,6,16,16])
    
    xShift = 4.*a*np.array([1.,0.,0.])
    yShift = 2.*a*np.sqrt(3.)*np.array([0.,1.,0.])
    
    totalpos = unitpos
    totalz = unitz
    
    for i in range(1,count):
        
        totalpos = np.concatenate((totalpos, unitpos + i*xShift))
        totalz = np.concatenate((totalz, unitz))
        
    newUnitPos = totalpos
    newUnitZ = totalz
        
    for i in range(1,length):
        
        totalpos = np.concatenate((totalpos, newUnitPos + i*yShift))
        totalz = np.concatenate((totalz, newUnitZ))
        
    #eliminate 'double' atoms
    deleteList = []
    for i,ipos in enumerate(totalpos):
        for j,jpos in enumerate(totalpos):
            if np.linalg.norm(ipos-jpos) < 0.1 and i > j:
                deleteList.append(j)
                
    totalpos = np.delete(totalpos, deleteList, axis=0)
    totalz = np.delete(totalz, deleteList)
    
    curledPos = curl(a, totalpos)
         
    return curledPos, totalz
    
def find_neighbors(a, posList):
    
    nList = []
    for i,ipos in enumerate(posList):
        innerList = []
        for j,jpos in enumerate(posList):
            if np.linalg.norm(ipos-jpos) < a + .1 and i != j :
                innerList.append(j)
        nList.append(innerList)
        
    return nList
        
def translate(posList, transVec):
    """Translate the positions by a given translation vector."""
    return posList + np.tile(transVec, (len(posList),1))
    
def rotate(posList, axis, angle):
    """Rotate the points about a given axis by a given angle."""
    #normalize axis, turn angle into radians
    axis = axis/np.linalg.norm(axis)
    angle = np.deg2rad(angle)
    #rotation matrix construction
    ux, uy, uz = axis
    sin, cos = np.sin(angle), np.cos(angle)
    rotMat = np.array([[cos+ux*ux*(1.-cos), ux*uy*(1.-cos)-uz*sin, ux*uz*(1.-cos)+uy*sin], 
                       [uy*ux*(1.-cos)+uz*sin, cos+uy*uy*(1.-cos), uy*uz*(1.-cos)-ux*sin], 
                       [uz*ux*(1.-cos)-uy*sin, uz*uy*(1.-cos)+ux*sin, cos+uz*uz*(1.-cos)]])              
    #rotate points
    return np.transpose(np.dot(rotMat,np.transpose(posList)))
    
def curl(a, posList):
    """Return the curled positions; in a tube."""
    
    maxX = np.amax(posList[:,0])
    minX = np.amin(posList[:,0])
    
    circum = maxX - minX + a
    radius = circum/2./np.pi
    
    theta = 2*np.pi*(posList[:,0]-np.ones(len(posList))*minX)/circum
    
    return np.transpose(np.array([radius*np.cos(theta),posList[:,1],radius*np.sin(theta)]))
    