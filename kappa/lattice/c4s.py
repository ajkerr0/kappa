# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 22:31:39 2016

@author: Alex
"""

import numpy as np
import copy

def main(count):
    
    posList, zList = get_sites(count)
    
    nList = find_neighbors(posList)
    
    return posList,nList,zList
    
def get_sites(count):
    
    a = 1.40
    a1 = a1 = a*np.array([1.,0.,0.])
    a2 = a*np.array([-.5,np.sqrt(3.)*.5,0.])
    angle = 2.*np.pi/count
    
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
    
    pos10 = pos3 + a1
    pos11 = pos7 + a1
    pos12 = pos2 - a1
    pos13 = pos6 - a1
    
    unitpos = np.array([pos0,pos1,pos2,pos3,pos4,pos5,
                      pos6,pos7,pos8,pos9,
                      pos10,pos11,pos12,pos13])
    unitz = np.array([6,6,6,6,6,6,6,6,6,6,16,16,16,16])
    
    axis = np.array([0.,1.,0.])
    
    totalpos = unitpos
    totalz = unitz
    
    if count == 4:
        
        #left side
        newpos = copy.deepcopy(unitpos)
        newpos = rotate(newpos, axis, -90.)
        newpos = translate(newpos, -newpos[10]+unitpos[12])
        totalpos = np.concatenate((totalpos,newpos))
        totalz = np.concatenate((totalz,unitz))
        
        #right side
        newpos = translate(newpos, 4.*a*np.array([1.,0.,0.]))
        totalpos = np.concatenate((totalpos,newpos))
        totalz = np.concatenate((totalz,unitz))
        
        #bottom side
        newpos = translate(unitpos, -4.*a*np.array([0.,0.,1.]))
        totalpos = np.concatenate((totalpos,newpos))
        totalz = np.concatenate((totalz,unitz))
        
    #eliminate 'double' atoms
    deleteList = []
    for i,ipos in enumerate(totalpos):
        for j,jpos in enumerate(totalpos):
            if np.linalg.norm(ipos-jpos) < 0.1 and i > j:
                deleteList.append(j)
                
    totalpos = np.delete(totalpos, deleteList, axis=0)
    totalz = np.delete(totalz, deleteList)
    
    print(len(totalpos))
        
    return totalpos, totalz
        
    
#    hookPos = unitpos[10]
#    
#    for i in range(count-1):
#        nextPos = copy.deepcopy(unitpos)
#        unitpos = nextPos
#        nextPos = rotate(nextPos, axis, angle)
#        nextPos = translate(nextPos, nextPos[10]-hookPos)
#        totalpos = np.concatenate((totalpos,nextPos))
#        totalz = np.concatenate((totalz,unitz))
#        
#    return totalpos,totalz
    
def find_neighbors(posList):
    
    nList = []
    for i,ipos in enumerate(posList):
        innerList = []
        for j,jpos in enumerate(posList):
            if np.linalg.norm(ipos-jpos) < 1.40 + .1 and i != j :
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