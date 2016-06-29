# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:18:45 2016

@author: Alex Kerr

Define functions that draw molecule objects.
"""

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.close("all")

atomColors = {1:"white",6:"black",7:"skyblue",8:"red",9:"green",15:"orange",16:"yellow",17:"green"}
atomicRadii = {1:25,6:70,7:65,8:60,9:50,15:100,16:100,17:100}
radList = np.zeros(max(list(atomicRadii.items()))[0]+1, dtype=np.int8)
for key,value in atomicRadii.items():
    radList[key] = value

def bonds2d(molecule, sites=False, faces=True):
    """Draw a 2d 'overhead' view of a molecule."""
    
    fig = plt.figure()
    figTitle = molecule.name
    
    posList = molecule.posList
    length = len(molecule)
    
    for bond in molecule.bondList:
        i,j = bond
        plt.plot([posList[i][0],posList[j][0]],
                 [posList[i][1],posList[j][1]],
                 color='k', zorder=-1)
        
    cList = np.zeros([length,3])
    
    if sites:
        for count in range(len(molecule)):
            cList[count] = colors.hex2color(colors.cnames[atomColors[molecule.zList[count]]])
        plt.scatter(posList[:,0],posList[:,1],s=radList[molecule.zList],c=cList)
        
    if faces:
        for face in molecule.faces:
            plt.plot(face.pos[0],face.pos[1], 'rx', markersize=15.)
            plt.scatter(posList[face.atoms][:,0], posList[face.atoms][:,1], s=50., c='red')
            plt.quiver(face.pos[0], face.pos[1], face.pos[0]+(5.*face.norm[0]), face.pos[1]+(5.*face.norm[1]),
                      color='r', headwidth=2, units='x')
    
    fig.suptitle(figTitle, fontsize=18)
    plt.axis('equal')
    plt.xlabel('x-position', fontsize=13)
    plt.ylabel('y-position', fontsize=13)
    
    plt.show()

def bonds(molecule, sites=False, indices=False):
    """Draw the molecule's bonds
    Keywords:
        sites (bool): Set True to draw atomic sites.  Default is False.
        indices (bool): Set True to draw atomic site indices near atomic sites. Default is False."""
    
    fig = plt.figure()
    ax=Axes3D(fig)
    figTitle = molecule.name
    plotSize = 5
    
    posList = molecule.posList
    length = len(posList)

    for bond in molecule.bondList:
        i,j = bond
        ax.plot([posList[i][0],posList[j][0]],
                [posList[i][1],posList[j][1]],
                [posList[i][2],posList[j][2]],
                color='k', zorder=-1)
        
    cList = np.zeros([length,3])
    
    if sites:
        for count in range(len(molecule)):
            cList[count] = colors.hex2color(colors.cnames[atomColors[molecule.zList[count]]])
            
        ax.scatter(posList[:,0],posList[:,1],posList[:,2],
                   s=radList[molecule.zList],c=cList,
                   marker='o',depthshade=False)
    
    if indices:
        ds = 0.1
        for index,pos in enumerate(posList):
            x,y,z = pos
            ax.text(x+ds,y+ds,z+ds,str(index),color="blue")           
    
    fig.suptitle(figTitle, fontsize=18)
    ax.set_xlim3d(-plotSize,plotSize)
    ax.set_ylim3d(-plotSize,plotSize)
    ax.set_zlim3d(-plotSize,plotSize)
    ax.set_xlabel('x-position' + ' (' + r'$\AA$' + ')')
    ax.set_ylabel('y-position' + ' (' + r'$\AA$' + ')')
    ax.set_zlabel('z-position' + ' (' + r'$\AA$' + ')')
    
    plt.show()
    
def faces(molecule):
    pass
    
def normal_modes(molecule,evec):
    """Draw a visualization of a normal mode of a molecule."""
    
    fig = plt.figure()
    ax=Axes3D(fig)
    
    length = len(molecule)
    x = np.zeros(length)
    y = np.zeros(length)
    z = np.zeros(length)
    u = np.zeros(length)
    v = np.zeros(length)
    w = np.zeros(length)
    
    for index,pos in enumerate(molecule.posList):
        xi,yi,zi = pos
        ui,vi,wi = np.real((evec[3*index],evec[3*index +1],evec[3*index +2]))
#        print(xi,yi,zi)
#        print(ui,vi,wi)
        x[index] = xi
        y[index] = yi
        z[index] = zi
        u[index] = ui
        v[index] = vi
        w[index] = wi
        
#    ax.quiver(x,y,z,u,v,w, length=1e-2, pivot="tail")
    ax.scatter(x,y,z)
    ax.quiver(x,y,z,u,v,w, pivot='tail')
    
    size = 10
    ax.set_xlim3d(-size,size)
    ax.set_ylim3d(-size,size)
    ax.set_zlim3d(-size,size)    
    
    ax._axis3don = False
    
#    fig.savefig("fig_nm.png",transparent=False,bbox_inches='tight')
    
    plt.show()