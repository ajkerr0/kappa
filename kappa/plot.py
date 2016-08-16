# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:18:45 2016

@author: Alex Kerr

Define functions that draw molecule objects.
"""

import copy

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

def bonds(molecule, faces=True, ftrack=False, sites=False):
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
        for i,face in enumerate(molecule.faces):
            openAtoms = [x for x in face.atoms if x not in face.closed]
            plt.plot(face.pos[0],face.pos[1], 'rx', markersize=15., zorder=-2)
            plt.scatter(posList[openAtoms][:,0], posList[openAtoms][:,1], s=75., c='red')
            plt.scatter(posList[face.closed][:,0], posList[face.closed][:,1], s=40, c='purple')
            plt.annotate(i, (face.pos[0]-.35*face.norm[0], face.pos[1]-.35*face.norm[1]), 
                         color='r', fontsize=20)
            if np.linalg.norm(face.norm[:2]) > 0.0001:
                plt.quiver(face.pos[0]+.5*face.norm[0], face.pos[1]+.5*face.norm[1], 5.*face.norm[0], 5.*face.norm[1],
                color='r', headwidth=1, units='width', width=5e-3, headlength=2.5)
    
    fig.suptitle(figTitle, fontsize=18)
    plt.axis('equal')
    plt.xlabel('x-position', fontsize=13)
    plt.ylabel('y-position', fontsize=13)
    
    plt.show()

def bonds3d(molecule, sites=False, indices=False, save=False):
    """Draw the molecule's bonds
    Keywords:
        sites (bool): Set True to draw atomic sites.  Default is False.
        indices (bool): Set True to draw atomic site indices near atomic sites. Default is False."""
    
    fig = plt.figure()
    ax=Axes3D(fig)
    figTitle = molecule.name
    plotSize = 5
    
    posList = molecule.posList/molecule.ff.lunits
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
    ax.grid(False)
    ax._axis3don = False
    ax.set_xlim3d(-plotSize,plotSize)
    ax.set_ylim3d(-plotSize,plotSize)
    ax.set_zlim3d(-plotSize,plotSize)
    ax.set_xlabel('x-position' + ' (' + r'$\AA$' + ')')
    ax.set_ylabel('y-position' + ' (' + r'$\AA$' + ')')
    ax.set_zlabel('z-position' + ' (' + r'$\AA$' + ')')
    
    if save:
        plt.savefig("./kappa_save/%s.png" % molecule.name)
    
    plt.show()
    
def face(molecule, facenum):
    """Plot the given interface of the molecule"""
    
    mol = copy.deepcopy(molecule)
    face = mol.faces[facenum]
    
    fig = plt.figure()
    
    #rotate molecule to 'camera' position
    axis = np.cross(face.norm, np.array([0.,0.,1.]))
    mag = np.linalg.norm(axis)
    if mag < 1e-10:
        #check for parallel/anti-parallel
        dot = np.dot(face.norm, np.array([0.,0.,1.]))
        if dot < 0.:
            #don't rotate
            pass
        if dot >0.:
            #flip the molecule
            mol.invert()
    else:
        angle = np.degrees(np.arcsin(mag))
        mol.rotate(axis,angle)
        
    #center interface
    mol.translate(-face.pos)
    
    plt.scatter(mol.posList[face.atoms][:,0], mol.posList[face.atoms][:,1], s=30., c='red')
    ds = .2

#    ds = np.full(len(face.atoms), ds)   
#    plt.text(mol.posList[face.atoms][:,0]+ds, mol.posList[face.atoms][:,1]+ds, str(face.atoms))
    for atom in face.atoms:
        plt.text(mol.posList[atom][0]+ds, mol.posList[atom][1]+ds, str(atom), color='blue')
    
    fig.suptitle("Interface %s of %s" % (facenum, molecule), fontsize=18)
    plt.axis('equal')
    
    plt.show()
    
def faces(molecule):
    for count in range(len(molecule.faces)):
        face(molecule, count)
    
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
    
def participation(mol):
    """Plot the participation ratios of each normal mode as a function of their frequencies."""
    
    fig = plt.figure()
    
    from .operation import hessian, evecs
    hess = hessian(mol)
    val, vec = evecs(hess)
    
    num = np.sum((vec**2), axis=0)**2
    den = len(vec)*np.sum(vec**4, axis=0)
    
    plt.scatter(val, num/den)
    
    fig.suptitle("Participation ratios of %s" % mol.name)
    
    plt.show()
    