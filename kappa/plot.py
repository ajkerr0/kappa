# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:18:45 2016

@author: Alex Kerr

Define functions that draw molecule objects.
"""

import copy
from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from .molecule import chains

plt.close("all")

atomColors = {1:"white",6:"black",7:"skyblue",8:"red",9:"green",15:"orange",16:"yellow",17:"green",35:"orange"}
atomicRadii = {1:25,6:70,7:65,8:60,9:50,15:100,16:100,17:100,35:115}
radList = np.zeros(max(list(atomicRadii.items()))[0]+1, dtype=np.int8)
for key,value in atomicRadii.items():
    radList[key] = value

def bonds(molecule, sites=False, indices=False, faces=False, order=False, 
          atomtypes=False, linewidth=4.):
    """Draw a 2d 'overhead' view of a molecule."""
    
    fig = plt.figure()
    figTitle = molecule.name
    
    posList = molecule.posList
    length = len(molecule)
    
    for bond in molecule.bondList:
        i,j = bond
        plt.plot([posList[i][0],posList[j][0]],
                 [posList[i][1],posList[j][1]],
                 color='k', zorder=-1, linewidth=linewidth)
        
    cList = np.zeros([length,3])
    
    if sites:
        for count in range(len(molecule)):
            cList[count] = colors.hex2color(colors.cnames[atomColors[molecule.zList[count]]])
        plt.scatter(posList[:,0],posList[:,1],s=1.5*radList[molecule.zList],c=cList,
                    edgecolors='k')
        
    if indices:
        for index, pos in enumerate(molecule.posList):
            plt.annotate(index, (pos[0]+.1, pos[1]+.1), color='b', fontsize=10)
            
    if atomtypes:
        for atomtype, pos in zip(molecule.atomtypes, molecule.posList):
            plt.annotate(atomtype, (pos[0]-.5, pos[1]-.5), color='b', fontsize=10)
        
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
                
    if order:
        for index, bo in enumerate(molecule.bondorder):
            i,j = molecule.bondList[index]
            midpoint = (molecule.posList[i]+molecule.posList[j])/2.
            plt.annotate(bo, (midpoint[0], midpoint[1]), color='k', fontsize=20)
    
    fig.suptitle(figTitle, fontsize=18)
    plt.axis('equal')
    plt.xlabel('x-position', fontsize=13)
    plt.ylabel('y-position', fontsize=13)
    
    plt.show()

def bonds3d(molecule, sites=False, indices=False, save=False,
            linewidth=2.):
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
                color='k', zorder=-1, linewidth=linewidth)
        
    cList = np.zeros([length,3])
    
    if sites:
        for count in range(len(molecule)):
            cList[count] = colors.hex2color(colors.cnames[atomColors[molecule.zList[count]]])
            
        ax.scatter(posList[:,0],posList[:,1],posList[:,2],
                   s=radList[molecule.zList],c=cList,
                   marker='o',depthshade=False,
                   edgecolors='k')
    
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
    
def bonds3d_list(molList, sites=False, indices=False, save=False,
            linewidth=2.):
    """Draw the molecule's bonds
    Keywords:
        sites (bool): Set True to draw atomic sites.  Default is False.
        indices (bool): Set True to draw atomic site indices near atomic sites. Default is False."""
    
    fig = plt.figure()
    ax=Axes3D(fig)
    plotSize = 5
    
    for molecule in molList:
    
        posList = molecule.posList/molecule.ff.lunits
        length = len(posList)
    
        for bond in molecule.bondList:
            i,j = bond
            ax.plot([posList[i][0],posList[j][0]],
                    [posList[i][1],posList[j][1]],
                    [posList[i][2],posList[j][2]],
                    color='k', zorder=-1, linewidth=linewidth)
            
        cList = np.zeros([length,3])
        
        if sites:
            for count in range(len(molecule)):
                cList[count] = colors.hex2color(colors.cnames[atomColors[molecule.zList[count]]])
                
            ax.scatter(posList[:,0],posList[:,1],posList[:,2],
                       s=radList[molecule.zList],c=cList,
                       marker='o',depthshade=False,
                       edgecolors='k')
        
        if indices:
            ds = 0.1
            for index,pos in enumerate(posList):
                x,y,z = pos
                ax.text(x+ds,y+ds,z+ds,str(index),color="blue")           
    
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
    
def normal_modes(mol,evec, track=None):
    """Draw a visualization of a normal mode of a molecule.
    
    Keywords:
        track (array-like): An array of indices to highlight in the plots.
            Indices should be in '3*index' format to reflect direction."""
    
    fig = plt.figure()
    ax=Axes3D(fig)
    
    length = len(mol)
    ar = np.arange(length, dtype=int)

    ax.scatter(mol.posList[:,0],mol.posList[:,1],mol.posList[:,2])
    ax.quiver( mol.posList[:,0],mol.posList[:,1],mol.posList[:,2],
               evec[3*ar].real, evec[3*ar + 1].real, evec[3*ar + 2].real, pivot='tail')
               
    if track is not None:
        for index in track:
            atom = int(index/3.)
            ax.scatter(mol.posList[atom,0], mol.posList[atom,1], mol.posList[atom,2], 
                       s=100., c='red', zorder=-3)
            point_index = index%3
            point = np.array([0.,0.,0.])
            point[point_index] = 1.
            ax.quiver(mol.posList[atom,0], mol.posList[atom,1], mol.posList[atom,2], 
                      point[0], point[1], point[2], pivot='tail', cmap='Reds', zorder=-2, lw=5.)
    
    size = 12
    ax.set_xlim3d(-size,size)
    ax.set_ylim3d(-size,size)
    ax.set_zlim3d(-size,size)    
    
    ax._axis3don = False
    
    plt.show()
    
def density(val):
    
#    density = gaussian_kde(val.flatten())
#    x = np.linspace(-20, 20, 1000)
#    density.covariance_factor = lambda: .25
#    density._compute_covariance()
#    plt.plot(x, density(x))
    
    n, bins, patches = plt.hist(val.flatten(), bins=200)
    plt.axis([-10000, 10000, 0, 1e6])
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
    
def grid(values):
    """Plot a grid of values."""
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['b', 'r', 'g', 'y']   

    xs = np.arange(values.shape[2])
    
    count = 0
    for block, c in zip(values, cycle(colors)):
        for row in block:
            ax.bar(xs, row, zs=count*2, zdir='y', color=c, alpha=.85)
            count += 1
            
def kappa(filename, cid, dim, dimval, avg=False, legend=True):
    """Plot kappa values along a particular dimension."""
    
    colors = ['b','r','y','c','m','g','k','w']
    
    data = np.genfromtxt(filename, 
                         dtype=[('kappa', 'f8'), ('cid', 'i4'),('clen','i4'),
                                ('cnum','i4'), ('dbav', 'i4'), ('param1', 'i4'),
                                ('param2', 'i4'), ('g', 'f8'), ('ff', 'S5'),
                                ('indices', 'S30'), ('time','S16')], delimiter=";")
    
    kappa = []
    param = []
    for datum in data:
        if datum[0] < 0.:
            continue
        else:
            kappa.append(datum[0])
            param.append(list(datum)[1:7])
    kappa = np.array(kappa)
    param = np.array(param)
    
    if dim.lower() == 'length':
        index = 1
        slice_ = 2
    elif dim.lower() == 'num':
        index = 2
        slice_ = 1
    else:
        raise ValueError('Dimension string was invalid')
    
    p = param[np.where(param[:,index]==dimval)[0],:]
    kappa = kappa[np.where(param[:,index]==dimval)[0]]
    
    for count, id_ in enumerate(cid):
        
        idnum = chains.index(id_)
        indices = np.where(p[:,0]==idnum)
        vals = p[indices,slice_][0]
        
        if avg is True:
            marker = '-'
            xy={}
            for val, k in zip(vals,kappa[indices]):
                try:
                    xy[val].append(k)
                except KeyError:
                    xy[val] = [k]
                
            x, y = [], []
            for key in xy:
                x.append(key)
                y.append(np.average(xy[key]))
        else:
            marker = 'o'
            x, y = vals, kappa[indices]
        
        plt.figure(1)
        plt.plot(x, y, colors[count]+marker,
                          label=id_, markersize=8, linewidth=3)
        
        plt.figure(2)
        plt.plot(x, np.cumsum(y), colors[count]+marker,
                          label=id_, markersize=8, linewidth=3)
        
    plt.suptitle("Integrated Thermal Conductivity")
    plt.figure(1)
    plt.suptitle("Thermal conductivity vs. Chain Length", fontsize=18)
        
    if legend:
        plt.legend()
        plt.figure(2)
        plt.legend()
        
    plt.xlabel("Chain Length (molecular units)", fontsize=15)
    plt.ylabel("Integrated Driving Power", fontsize=15)
    
    plt.figure(1)
    plt.xlabel("Chain Length (molecular units)", fontsize=15)
    plt.ylabel("Total Driving Power (ff units)", fontsize=15)
    
    plt.show()