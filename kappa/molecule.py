# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:09:30 2016

@author: Alex Kerr

Define the Molecule class and a set of functions that `build' preset molecules.
"""

import random
from copy import deepcopy

import numpy as np

from . import package_dir

#change in position for the finite difference equations
ds = 1e-5
dx = ds
dy = ds
dz = ds

vdx = np.array([dx,0.0,0.0])
vdy = np.array([0.0,dy,0.0])
vdz = np.array([0.0,0.0,dz])
        
class Molecule:
    """A molecule, representing a collection of interacting atoms
    
    Args:
        ff (Forcefield): Forcefield that determines how the atoms interact.
        name (str): Human readable string that identifies the molecule.
        posList (ndarray): Numpy 2d array (N by 3) that contains the x,y,z coordinates of the atoms.
        nList (list): List of lists of each neighboring atom that determines bonding, indexed like posList.
        zList (ndarray): Numpy 1d array (1 by N) of the atomic numbers in the molecule, indexed like posList.
        
    Keywords:
        cbase (bool):  True if instantiated Molecule is to be designated as a carbon base structure, basically
            one of the few canonical, 2-interface, macromolecules we're trying to calculate thermal conductivity of
            like graphene, cnts, etc.
            
    Forcefield Parameters (if applicable):
        kb (ndarray): Array of harmonic bond stretching spring constants indexed like bondList.
        b0 (ndarray): Array of harmonic bond stretching equilibrium displacements indexed like bondList.
        kt (ndarray): Array of harmonic bond bending spring constants indexed like angleList.
        t0 (ndarray): Array of harmonic bond bending equilibirum displacements indexed like angleList."""
    
    def __init__(self, ff, name, posList, nList, zList, cbase=False):
        self.ff = ff
        self.name = name
        self.posList = np.array(posList)
        self.nList = nList
        self.zList = np.array(zList)
        self.faces = []
        self.cbase = cbase
        
    def __len__(self):
        return len(self.posList)
        
    def __str__(self):
        return self.name
        
    def __getitem__(self, index):
        return [self.posList[index], self.nList[index], self.zList[index]]
        
    def translate(self, transVec):
        """Translate the entire molecule by a given translation vector."""
        self.posList += np.tile(transVec, (len(self),1))
        for face in self.faces:
            face.pos += transVec
        
    def rotate(self, axis, angle):
        """Rotate the molecule about a given axis by a given angle."""
        #normalize axis, turn angle into radians
        axis = axis/np.linalg.norm(axis)
        angle = np.deg2rad(angle)
        #rotation matrix construction
        ux, uy, uz = axis
        sin, cos = np.sin(angle), np.cos(angle)
        rotMat = np.array([[cos+ux*ux*(1.-cos), ux*uy*(1.-cos)-uz*sin, ux*uz*(1.-cos)+uy*sin], 
                           [uy*ux*(1.-cos)+uz*sin, cos+uy*uy*(1.-cos), uy*uz*(1.-cos)-ux*sin], 
                           [uz*ux*(1.-cos)-uy*sin, uz*uy*(1.-cos)+ux*sin, cos+uz*uz*(1.-cos)]])
        #rotate points & interfaces
        self.posList = np.transpose(np.dot(rotMat,np.transpose(self.posList)))
        for face in self.faces:
            face.pos = np.transpose(np.dot(rotMat, np.transpose(face.pos)))
            face.norm = np.transpose(np.dot(rotMat, np.transpose(face.norm)))
            
    def invert(self):
        """Invert the molecule across the origin."""
        self.posList = -self.posList
        for face in self.faces:
            face.pos = -face.pos
            face.norm = -face.norm
            
    def hydrogenate(self):
        """Attach a hydrogen atom to every open interface atom."""
        #find all the open atoms
        openList = []
        for face in self.faces:
            openList.extend([x for x in face.atoms if x not in face.closed])
        ch = build_ch(self.ff, bond=False)
        #populate indexList
        indexList = []
        for openatom in openList:
            indexList.append((openatom, 0))
        from .operation import _combine
        for pair in indexList:
            i,j = pair
            self = _combine(self, ch, i, j, copy=False)
        self._configure()
        return self
        
    def _check_neighbors(self):
        """Raise an error if Molecule's neighbor list is not symmetric."""
        for index, nList in enumerate(self.nList):
            for neighbor in nList:
                if index not in self.nList[neighbor]:
                    raise ValueError("Molecule's neighbor list needs to be symmetric")
            
    def _configure_topology_lists(self):
        """Assign lists of the unique bonds, bond angles, dihedral angles, and improper torsionals 
        to the molecule instance."""
        bondList = []
        for i,iNList in enumerate(self.nList):
            ineighbors = [j for j in iNList if j > i]
            for j in ineighbors:
                bondList.append([i,j])
        angleList = []
        for bond in bondList:
            i,j = bond
            for k in [k for k in self.nList[j] if k > i]:
                angleList.append([i,j,k])
            for k in [k for k in self.nList[i] if k > j]:
                angleList.append([j,i,k])
        dihedralList = []
        imptorsList = []
        for angle in angleList:
            i,j,k = angle
            for l in [l for l in self.nList[k] if l != j and l > i]:
                dihedralList.append([i,j,k,l])
            for l in [l for l in self.nList[i] if l != j and l > k]:
                dihedralList.append([l,i,j,k])
            for l in [l for l in self.nList[j] if l > k]:
                imptorsList.append([i,j,k,l])
        self.bondList = np.array(bondList)
        self.angleList = np.array(angleList)
        self.dihList = np.array(dihedralList)
        self.imptorsList = np.array(imptorsList)
        
    def _configure_nonbonded_neighbors(self):
        """Assign lists of non-bonded neighbor pairings; construct the Verlet neighbor lists"""
        cutoff = 5.*self.ff.lunits
        nbnList = []
        for i, ipos in enumerate(self.posList):
            for j in [j for j in range(len(self)) if j > i]:
                if np.linalg.norm(ipos-self.posList[j]) < (cutoff + 2.) and j not in self.nList[i]:
                    nbnList.append([i,j])
        self.nbnList = np.array(nbnList)
        
    def _configure_ring_lists(self):
        """Assign all of the unique rings to the molecule instance, 
        between sizes minSize and maxSize."""
        maxSize = 9
        minSize = 3
        pathList = []
        def dfs(index, discoverList, start):
            disList = list(discoverList)
            disList.append(index)
            if len(disList) <= maxSize:
                for neighbor in self.nList[index]:
                    if neighbor not in disList:
                        dfs(neighbor, disList, start)
                    elif neighbor == start:
                        if len(disList) >= minSize:
                            pathList.append(disList)
        
        for i in range(len(self)):
            disList = []
            dfs(i, disList, i) 
        #now reduce to unique rings
        #sort each path so they can be compared
        pathList = [sorted(x) for x in pathList]
        #get unique paths (rings)
        ringList =  [list(x) for x in set(tuple(x) for x in pathList)]
        self.ringList = np.array(ringList)
        
    def _configure_aromaticity(self):
        """Assign the aromaticity of each ring in the molecule instance, 
        as defined by Antechamber (AR1 is purely aromatic).  ringList must
        be assigned first."""
        #for now define each 6-membered ring as 'AR1'
        aromaticList = []
        for ring in self.ringList:
            if len(ring) == 6:
                aromaticList.append('AR1')
            else:
                aromaticList.append(None)
        self.aromaticList = aromaticList
        
    def _configure_bondtypes(self):
        """Assign the bondtypes to the molecule instance."""
        if self.cbase is True:
            #assign valence state based on connectivity
            vstate = np.array([4 if len(x) == 3 else 3 for x in self.nList], dtype=int)
            from .antechamber.bondtype.bondtype import boaf
            match, bondorder = boaf(vstate, self.bondList)
            if match is True:
                self.bondorder = bondorder
            else:
                raise ValueError("Bond order assignment did not work for base molecule.")
        else:
            from .antechamber.bondtype.bondtype import main
            self.bondorder, self.bondtypes = main(self)
        
    def _configure_atomtypes(self):
        """Assign the atomtypes and corresponding parameter IDs to the molecule instance."""
        from .antechamber.atomtype.atomtype import main
        self.atomtypes = main(self)
        if "DU" in self.atomtypes:
            print("WARNING: Dummy atom was assessed during atomtype assignment.")
        #from these atomtypes, get their IDs
        idList = []
        for atomtype in self.atomtypes:
                idList.append(self.ff.atomTypeIDDict[atomtype]) 
        self.idList = np.array(idList)
        
    def _configure_parameters(self):
        """Assign the force parameters to the molecule instance."""
        idList = self.idList
        filename = '%s/param/%s' % (package_dir, self.ff.name)
        
        if self.ff.lengths:
            #assign kr, r0 parameters
            try:
                kbArr, b0Arr = np.load(filename+"/kb.npy"), np.load(filename+"/b0.npy")
                self.kb = kbArr[idList[self.bondList[:,0]],idList[self.bondList[:,1]]]
                self.b0 = b0Arr[idList[self.bondList[:,0]],idList[self.bondList[:,1]]]
            except IndexError:
                self.kb, self.b0 = [], []
            
        if self.ff.angles:
            #assign kt,t0 parameters
            try:
                ktArr, t0Arr = np.load(filename+"/kt.npy"), np.load(filename+"/t0.npy")
                self.kt = ktArr[idList[self.angleList[:,0]], idList[self.angleList[:,1]],
                                    idList[self.angleList[:,2]]]
                self.t0 = t0Arr[idList[self.angleList[:,0]], idList[self.angleList[:,1]],
                                    idList[self.angleList[:,2]]]
            except IndexError:
                self.kt, self.t0 = [],[]
                                
        if self.ff.dihs:
            #assign, vn, nn, gn parameters
            try:
                vnArr, nnArr, gnArr = np.load(filename+"/vn.npy"), np.load(filename+"/nn.npy"), np.load(filename+"/gn.npy")
                self.vn = vnArr[idList[self.dihList[:,0]], idList[self.dihList[:,1]],
                                    idList[self.dihList[:,2]], idList[self.dihList[:,3]]]
                self.nn = nnArr[idList[self.dihList[:,0]], idList[self.dihList[:,1]],
                                    idList[self.dihList[:,2]], idList[self.dihList[:,3]]]
                self.gn = gnArr[idList[self.dihList[:,0]], idList[self.dihList[:,1]],
                                    idList[self.dihList[:,2]], idList[self.dihList[:,3]]]
            except IndexError:
                self.vn, self.nn, self.gn = [], [], []
                                
        if self.ff.lj:
            #assign Van-dr-Waals parameters
            rvdw0Arr, epvdwArr = np.load(filename+"/rvdw0.npy"), np.load(filename+"/epvdw.npy")
            self.rvdw0 = rvdw0Arr[idList]
            self.epvdw = epvdwArr[idList]
        
    def _configure(self):
        """Call the 'configure' methods sequentially."""
#        print('Configuring bond topology...')
        self._check_neighbors()
        self._configure_topology_lists()
        self._configure_nonbonded_neighbors()
#        print('Configuring rings...').
        self._configure_ring_lists()
#        print('Configuring aromaticity...')
        self._configure_aromaticity()
#        print('Configuring bond types')
#        self._configure_bondtypes()
#        print('Configuring atom types...')
        self._configure_atomtypes()
#        print('Configuring forcefield parameters...')
        self._configure_parameters()
        
    def define_energy_routine(self):
        """Return the function that would calculate the energy of the
        molecule instance."""
        
        e_funcs = []
        
        if self.ff.lengths:
            
            ibonds,jbonds = self.bondList[:,0], self.bondList[:,1]
            def e_lengths():
                rij = self.posList[ibonds] - self.posList[jbonds]
                rij = np.linalg.norm(rij, axis=1)
                return np.sum(self.kb*(rij-self.b0)**2)
                
            e_funcs.append(e_lengths)
            
        if self.ff.angles:
            
            iangles,jangles,kangles = self.angleList[:,0], self.angleList[:,1], self.angleList[:,2]
            def e_angles():
                posij = self.posList[iangles] - self.posList[jangles]
                poskj = self.posList[kangles] - self.posList[jangles]
                rij = np.linalg.norm(posij,axis=1)
                rkj = np.linalg.norm(poskj,axis=1)
                cosTheta = np.einsum('ij,ij->i',posij,poskj)/rij/rkj
                theta = np.rad2deg(np.arccos(cosTheta))
                return np.sum(self.kt*(theta-self.t0)**2)
                
            e_funcs.append(e_angles)
            
        if self.ff.dihs:
            
            idih,jdih,kdih,ldih = self.dihList[:,0],self.dihList[:,1],self.dihList[:,2],self.dihList[:,3]
            def e_dihs():
                posji = self.posList[jdih] - self.posList[idih]
                poskj = self.posList[kdih] - self.posList[jdih]
                poslk = self.posList[ldih] - self.posList[kdih]
                rkj = np.linalg.norm(poskj,axis=1)
                cross12 = np.cross(posji, poskj)
                cross23 = np.cross(poskj, poslk)
                n1 = cross12/np.linalg.norm(cross12, axis=1)[:,None]
                n2 = cross23/np.linalg.norm(cross23, axis=1)[:,None]
                m1 = np.cross(n1, poskj/rkj[:,None])
                x,y = np.einsum('ij,ij->i', n1, n2),  np.einsum('ij,ij->i', m1, n2)
                omega = np.rad2deg(np.arctan2(y,x))
                return np.sum(self.vn*(np.ones(len(self.dihList)) + np.cos(np.radians(self.nn*omega - self.gn))))
                
            e_funcs.append(e_dihs)
            
        #non-bonded interactions
            
        if self.ff.lj:
            
            ipairs, jpairs = self.nbnList[:,0], self.nbnList[:,1]
            def e_lj(grad):
                posij = self.posList[ipairs] - self.posList[jpairs]
                rij = np.linalg.norm(posij, axis=1)
                rTerm = ((self.rvdw0[ipairs] + self.rvdw0[jpairs])/rij)**6
                return np.sum(np.sqrt(self.epvdw[ipairs]*self.epvdw[jpairs])*(rTerm**2 - 2*(rTerm)))      
            
        if self.ff.tersoff:
            #tersoff interaction here
            pass
            
        def calculate_e():
            e = 0.0 #base energy level
            for e_func in e_funcs:
                e += e_func()
            return e
                
        return calculate_e    
        
    def define_gradient_routine_numerical(self):
        """Return the function that would calculate the gradients (negative forces)
        of the atoms in the molecule instance."""
        #this will change if analytical gradients get implemented
        #for now call `define_energy_routine` again, which can be wasteful
        #we're also doing bruteforce gradient calculations (calculating all the energy, not just what bonds are involved)
        calculate_e = self.define_energy_routine()
        
        def calculate_grad():
        
            gradient = np.zeros([len(self),3])
            
            for i in range(len(self)):
                ipos = self.posList[i]
            
                ipos += vdx
                vPlusX = calculate_e()
                ipos += -2.0*vdx
                vMinusX = calculate_e()
                ipos += vdx+vdy
                vPlusY = calculate_e()
                ipos += -2.0*vdy
                vMinusY = calculate_e()
                ipos += vdy+vdz
                vPlusZ = calculate_e()
                ipos += -2.0*vdz
                vMinusZ = calculate_e()
                ipos += vdz
                
                xGrad = (vPlusX - vMinusX)/dx/2.0
                yGrad = (vPlusY - vMinusY)/dy/2.0
                zGrad = (vPlusZ - vMinusZ)/dz/2.0
                
                gradient[i] += np.array([xGrad,yGrad,zGrad])
                
            magList = np.sqrt(np.hstack(gradient)*np.hstack(gradient))
            maxForce = np.amax(magList)
            totalMag = np.linalg.norm(magList)
            return gradient, maxForce, totalMag
            
        return calculate_grad
        
    def define_gradient_routine_analytical(self):
        """Return the function that would calculate the gradients (negative forces)
        of the atoms; calculated analytically"""
        
        grad_funcs = []
        
        if self.ff.lengths:
            
            ibonds,jbonds = self.bondList[:,0], self.bondList[:,1]
            def grad_lengths(grad):
                posij = self.posList[ibonds] - self.posList[jbonds]
                rij = np.linalg.norm(posij, axis=1)
                lengthTerm = 2.*(self.kb*(rij-self.b0)/rij)[:,None]*posij
                np.add.at(grad, ibonds, lengthTerm)
                np.add.at(grad, jbonds, -lengthTerm)
                
            grad_funcs.append(grad_lengths)
                
        if self.ff.angles:
            
            iangles,jangles,kangles = self.angleList[:,0], self.angleList[:,1], self.angleList[:,2]
            def grad_angles(grad):
                posij = self.posList[iangles] - self.posList[jangles]
                poskj = self.posList[kangles] - self.posList[jangles]
                rij, rkj = np.linalg.norm(posij,axis=1), np.linalg.norm(poskj,axis=1)
                cosTheta = np.einsum('ij,ij->i',posij,poskj)/(rij*rkj)
                sqrtCos = np.sqrt(np.ones(len(cosTheta), dtype=float)-(cosTheta**2))
                dtdri = (posij*(cosTheta/rij)[:,None] - poskj/(rkj[:,None]))/((rij*sqrtCos)[:,None])
                dtdrk = (poskj*(cosTheta/rkj)[:,None] - posij/(rij[:,None]))/((rkj*sqrtCos)[:,None])
                theta = np.rad2deg(np.arccos(cosTheta))
                uTerm = (360./np.pi)*(self.kt*(theta - self.t0))
                dudri =  uTerm[:,None]*dtdri
                dudrj = -uTerm[:,None]*(dtdri + dtdrk)
                dudrk =  uTerm[:,None]*dtdrk
                np.add.at(grad, iangles, dudri)
                np.add.at(grad, jangles, dudrj)
                np.add.at(grad, kangles, dudrk)
                
            grad_funcs.append(grad_angles)
                
        if self.ff.dihs:
            
            idih,jdih,kdih,ldih = self.dihList[:,0],self.dihList[:,1],self.dihList[:,2],self.dihList[:,3]
            def grad_dihs(grad):
                posij = self.posList[idih] - self.posList[jdih]
                poskj = self.posList[kdih] - self.posList[jdih]
                poskl = self.posList[kdih] - self.posList[ldih]
                rkj = np.linalg.norm(poskj, axis=1)
                cross12 = np.cross(-posij, poskj)
                cross23 = np.cross(poskj, -poskl)
                n1 = cross12/np.linalg.norm(cross12, axis=1)[:,None]
                n2 = cross23/np.linalg.norm(cross23, axis=1)[:,None]
                m1 = np.cross(n1, poskj/rkj[:,None])
                x,y = np.einsum('ij,ij->i', n1, n2),  np.einsum('ij,ij->i', m1, n2)
                omega = np.rad2deg(np.arctan2(y,x))
                dotijkj = np.einsum('ij,ij->i',posij,poskj)/(rkj**2)
                dotklkj = np.einsum('ij,ij->i',poskl,poskj)/(rkj**2)
                dwdri = -cross12*(rkj/(np.linalg.norm(cross12, axis=1)**2))[:,None]
                dwdrl = -cross23*(-rkj/(np.linalg.norm(cross23, axis=1)**2))[:,None]
                dwdrj = (dotijkj - np.ones(len(rkj)))[:,None]*dwdri - dotklkj[:,None]*dwdrl
                dwdrk = (dotklkj - np.ones(len(rkj)))[:,None]*dwdrl - dotijkj[:,None]*dwdri
                uTerm = -(180./np.pi)*self.nn*self.vn*np.sin(self.nn*omega - self.gn)
                dudri = uTerm[:,None]*dwdri
                dudrj = uTerm[:,None]*dwdrj
                dudrk = uTerm[:,None]*dwdrk
                dudrl = uTerm[:,None]*dwdrl
                np.add.at(grad, idih, dudri)
                np.add.at(grad, jdih, dudrj)
                np.add.at(grad, kdih, dudrk)
                np.add.at(grad, ldih, dudrl)
                
            grad_funcs.append(grad_dihs)
            
        if self.ff.lj:
            
            ipairs, jpairs = self.nbnList[:,0], self.nbnList[:,1]
            def grad_lj(grad):
                posij = self.posList[ipairs] - self.posList[jpairs]
                rij = np.linalg.norm(posij, axis=1)
                rTerm = ((self.rvdw0[ipairs] + self.rvdw0[jpairs])/rij)**6
                ljTerm = 12.*np.sqrt(self.epvdw[ipairs]*self.epvdw[jpairs])*(rTerm - rTerm**2)*posij/rij/rij
                np.add.at(grad, ipairs, ljTerm)
                np.add.at(grad, jpairs, -ljTerm)
                
            grad_funcs.append(grad_lj)
                
        def calculate_grad():
            grad = np.zeros((len(self),3))
            for grad_func in grad_funcs:
                grad_func(grad)
            magList = np.sqrt(np.hstack(grad)*np.hstack(grad))
            maxForce = np.amax(magList)
            totalMag = np.linalg.norm(magList)
            return grad, maxForce, totalMag
            
        return calculate_grad
        
class Interface():
    """A molecular interface, calculate thermal conductivity across it.
    
    Args:
        atoms (array-like): Array of atom indices that compose the interface.
        norm (array-like): 3x1 array of the vector normal to the interface.
        mol (Molecule): Molecule object that the interface is in.
    Attributes:
        atoms: See above.
        norm: See above.
        pos (1x3 ndarray): The position of the interface in 3D space, calculated from the geometric center
            of the interfacial atoms
        open (1xN ndarray): Array of boolean values used to denote whether the atoms in the interface
            are 'open' or 'occupied' by an attachment.
        path (list): List of the path to this interface."""
        

    def __init__(self, atoms, norm, mol):
        self.atoms = atoms
        self.norm = np.array(norm)/np.linalg.norm(np.array(norm))
        #define interface position to be at the geometric center of its atoms
        self.pos = np.sum(mol.posList[np.array(atoms)], axis=0)/len(atoms)
        self.closed = []
        self.attached = np.array([], dtype=int)
        mol.faces.append(self)
        
def _combine(mol1, mol2, index1, index2, copy=True):
    """Return a single molecule which is the combination of input molecules.  If nextIndex1 is not
    None, also return the next index1 in the chain process in a tuple.
    
    Args:
        mol1 (Molecule): Base molecule to be combined whose indexing will be
            carried over into the new molecule.
        mol2 (Molecule): Molecule to be effectively attached to mol1.
        index1 (int): The atomic index of mol1 in which mol2 will be attached.
        index2 (int): The atomic index of mol2 that will become index1 of the new molecule.
        
    Keywords:
        copy (bool): True if the new molecule is to be a new Molecule instance, False if
            the new olecule is to be an altered mol1.  Default is True."""
            
    #check validity of parameters
    if mol1.zList[index1] != mol2.zList[index2]:
        raise ValueError("Molecules must be joined at atoms of the same atomic number")
    
    #create new instance if copy is true        
    if copy is True:
        mol1 = deepcopy(mol1)
    mol2 = deepcopy(mol2)
    
    #regularly referenced quantities
    size1, size2 = len(mol1), len(mol2)
    pos1 = mol1.posList
    z1, z2 = mol1.zList, mol2.zList
            
    #find faces of indices
    for count, face in enumerate(mol1.faces):
        if index1 in face.atoms:
            face1 = count
            norm1 = face.norm
    for count, face in enumerate(mol2.faces):
        if index2 in face.atoms:
            face2 = count
            norm2 = face.norm
            
    #change position of mol2 to put it in place
    #rotate mol2
    axis = np.cross(norm1, norm2)
    mag = np.linalg.norm(axis)
    if mag < 1e-10:
        #check for parallel/anti-parallel
        dot = np.dot(norm1, norm2)
        if dot > 0.:
            #flip the molecule
            mol2.invert()
    else:
        angle = np.degrees(np.arcsin(mag))
        mol2.rotate(axis,angle)
    #translate mol2 into position
    #mol2 might have just rotated so need to find its position again
    mol2.translate(pos1[index1] - mol2.posList[index2])
    
    #adjust molecule neighbors
    for neighbor in mol2.nList[index2]:
        if neighbor > index2:
            mol1.nList[index1].append(neighbor + size1 - 1)
        elif neighbor < index2:
            mol1.nList[index1].append(neighbor + size1)
        else:
            raise ValueError("An atom can't neighbor itself")
    for index, nList in enumerate(mol2.nList):
        if index != index2:
            newNList = []
            for neighbor in nList:
                if neighbor == index2:
                    newNList.append(index1)
                elif neighbor > index2:
                    newNList.append(neighbor + size1 - 1)
                else:
                    newNList.append(neighbor + size1)
            mol2.nList[index] = newNList
            
    #adjust face attached lists
    mol1.faces[face1].attached = np.concatenate((mol1.faces[face1].attached, np.arange(size1, size1+size2-1, dtype=int)))
            
    #delete single atom interfaces
    if len(mol1.faces[face1].atoms) == 1:
        del mol1.faces[face1]
    if len(mol2.faces[face2].atoms) == 1:
        del mol2.faces[face2]
        
    #add interfaces to base molecule
    for count, face in enumerate(mol2.faces):
        newAtoms = []
        newClosed = []
        newAttached = []
        for oldatom in face.atoms:
            if oldatom == index2:
                newIndex = index1
            elif oldatom > index2:
                newIndex = oldatom + size1 - 1
            else:
                newIndex = oldatom + size1
            newAtoms.append(newIndex)
            if oldatom in face.closed:
                newClosed.append(newIndex)
            if oldatom in face.attached:
                newAttached.append(newIndex)
        face.atoms = newAtoms
        face.closed = newClosed
        face.attached = np.array(newAttached)
        mol1.faces.append(face)
    
    #complete new molecule
    #delete the merging atom of mol2
    pos2 = np.delete(mol2.posList, index2, 0)
    z2 = np.delete(z2, index2, 0)
    del mol2.nList[index2]
    #add atoms to mol1
    mol1.posList = np.concatenate((pos1,pos2), axis=0)
    mol1.zList = np.concatenate((z1,z2), axis=0)
    mol1.nList.extend(mol2.nList)
    
    return mol1
    
def chain(molList, indexList, name=None):
    """Return a single molecule as the combination of the input molecules chained successively
    
    Args:
        molList (list): List of molecules to be chained together.
        indexList (list): List of tuples of the current indices of the [separated] molecules
            to be chained.
            
    Keywords:
        name (str): If not None, is to be the name of the complete returned molecule.  If None,
            the name will be the same as the first molecule in molList"""
            
    if len(molList) != len(indexList)+1:
        raise ValueError("There should be one more molecule than connections")
        
    molChain = molList[0]
    i,j = indexList[0]
    
    for molNum, mol in enumerate(molList[1:]):
        
        if molNum+1 < len(indexList):
            nextI = indexList[molNum+1][0]
        else:
            nextI = 0
        j = indexList[molNum][1]
        
        sizei = len(molChain)
        molChain = _combine(molChain, mol, i, j)
        
        if nextI > j:
            i = nextI + sizei - 1
        elif nextI < j:
            i = nextI + sizei
        #else i doesn't change
            
    if name is not None:
        molChain.name = name
        
    return molChain
        
def build_graphene(ff, name="", radius=3):
        
    from .lattice.graphene import main as lattice
    posList,nList,faceList = lattice(radius)
    size = len(posList)
    if not name:
        name = 'graphene_N%s' % (str(size))
    posList = np.array(posList)
    zList = np.full(size, 6, dtype=int)  #full of carbons
    graphene = Molecule(ff, name, posList, nList, zList, cbase=True)

    #add faces
    Interface(faceList[0],np.array([1.,0.,0.]), graphene)
    Interface(faceList[1], np.array([-1.,0.,0.]), graphene)

    return graphene
        
def build_cnt_armchair(ff, name="", radius=2, length=15):
    
    from .lattice.cntarm import main as lattice
    posList,nList,faceList = lattice(radius,length)
    size = len(posList)
    if not name:
        name = 'cnt_R%s_L%s' % (str(radius), str(length))
    posList = np.array(posList)
    zList = np.full(size, 6, dtype=int) #full of carbons
    cnt = Molecule( ff, name, posList, nList, zList, cbase=True)
    
    #add faces
    Interface(faceList[0], np.array([0.,1.,0.]), cnt)
    Interface(faceList[1], np.array([0.,-1.,0.]), cnt)
    
    return cnt
    
def build_dingus(ff, name="", count=5, angle=160.):
    
    from .lattice.dingus import main as lattice
    posList,nList,zList = lattice(count,angle)
    if not name:
        name = 'dingus_N%s' % (count)
    posList = 1.1*np.array(posList)
    dingus = Molecule(ff, name, posList, nList, zList)
    
    dingus._configure_topology_lists()
    if ff.name == "amber":
        dingus.idList = np.full(len(dingus),3, dtype=np.int8)
    dingus._configure_parameters()
    return dingus

def build_amine(ff, name="amine"):
    
    from .lattice.amine import main as lattice
    posList, nList = lattice()
    posList = np.array(posList)
    zList = np.array([6,7,1,1])
    amine = Molecule( ff, name, posList, nList, zList)
    
    Interface([0], np.array([0.,-1.,0.]), amine)
    
    #select one of the H's to be a driver at random
#    amine.driver = random.randint(2,3)
    amine.driver = 2
    
    return amine
        
def build_imine_chain(ff, name="", count=1):
    
    molList = [build_imine(ff)]
    indexList = [(1,0)]
    
    benz = build_benzene_block(ff)
    cc = build_cc(ff)
    ch = build_ch(ff)
    
    for i in range(count-1):
        molList.append(benz)
        molList.append(cc)
        indexList.append((5,0))
        indexList.append((1,0))
               
    molList.append(benz)
    molList.append(ch)
    indexList.append((5,0))
    
    imineChain = chain(molList,indexList)
    
    if not name:
        imineChain.name = "iminechainN%s" % count
    
    #select a H to be a preferred driver; one of the far-right hydrogens selected at random
    size = len(imineChain)
    imineChain.driver = random.choice([size-1, size-3, size-4]) 
    
    return imineChain
    
def build_pmma(ff, name="pmma", count=1):
    
    from .lattice.pmma import main as lattice
    posList, nList, zList = lattice()
    posList = np.array(posList)
    pmma = Molecule(ff, name, posList, nList, zList)
    
    Interface([0], np.array([-1.,0.,0.]), pmma)
    Interface([15], np.array([1.,0.,0.]), pmma)
    
    molList = [pmma]
    indexList = [(15,0)]
    
    for i in range(count-1):
        molList.append(pmma)
        indexList.append((15,0))
    
    molList.append(build_ch(ff))
    return chain(molList, indexList)
    
def build_imine(ff, name="imine"):

    from .lattice.imine import main as lattice
    posList, nList, zList = lattice()
    posList = np.array(posList)
    imine = Molecule(ff, name, posList, nList, zList)
    
    Interface([0], np.array([-1.,0.,0.]), imine)
    Interface([1], np.array([1.,0.,0.]), imine)    
    
    return imine
        
def build_benzene_block(ff, name="bblock"):

    from .lattice.benzene import main as lattice
    posList, nList, zList = lattice()
    posList = np.array(posList)
    bblock = Molecule(ff, name, posList, nList, zList)
    
    Interface([0], np.array([-1.,0.,0.]), bblock)
    Interface([5], np.array([1.,0.,0.]), bblock)
    
    return bblock
    
def build_pan(ff, name="polyacrylonitrile", count=2):
    """Return a polyacrylonitrile molecule."""
    
    from .lattice.panitrile import main as lattice
    posList, nList, zList = lattice()
    posList = np.array(posList)
    pan = Molecule(ff, name, posList, nList, zList)
    
    Interface([0], np.array([-1.,0.,0.]), pan)
    Interface([2], np.array([ 1.,0.,0.]), pan)
    
    molList = [pan]
    indexList = [(2,0)]
    
    for i in range(count-1):
        molList.append(pan)
        indexList.append((2,0))
        
    molList.append(build_ch(ff))
    return chain(molList, indexList)
    
def build_pvf(ff, name="polyvinylidenefluoride", count=2):
    """Return a polyvinylidenefluoride molecule."""
    
    from .lattice.polyx import main as lattice
    posList, nList, zList = lattice(1,1,9,9)
    pvf = Molecule(ff, name, posList, nList, zList)
    
    Interface([0], np.array([-1.,0.,0.]), pvf)
    Interface([2], np.array([1., 0.,0.]), pvf)
    
    molList = [pvf]
    indexList = [(2,0)]
    
    for i in range(count-1):
        molList.append(pvf)
        indexList.append((2,0))
        
    molList.append(build_ch(ff))
    return chain(molList, indexList)
    
def build_pvcl(ff, name="polyvinylidenechloride", count=2):
    """Return a polyvinylidenechloride molecule."""
    
    from .lattice.polyx import main as lattice
    posList, nList, zList = lattice(1,1,17,1)
    pvcl = Molecule(ff, name, posList, nList, zList)
    
    Interface([0], np.array([-1.,0.,0.]), pvcl)
    Interface([2], np.array([1., 0.,0.]), pvcl)
    
    molList = [pvcl]
    indexList = [(2,0)]
    
    for i in range(count-1):
        molList.append(pvcl)
        indexList.append((2,0))
        
    molList.append(build_ch(ff))
    return chain(molList, indexList)
    
def build_polyethylene(ff, name="polyethylene", count=2):
    """Return a polyethylene chain."""
    
    from .lattice.polyx import main as lattice
    posList, nList, zList = lattice(1,1,1,1)
    polyeth = Molecule(ff, name, posList, nList, zList)
    
    Interface([0], np.array([-1.,0.,0.]), polyeth)
    Interface([2], np.array([1., 0.,0.]), polyeth)
    
    molList = [polyeth]
    indexList = [(2,0)]
    
    for i in range(count-1):
        molList.append(polyeth)
        indexList.append((2,0))
        
    molList.append(build_ch(ff))
    return chain(molList, indexList)
    
def build_teflon(ff, name="teflon", count=2):
    """Return a Teflon chain."""
    
    from .lattice.polyx import main as lattice
    posList, nList, zList = lattice(9,9,9,9)
    tef = Molecule(ff, name, posList, nList, zList)
    
    Interface([0], np.array([-1.,0.,0.]), tef)
    Interface([2], np.array([1., 0.,0.]), tef)
    
    molList = [tef]
    indexList = [(2,0)]
    
    for i in range(count-1):
        molList.append(tef)
        indexList.append((2,0))
        
    molList.append(build_ch(ff))
    return chain(molList, indexList)
    
def build_c4s(ff, count=4, length=1, name=""):
    
     from .lattice.c4s import main as lattice
     posList,nList,zList = lattice(count, length)
     posList = np.array(posList)
     
     if not name:
         name  = "c4s_C%s_L%s" % (count, length)
     
     mol = Molecule(ff, name, posList, nList, zList)
     mol._configure()
     
     return mol
        
def build_ch(ff, bond=True, name="CH"):
    
    posList = np.array([[0.,0.,0.], [1.15,0.,0.]])
    if bond:
        nList = [[1],[0]]
    else:
        nList = [[],[]]
    ch = Molecule(ff, name, posList, nList, np.array([6,1]))

    Interface([0], np.array([-1.,0.,0.]), ch)    
    
    return ch
    
def build_cs(ff, norm2=[1.,0.,0.], name="CS"):
    
    posList = np.array([0.,0.,0.], [1.3,0.,0.])
    nList = [[1],[0]]
    cs = Molecule(ff, name, posList, nList, np.array([6,16]))
    
    Interface([0], np.array([-1.,0.,0.]), cs)
    Interface([1], np.array(norm2), cs)
    
    return cs
        
def build_cc(ff, name="CC"):
    
    posList = np.array([[0.,0.,0.], [1.42,0.,0.]])
    nList = [[1],[0]]
    cc = Molecule(ff, name, posList, nList, np.array([6,6]))
    
    Interface([0], np.array([-1.,0.,0.]), cc)
    Interface([1], np.array([1.,0.,0.]), cc)
    
    return cc
            
_latticeDict = {"graphene":build_graphene, "cnt":build_cnt_armchair, "amine":build_amine, 
                "imine":build_imine, "imine_chain":build_imine_chain, "pmma":build_pmma,
                "pan":build_pan}
lattices = list(_latticeDict.keys())

def build(ff, lattice, **kwargs):
    mol = _latticeDict[lattice](ff, **kwargs)
    mol._configure()
    mol.posList *= mol.ff.lunits
    return mol
        
        
            
        
        
            
        