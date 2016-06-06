# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:09:30 2016

@author: Alex Kerr

Define the Molecule class and a set of functions that `build' preset molecules.
"""

import numpy as np
#from numpy import array,full
from .forcefield import forcefieldList

#default values
defaultFF = forcefieldList[0]()  #Amber

#change in position for the finite difference equations
ds = 1e-5
dx = ds
dy = ds
dz = ds

vdx = np.array([dx,0.0,0.0])
vdy = np.array([0.0,dy,0.0])
vdz = np.array([0.0,0.0,dz])
        
class Molecule:
    """A molecule, representing a collection of atoms
    
    Args:
        ff (Forcefield): Forcefield that determines how the atoms interact.
        name (str): Human readable string that identifies the molecule.
        posList (ndarray): Numpy 2d array (N by 3) that contains the x,y,z coordinates of the atoms.
        nList (list): List of lists of each neighboring atom that determines bonding, indexed like posList.
        zList (ndarray): Numpy 1d array (N by 1) of the atomic numbers in the molecule, indexed like posList
        
    Keywords:
        orientation (ndarray): Numpy 1d array (3 by 1) that determines the 'direction' of the molecule.
            Primarily used for chaining molecules together."""
    
    def __init__(self, ff, name, posList, nList, zList, orientation=None):
        self.ff = ff
        self.name = name
        self.posList = posList
        self.nList = nList
        self.zList = zList
        self.orientation = orientation
        
    def __len__(self):
        return len(self.posList)
        
    def __str__(self):
        return self.name
        
    def __getitem__(self, index):
        return [self.posList[index], self.nList[index], self.zList[index]]
        
    def translate(self, transVec):
        """Translate the entire molecule by a given translation vector."""
        self.posList += np.tile(transVec, (len(self),1))
        
    def rotate(self, axis, angle):
        """Rotate the molecule about a given axis by a given angle."""
        #normalize axis, turn angle into radians
        axis = axis/np.linalg.norm(axis)
        angle = np.deg2rad(angle)
        #rotation matrix construction
        ux, uy, uz = axis
        sin, cos = np.sin(angle), np.cos(angle)
        rotMat = np.array([[cos+ux*ux*(1.-cos), ux*uy*(1.-cos)-uz*sin, ux*uz*(1.-cos)+uy*sin], \
                           [uy*ux*(1.-cos)+uz*sin, cos+uy*uy*(1.-cos), uy*uz*(1.-cos)-ux*sin], \
                           [uz*ux*(1.-cos)-uy*sin, uz*uy*(1.-cos)+ux*sin, cos+uz*uz*(1.-cos)]])              
        #rotate points & orientation
        pos = np.matrix(self.posList).T
        pos = rotMat*pos
        orient = np.matrix(self.orientation).T
        orient = rotMat*orient
        self.posList = pos.T.A  #turn it back into np array format
        self.orientation = orient.T.A[0]
            
    def _configure_structure_lists(self):
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
            iNList,jNList = self.nList[i],self.nList[j]
            jneighbors = [k for k in jNList if k > i]
            for k in jneighbors:
                angleList.append([i,j,k])
            ineighbors = [k for k in iNList if k > j]
            for k in ineighbors:
                angleList.append([j,i,k])
        dihedralList = []
        for bond in bondList:
            i,j = bond
            jNList = self.nList[j]
            jneighbors = [k for k in jNList if k != i]
            for k in jneighbors:
                kNList = self.nList[k]
                kneighbors = [l for l in kNList if l > i and l != j]
                for l in kneighbors:
                    dihedralList.append([i,j,k,l])
        imptorsList = []
        for angle in angleList:
            j,i,k = angle
            jNList = self.nList[j]
            jneighbors = [l for l in jNList if l != i and l > k]
            for l in jneighbors:
                imptorsList.append([i,j,k,l])
        self.bondList = np.array(bondList)
        self.angleList = np.array(angleList)
        self.dihedralList = np.array(dihedralList)
        self.imptorsList = np.array(imptorsList)
        
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
        
    def _configure_atomtypes(self):
        """Assign the atomtypes and corresponding parameter IDs to the molecule instance."""
        from .antechamber.atomtype import main
        self.atomtypes = main(self)
        if "DU" in self.atomtypes:
            print("WARNING: Dummy atom was assessed during atomtype assignment.")
        #from these atomtypes, get their IDs
        idList = []
        for atomtype in self.atomtypes:
                idList.append(self.ff.atomTypeIDDict[atomtype]) 
        self.idList = np.array(idList)
        
    def _configure_parameters(self):
        """Assign the forcefield parameters to the molecule instance."""
        self.ff._configure_parameters(self)
        
    def _configure(self):
        """Call the `configure' methods."""
        self._configure_structure_lists()
        self._configure_ring_lists()
        self._configure_aromaticity()
        self._configure_atomtypes()
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
                return np.sum(self.kbList*(rij-self.l0List)**2)
                
            e_funcs.append(e_lengths)
            
        if self.ff.angles:
            
            iangles,jangles,kangles = self.angleList[:,0], self.angleList[:,1], self.angleList[:,2]
            def e_angles():
                posij = self.posList[iangles] - self.posList[jangles]
                poskj = self.posList[kangles] - self.posList[jangles]
                rij = np.linalg.norm(posij,axis=1)
                rkj = np.linalg.norm(poskj,axis=1)
                cosTheta = np.einsum('ij,ij->i',posij,poskj)/rij/rkj
                theta = np.degrees(np.arccos(cosTheta))
                return np.sum(self.kaList*(theta-self.t0List)**2)
                
            e_funcs.append(e_angles)
            
#        if self.ff.dihs:
            
#            idih,jdih,kdih,ldih = dihs[:,0],dihs[:,1],dihs[:,2],dihs[:3]
#            def e_dihs():
#                posji = pos[jdih] - pos[idih]
#                poskj = pos[kdih] - pos[jdih]
#                poslk = pos[ldih] - pos[kdih]
#                rkj = np.linalg.norm(poskj,axis=1)
#                cross12 = 2
            
        #non-bonded interactions    
            
        def calculate_e():
            e = 0.0 #base energy level
            for e_func in e_funcs:
                e += e_func()
            return e
                
        return calculate_e    
        
    def define_gradient_routine(self):
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
#                print vPlusX
                ipos += -2.0*vdx
                vMinusX = calculate_e()
#                print vMinusX
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
        
        
def build_graphene(ff, name="", radius=3):
        
    from .lattice.graphene import main as lattice
    posList,nList = lattice(radius)
    size = len(posList)
    if not name:
        name = 'graphene_N%s' % (str(size))
    posList = np.array(posList)
    zList = np.full(size, 6, dtype=int)  #full of carbons
    graphene = Molecule(ff, name, posList, nList, zList, orientation=np.array([0.,1.,0.]))
    graphene._configure()
    return graphene
        
def build_cnt_armchair(ff, name="", radius=2, length=15):
    
    from .lattice.cntarm import main as lattice
    posList,nList = lattice(radius,length)
    size = len(posList)
    if not name:
        name = 'cnt_R%s_L%s' % (str(radius), str(length))
    posList = np.array(posList)
    zList = np.full(size, 6, dtype=int) #full of carbons
    cnt = Molecule( ff, name, posList, nList, zList, orientation=np.array([0.,1.,0.]))
    cnt._configure()
    return cnt

def build_amine(ff, name=""):
    
    from .lattice.amine import main as lattice
    posList, nList = lattice()
    if not name:
        name = 'amine'
    posList = np.array(posList)
    zList = np.array([6,7,1,1])
    amine = Molecule( ff, name, posList, nList, zList, orientation=np.array([0.,1.,0.]))
    amine._configure()
    return amine
        
def build_imine_chain(ff, name="", count=1):
    
    molList = [build_imine(ff)]
    indexList = [(4,0)]
    
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
    
    from operation import chain
    imineChain = chain(molList,indexList)
    
    return imineChain
    
def build_imine(ff, name=""):

    from .lattice.imine import main as lattice
    posList, nList, zList = lattice()
    if not name:
        name = 'imine'
    posList = np.array(posList)
    imine = Molecule(ff, name, posList, nList, zList, orientation=np.array([1.,0.,0.]))
    imine._configure()
    return imine
        
def build_benzene_block(ff, name=""):

    from .lattice.benzene import main as lattice
    posList, nList, zList = lattice()
    if not name:
        name = "bblock"
    posList = np.array(posList)
    bblock = Molecule(ff, name, posList, nList, zList, orientation=np.array([1.,0.,0.]))
    bblock._configure()
    return bblock
        
def build_ch(ff, name="CH"):
    
    posList = np.array([[0.,0.,0.], [1.15,0.,0.]])
    nList = [[1],[0]]
    ch = Molecule(ff, name, posList, nList, np.array([6,1]), orientation=np.array([1.,0.,0.]))
    ch._configure()
    return ch
        
def build_cc(ff, name="CC"):
    
    posList = np.array([[0.,0.,0.], [1.42,0.,0.]])
    nList = [[1],[0]]
    cc = Molecule(ff, name, posList, nList, np.array([6,6]), orientation=np.array([1.,0.,0.]))
    cc._configure()
    return cc  
    
            
_latticeDict = {"graphene":build_graphene, "cnt":build_cnt_armchair, "amine":build_amine, 
                "imine":build_imine, "chain":build_imine_chain}
lattices = _latticeDict.keys()

def build(lattice, ff=defaultFF, **kwargs):
    mol = _latticeDict[lattice](ff, **kwargs)
    return mol
        
        
            
        
        
            
        