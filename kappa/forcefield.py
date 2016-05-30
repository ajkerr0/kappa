# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:16:37 2016

@author: alex
"""

import numpy as np

from . import package_dir

#change in position for the finite difference equations
ds = 1e-7
dx = ds
dy = ds
dz = ds

vdx = np.array([dx,0.0,0.0])
vdy = np.array([0.0,dy,0.0])
vdz = np.array([0.0,0.0,dz])

#forcefield class definitions
global_cutoff = 5.0 #angstroms

class Forcefield:
    
    def __init__(self, name, energyUnits, lengthUnits):
        self.name = name
        self.eUnits = energyUnits  #relative to kcal/mol
        self.lUnits = lengthUnits  #relative to angstroms
        
class Amber(Forcefield):
    
    def __init__(self, name="amber", energyUnits=1.0, lengthUnits=1.0):
        Forcefield.__init__(self, name, energyUnits, lengthUnits)
        self.atomTypeIDDict = {"CT":1,"C":2,"CA":3,"CM":4,"CC":5,"CV":6,"CW":7,"CR":8,"CB":9,"C*":10, "CZ":3,
                               "CN":11,"CK":12,"CQ":13,"N":14,"NA":15,"NB":16,"NC":17,"N*":18,"N2":19,"N3":20, "NT":19,
                               "OW":21,"OH":22,"OS":23,"O":24,"O2":25,"S":26,"SH":27,"P":28,"H":29,"HW":30, 
                               "HO":31,"HS":32,"HA":33,"HC":34,"H1":35,"H2":36,"H3":37,"HP":38,"H4":39,"HS":40,
                               "DU":1}
        self.atomtypeFile = "AMBER_kerr_edit.txt"
        
    def _configure_parameters(self, molecule):
        """Store parameters in bonds, angles, dihedrals, etc. along with non-bonded iteractions."""
        idList = molecule.idList
        uIDList = np.unique(idList)
        uiddim = len(uIDList)
        bondList = molecule.bondList
        angleList = molecule.angleList
        dihedralList = molecule.dihedralList
        fileName = "%s/param/%s" % (package_dir, self.name)
        refArr = np.load(fileName+"/refArr.npy")
        r0Arr, epArr = np.load(fileName+"/vdwR.npy"), np.load(fileName+"/vdwE.npy")
        kbArr, l0Arr = np.load(fileName+"/kb.npy"), np.load(fileName+"/rb0.npy")
        kaArr, theta0Arr = np.load(fileName+"/ka.npy"), np.load(fileName+"/theta0.npy")
        vnArr, nArr, gammaArr = np.load(fileName+"/vtors.npy"), np.load(fileName+"/ntors.npy"), np.load(fileName+"/gammators.npy")
        r0dim, epdim = np.full(len(r0Arr.shape),uiddim,dtype=int), np.full(len(epArr.shape),uiddim,dtype=int)
        r0Array, epArray = np.zeros(r0dim), np.zeros(epdim)
        kbList, l0List = np.zeros(len(bondList)), np.zeros(len(bondList))
        kaList, t0List = np.zeros(len(angleList)), np.zeros(len(angleList))
        vnList, nnList, gnList = np.zeros(len(dihedralList)), np.zeros(len(dihedralList)), np.zeros(len(dihedralList))
        paramLists = [[r0Array,r0Arr],[epArray,epArr]]
        for paramList in paramLists:
            if len(paramList[0].shape) == 1:
                for i,refi in enumerate(uIDList):
                    paramList[0][i] = paramList[1][refArr[refi]]
            else:
                print "ERROR IN AMBER PARAMETERS"
        for count,bond in enumerate(bondList):
            i,j = bond
            kbList[count],l0List[count] = kbArr[idList[i],idList[j]], l0Arr[idList[i],idList[j]]
        for count,angle in enumerate(angleList):
            i,j,k = angle
            kaList[count],t0List[count] = kaArr[idList[i],idList[j],idList[k]], theta0Arr[idList[i],idList[j],idList[k]]
        for count,dihedral in enumerate(dihedralList):
            i,j,k,l = dihedral
            idi,idj,idk,idl = idList[i], idList[j], idList[k], idList[l]
            vnList[count], nnList[count], gnList[count] = vnArr[idi,idj,idk,idl],nArr[idi,idj,idk,idl],gammaArr[idi,idj,idk,idl]
        maxIndex = np.max(idList)
        molecule.refList = np.zeros(maxIndex+1)
        for i,index in enumerate(idList):
            molecule.refList[index] = i
        molecule.kbList, molecule.l0List = kbList, l0List
        molecule.kaList, molecule.t0List = kaList, t0List
        molecule.vnList, molecule.nnList, molecule.gammaList = vnList, nnList, gnList
            
    def define_energy_routine(self,molecule,grad=True):
        
        pos = molecule.posList
        bondList = molecule.bondList
        angleList = molecule.angleList
        dihList = molecule.dihedralList
        
        kbList,l0List = molecule.kbList, molecule.l0List
        kaList,t0List = molecule.kaList, molecule.t0List
#        vnList,nnList,gnList = molecule.vnList,molecule.nnList,molecule.gammaList
        
#        refList = molecule.refList
#        cutoff = self.cutoff
        
        def calculate_e(index):
            
            e = 0.0
                
            if index is None:
                bonds = bondList
                angles = angleList
                dihs = dihList
                kb,l0 = kbList,l0List
                ka,t0 = kaList,t0List
#                vn,nn,gn = vnList,nnList,gnList
            else:
                whereBonds = np.where(bondList==index)[0]
                whereAngles = np.where(angleList==index)[0]
                whereDihs = np.where(dihList==index)[0]
                bonds = bondList[whereBonds]
                angles = angleList[whereAngles]
                dihs = dihList[whereDihs]
                kb,l0 = kbList[whereBonds],l0List[whereBonds]
                ka,t0 = kaList[whereAngles],t0List[whereAngles]
#                vn,nn,gn = vnList[whereDihs],nnList[whereDihs],gnList[whereDihs]
                
#                bonds = np.array([bond for bond in bondList if index in bond])
#                angles = np.array([angle for angle in angleList if index in angle])
#                dih = np.array([dihedral for dihedral in dihedralList if index in dihedral])
                    
            ibonds,jbonds = bonds[:,0], bonds[:,1]
            iangles,jangles,kangles = angles[:,0],angles[:,1],angles[:,2]
#            idih,jdih,kdih,ldih = dihs[:,0],dihs[:,1],dihs[:,2],dihs[:3]
            
            #bond terms
            rij = pos[ibonds] - pos[jbonds]
            rij = np.linalg.norm(rij,axis=1)
            eList = kb*(rij-l0)**2
            e += np.sum(eList)
            
#            if grad==False:
#                print rij
#                print kb
#                print l0
            
            #angle terms
            posij = pos[iangles] - pos[jangles]
            poskj = pos[kangles] - pos[jangles]
            rij = np.linalg.norm(posij,axis=1)
            rkj = np.linalg.norm(poskj,axis=1)
            cosTheta = np.einsum('ij,ij->i',posij,poskj)/rij/rkj
            theta = np.degrees(np.arccos(cosTheta))
            eList = ka*(theta-t0)**2
            e += np.sum(eList)
                
            #dihedral terms
#            posji = pos[jdih] - pos[idih]
#            poskj = pos[kdih] - pos[jdih]
#            poslk = pos[ldih] - pos[kdih]
#            rkj = np.linalg.norm(poskj,axis=1)
#            cross12 = 2
#                
#            for dihedral in iDihedrals:
#                
#                i,j,k,l = dihedral.dihedral
#                atomi,atomj,atomk,atoml = atomList[i],atomList[j],atomList[k],atomList[l]
#                posi,posj,posk,posl = atomi.pos,atomj.pos,atomk.pos,atoml.pos
#                #coordination
#                posji = posj-posi #b1
#                poskj = posk-posj #b2
#                poslk = posl-posk #b3
#                rkj = np.sqrt(poskj.dot(poskj))
#                cross12 = np.cross(posji,poskj)
#                cross23 = np.cross(poskj,poslk)
#                n1 = cross12/np.sqrt(cross12.dot(cross12))
#                n2 = cross23/np.sqrt(cross23.dot(cross23))
#                m1 = np.cross(n1, poskj/rkj)
#                x = n1.dot(n2)
#                y = m1.dot(n2)
#                omega = np.degrees(np.arctan2(y,x))
#                #values
#                vn,n,gamma = dihedral.vn,dihedral.n,dihedral.gamma
#                #energy
#                e += vn*(1.0 + np.cos(np.radians(n*omega - gamma)))
                
            #non-bonded terms
#            for i,atomi in enumerate(atomList):
#                
#                iActs = np.array([j for j in range(len(atomList)) if j > i and j not in atomi.nList])  #analysis:ignore
#                
#                for j in iActs:
#                    
#                    atomj = atomList[j]
#                    idi,idj = refList[atomi.id],refList[atomj.id]
#                    posi,posj = atomi.pos,atomj.pos
#                    posij = posi-posj
#                    rij = np.sqrt(posij.dot(posij))
#                    
#                    if rij < cutoff:
#                        
#                        epi = molecule.epList[idi]
#                        epj = molecule.epList[idj]
#                        epij = np.sqrt(epi*epj)
#                        r0i = molecule.r0List[idi]
#                        r0j = molecule.r0List[idj]
#                        r0ij = r0i + r0j
#                        r = (r0ij/rij)**6
#                        #energy & energy derivatives
#                        e += epij*(r*r - 2.0*r)
                
            return e
                
        def calculate_gradient(index):
            
            ipos = pos[index]
            
            ipos += vdx
            vPlusX = calculate_e(index)
            ipos += -2.0*vdx
            vMinusX = calculate_e(index)
            ipos += vdx+vdy
            vPlusY = calculate_e(index)
            ipos += -2.0*vdy
            vMinusY = calculate_e(index)
            ipos += vdy+vdz
            vPlusZ = calculate_e(index)
            ipos += -2.0*vdz
            vMinusZ = calculate_e(index)
            ipos += vdz
            
            xGrad = (vPlusX - vMinusX)/dx/2.0
            yGrad = (vPlusY - vMinusY)/dy/2.0
            zGrad = (vPlusZ - vMinusZ)/dz/2.0
            
            iGrad = np.array([xGrad,yGrad,zGrad])
            
            return iGrad
           
        def energy(grad=grad):
            
            size = len(pos)
            
            #first calculate the total energy
            e = calculate_e(index=None)
            
            #second calculate the gradient numerically
            gradient = np.zeros((size,3))
            
            if grad:
                for i in range(size):
                    iGrad = calculate_gradient(i)
                    gradient[i] += iGrad
                
            #third calculate the maximum force
            magList = np.sqrt(np.hstack(gradient)*np.hstack(gradient))
            maxForce = np.amax(magList)
            totalMag = np.linalg.norm(magList)
#            totalMag = np.linalg.norm(magList)/len(magList)**.5
            
            return e, gradient, maxForce, totalMag
            
        return energy

            
class Tersoff(Forcefield):
    
    def __init__(self, name="tersoff", energyUnits=0.043, lengthUnits=1.0):
        Forcefield.__init__(self,name,energyUnits,lengthUnits)
        
    def configure_parameters(self, molecule):
        pass
        
    def define_energy_routine(self,molecule,grad=True):
        
        atomList = molecule.atomList        
        
        #Tersoff potential terms
        #define some constants, specific to carbon
        A = 1.3936e3                 #eV
        B = 3.467e2                  #eV
        LAMBDA1 = 3.4879             #angstrom^-1
        LAMBDA2 = 2.2119             #angstrom^-1
        BETA = 1.5724e-7
        N = 0.72751
        C = 3.8049e4
        D = 4.384
        H = -5.7058e-1
        
        #values for calculating g(theta) terms
        c2 = C*C
        d2 = D*D
        g1 = 1 + (c2/d2)
        
        #values for calculating bij terms
        bijPower = -1/2/N
        
        def calculate_e(index):
            
            e = 0.0
            
            if index is None:
                eList = range(len(atomList))
            else:
                eList = atomList[index].nList[:]
                eList.append(index)
            
            for i in eList:
        
                atomi = atomList[i]
                posi = atomi.pos
                ineighbors = atomi.nList
                
                for j in ineighbors:
                    
                    atomj = atomList[j]
                    posj = atomj.pos
                    posij = posi - posj
                    rij = np.sqrt(posij.dot(posij))
                    #initialize zetaij
                    zetaij = 0.0
                    
                    for j2 in [x for x in ineighbors if x != j]:
                        
                        atomj2 = atomList[j2]
                        posj2 = atomj2.pos
                        
                        posij2 = posi - posj2
                        rij2 = np.sqrt(posij2.dot(posij2))       
                        cosTheta = posij.dot(posij2)/rij/rij2
                        gTerm = (H - cosTheta)*(H - cosTheta)
                        g = g1 - c2/(d2 + gTerm)
                        #increment zetaij by g
                        zetaij += g
                        
                    fR = A*np.exp(-LAMBDA1*rij)
                    fA = B*np.exp(-LAMBDA2*rij)
                    
                    #aij term
        #            aij = 1
                    
                    #bond angle term
                    bijTerm = 1 + np.power(BETA*zetaij, N)
                    bij = np.power(bijTerm, bijPower)
                    
                    e += fR - (bij*fA)
                
            e *= 0.5
            
            return e
            
            
            
        def calculate_gradient(index):
            atom = atomList[index]
            
            atom.shift(vdx)
            vPlusX = calculate_e(index)
            atom.shift(-2.0*vdx)
            vMinusX = calculate_e(index)
            atom.shift(vdx+vdy)
            vPlusY = calculate_e(index)
            atom.shift(-2.0*vdy)
            vMinusY = calculate_e(index)
            atom.shift(vdy+vdz)
            vPlusZ = calculate_e(index)
            atom.shift(-2.0*vdz)
            vMinusZ = calculate_e(index)
            atom.shift(vdz)
            
            xGrad = (vPlusX - vMinusX)/dx/2.0
            yGrad = (vPlusY - vMinusY)/dy/2.0
            zGrad = (vPlusZ - vMinusZ)/dz/2.0
            
            iGrad = np.array([xGrad,yGrad,zGrad])
            
            return iGrad
            
        def energy(grad=grad):
            
            #first calculate the total energy
            e = calculate_e(index=None)
            
            #second calculate the gradient numerically
            gradient = np.zeros([len(atomList),3])
            
            if grad:
                for i,atomi in enumerate(atomList):
                    iGrad = calculate_gradient(i)
                    gradient[i] += iGrad
                
            #third calculate the maximum force
            magList = np.sqrt(np.hstack(gradient)*np.hstack(gradient))
            maxForce = np.amax(magList)
            totalMag = np.linalg.norm(magList)
            
            return e, gradient, maxForce, totalMag
            
        return energy

forcefieldList = [Amber, Tersoff]
