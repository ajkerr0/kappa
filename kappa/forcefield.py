# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:16:37 2016

@author: alex
"""

import numpy as np

from . import package_dir

#forcefield class definitions
global_cutoff = 5.0 #angstroms

class Forcefield:
    
    def __init__(self, name, energyUnits, lengthUnits,
                 lengths, angles, dihs, lj, es, tersoff):
        self.name = name
        self.eUnits = energyUnits    #relative to kcal/mol
        self.lUnits = lengthUnits    #relative to angstroms
        ##########
        self.lengths = lengths       #bond length interaction
        self.angles = angles         #bond angle interaction
        self.dihs = dihs             #dihedral angle interaction
        self.lj = lj                 #lennard-jones, non-bonded interaction
        self.es = es                 #electrostatic (point charge), non-bonded interaction
        self.tersoff = tersoff       #tersoff interaction        
        
class Amber(Forcefield):
    
    def __init__(self, lengths=True, angles=True, dihs=True, lj=False):
        Forcefield.__init__(self, "amber", 1.0, 1.0,
                            lengths, angles, dihs, lj, False, False)
        self.atomTypeIDDict = {"CT":1, "C":2,  "CA":3,  "CM":4, "CC":5,  "CV":6, "CW":7, "CR":8, "CB":9, "C*":10, "CZ":3,
                               "CN":11,"CK":12,"CQ":13, "N":14, "NA":15, "NB":16,"NC":17,"N*":18,"N2":19,"N3":20, "NT":19,
                               "OW":21,"OH":22,"OS":23, "O":24, "O2":25, "S":26, "SH":27,"P":28, "H":29, "HW":30, 
                               "HO":31,"HS":32,"HA":33, "HC":34,"H1":35, "H2":36,"H3":37,"HP":38,"H4":39,"HS":40,
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

forcefieldList = [Amber, Tersoff]
