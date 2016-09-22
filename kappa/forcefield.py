# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:16:37 2016

@author: Alex Kerr

Define general Forcefield class, and specific forcefields (AMBER, etc.) that inherit
the general one.
"""

import numpy as np

#forcefield class definitions
global_cutoff = 5.0 #angstroms

class Forcefield:
    """A classical forcefield that determines how atoms interact
    
    Args:
        name (str): Human readable string that identifies the forcefield.
        eunits (float): The units of energy used in the ff, relative to kcal/mol.
        lunits (float): The units of length used in the ff, relative to angstroms
        lengths (bool): Boolean that determines if bond length interactions exist,
            that is energy that is quadratic in the bond lengths.
        angles (bool): Boolean that determines if bond angle interactions exist,
            energy that is quadraic in the bond angles.
        dihs (bool): Determines dihedral angle interactions,
            energy is an effective Fourier series of the angle(s).
        lj (bool): Determines Lennard-Jones non-bonded interactions.
        es (bool): Determines electrostatic point charge interactions.
        tersoff (bool): Determines Tersoff-type interactions."""
    
    def __init__(self, name, eunits, lunits,
                 lengths, angles, dihs, lj, es, tersoff):
        self.name = name
        self.eunits = eunits    #relative to kcal/mol
        self.lunits = lunits    #relative to angstroms
        ##########
        self.lengths = lengths       #bond length interaction
        self.angles = angles         #bond angle interaction
        self.dihs = dihs             #dihedral angle interaction
        self.lj = lj                 #lennard-jones, non-bonded interaction
        self.es = es                 #electrostatic (point charge), non-bonded interaction
        self.tersoff = tersoff       #tersoff interaction
        
class Amber(Forcefield):
    """Amber forcefield inheriting from Forcefield,
    as presented by Cornell et al. (1994)"""
    
    def __init__(self, lengths=True, angles=False, dihs=False, lj=False):
        super().__init__("amber", 1.0, 1.0,
                         lengths, angles, dihs, lj, False, False)
        self.atomTypeIDDict = {"CT":1, "C":2,  "CA":3,  "CM":4, "CC":5,  "CV":6, "CW":7, "CR":8, "CB":9, "C*":10, "CZ":3,
                               "CN":11,"CK":12,"CQ":13, "N":14, "NA":15, "NB":16,"NC":17,"N*":18,"N2":19,"N3":20, "NT":19,
                               "OW":21,"OH":22,"OS":23, "O":24, "O2":25, "S":26, "SH":27, "SO":26,"P":28, "H":29, "HW":30, 
                               "HO":31,"HS":32,"HA":33, "HC":34,"H1":35, "H2":36,"H3":37,"HP":38,"H4":39,"HS":40, "Cl":34,  #edit
                               "DU":1}
        self.atomtypeFile = "ATOMTYPE_AMBER_KERR.DEF"
            
class Tersoff(Forcefield):
    """Under construction, a remnant of code past."""
    
    def __init__(self, name="tersoff", energyUnits=0.043, lengthUnits=1.0):
        super().__init__(self, "tersoff", 0.0433634, 1.0)
        
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
