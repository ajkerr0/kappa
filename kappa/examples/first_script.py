# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:35:00 2016

@author: Alex Kerr
"""

#import the package
import kappa

#create a forcefield object
#we'll use amber
amber = kappa.Amber()

#print the possible pre-made lattices
print(kappa.lattices)

#build a carbon nanotube
#assign our forcefield first
#use default radius and length parameters (we won't explicitly assign them)

cnt = kappa.build(ff=amber, lattice="cnt")

#print certain attributes of our cnt
#bondList
print(cnt.bondList)
#angleList
print(cnt.angleList)
#print the dihedralList
print(cnt.dihList)

#print the ff atomtypes
print(cnt.atomtypes)