"""
A module that parses the GAFF parameter file.

@author: Alex Kerr
"""

from itertools import islice
import csv

file_ = './gaff.dat'

def parse():
    
    lines = list(csv.reader(open(file_)))
    
    #indices of entries
    atom_type = (1,71)
    bond = (74, 905)
    angle = (907, 5524)
    dihedral = (5526, 6269)
    imptor = (6271, 6308)
    
    #give each atom type an index
    i,j = atom_type
    atom_types = []
    for line in lines[i:j+1]:
        atom_types.append(str(line[0][:2]).rstrip())
        
    print(atom_types)
    
parse()