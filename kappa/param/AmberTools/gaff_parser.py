"""
An ugly parser of ff parameter files.

@author: Alex Kerr
"""

import csv

import numpy as np

file_ = './gaff.dat'

def parse():
    
    lines = list(csv.reader(open(file_),))
    
    #find indices of blank lines
    section = []
    for count,line in enumerate(lines):
        if not line:
            section.append(count)
            
    print(lines)    
    
    #retrieve atom types
    atom_types = []
    for line in lines[1:section[0]]:
        atom_types.append(str(line[0][:2]).rstrip())
        
    #retrieve bonds
    bond_types = []
    kb = []
    b0 = []
    
    for line in lines[section[0]+2:section[1]]:
        line=line[0]
        a,b = line[:2].rstrip(), line[3:5].rstrip()
        bond_types.append([a,b])
        kb.append(float(line[6:12]))
        b0.append(float(line[15:22]))
        
    angle_types = []
    kt = []
    t0 = []
        
    for line in lines[section[1]+1:section[2]]:
        line=line[0]
        a,b,c = map(str.rstrip, [line[:2],line[3:5],line[6:8]])
        angle_types.append([a,b,c])
        kt.append(float(line[10:16]))
        t0.append(float(line[21:28]))
        
    dih_types = []
    vn = []
    nn = []
    gn = []
    
    for line in lines[section[2]+1:section[3]]:
        line=line[0]
        a,b,c,d = map(str.rstrip, [line[:2],line[3:5],line[6:8],line[9:11]])
        dih_types.append([a,b,c,d])
        vn.append(float(line[18:24]))
        gn.append(float(line[31:38]))
        nn.append(abs(int(float(line[49:54]))))
        
    #the ugly way to do this
    dim = len(atom_types)
    
    bond_types = np.array([(atom_types.index(a), atom_types.index(b))
                            for a,b in bond_types])
                                
    kbArr = np.zeros((dim,dim,2))
    kbArr[bond_types[:,0], bond_types[:,1],0] = np.array(kb)
    kbArr[bond_types[:,1], bond_types[:,0],0] = np.array(kb)
    kbArr[bond_types[:,0], bond_types[:,1],1] = np.array(b0)
    kbArr[bond_types[:,1], bond_types[:,0],1] = np.array(b0)
    
    print(kbArr[51,29])
    print(kbArr[29,51])
                                
    print(bond_types)
    
#    def find(a,b):
#        return atom_types.index(a), atom_types.index(b)
#    
#    bond_types = starmap(find, bond_types)
#    print(bond_types)
    
        
#    print(atom_types)
#    print(bond_types)
#    print(kb)
#    print(b0)
#    print(angle_types)
#    print(kt)
#    print(t0)
#    print(vn)
#    print(nn)
#    print(gn)
    
#    bond_types = np.array(bond_types, dtype="str")

    
    
parse()