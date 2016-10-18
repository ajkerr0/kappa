"""
An ugly parser of ff parameter files.

@author: Alex Kerr
"""

import csv

import numpy as np

def parse():
    
    file_ = './parm99.dat'
    
    lines = list(csv.reader(open(file_),))
    
    #find indices of blank lines
    section = []
    for count,line in enumerate(lines):
        if not line:
            section.append(count)
    
    #retrieve atom types
    a_types = []
    for line in lines[1:section[0]]:
        a_types.append(str(line[0][:2]).rstrip())
    np.save("atomtypes", a_types)
        
    dim = len(a_types)
        
    #bonds
    bond_types = []
    kb = []
    b0 = []
    for line in lines[section[0]+2:section[1]]:
        line=line[0]
        a,b = line[:2].rstrip(), line[3:5].rstrip()
        bond_types.append([a,b])
        kb.append(float(line[6:12]))
        b0.append(float(line[15:22]))
    bond_types = np.array([(a_types.index(a), a_types.index(b))
                            for a,b in bond_types])
    kbArr = np.zeros((dim,dim,2))
    kbArr[bond_types[:,0], bond_types[:,1],0] = np.array(kb)
    kbArr[bond_types[:,1], bond_types[:,0],0] = np.array(kb)
    kbArr[bond_types[:,0], bond_types[:,1],1] = np.array(b0)
    kbArr[bond_types[:,1], bond_types[:,0],1] = np.array(b0)
    np.save("blengths", kbArr)
      
    #angles
    angle_types = []
    kt = []
    t0 = []
    for line in lines[section[1]+1:section[2]]:
        line=line[0]
        a,b,c = map(str.rstrip, [line[:2],line[3:5],line[6:8]])
        angle_types.append([a,b,c])
        kt.append(float(line[10:16]))
        t0.append(float(line[21:28]))
    angle_types = np.array([(a_types.index(a), a_types.index(b), a_types.index(c))
                            for a,b,c in angle_types])
    ktArr = np.zeros((dim,dim,dim,2))
    ktArr[angle_types[:,0], angle_types[:,1], angle_types[:,2],0] = np.array(kt)
    ktArr[angle_types[:,2], angle_types[:,1], angle_types[:,0],0] = np.array(kt)
    ktArr[angle_types[:,0], angle_types[:,1], angle_types[:,2],1] = np.array(t0)
    ktArr[angle_types[:,2], angle_types[:,1], angle_types[:,0],1] = np.array(t0)
    np.save("bangles", ktArr)
    
    #dihedrals
    #dihedrals are different because we must handle wildcard atoms
    #we must also use every term in the Fourier series
    wc = 'X'
    dih_types = []
    vns = []
    
    for line in lines[section[2]+1:section[3]]:
        line=line[0]
        a,b,c,d = map(str.rstrip, [line[:2],line[3:5],line[6:8],line[9:11]])
        dih_types.append([a,b,c,d])
        vns.append([float(line[18:24]),float(line[31:38]),abs(int(float(line[49:54])))])
        
#    print(vns)
    
    vnArr = np.zeros((dim,dim,dim,dim,4,2))
    
    slices = [[slice(None) if i == wc else a_types.index(i) for i in dih] for dih in dih_types]
    
    for slice_, vn in zip(slices,vns):
        
        #first make sure the array is cleared from wildcards
#        print(slice_)
        vnArr[slice_+[vn[-1]]] = 0.
#        vnArr[slice_+[vn[-1], 0]] = vn[0]
#        vnArr[slice_+[vn[-1], 1]] = vn[1]
        vnArr[slice_+[vn[-1]]][0] = vn[0]
        vnArr[slice_+[vn[-1]]][1] = vn[1]
        slice_.reverse()
        vnArr[slice_+[vn[-1]]] = 0.
#        vnArr[slice_+[vn[-1], 0]] = vn[0]
#        vnArr[slice_+[vn[-1], 1]] = vn[1]
        vnArr[slice_+[vn[-1]]][0] = vn[0]
        vnArr[slice_+[vn[-1]]][1] = vn[1]
#        print(slice_)
        
    np.save("bdih", vnArr)

if __name__ == "__main__":
    parse()