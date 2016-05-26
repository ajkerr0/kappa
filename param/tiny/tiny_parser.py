# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:06:19 2015

@author: alex
"""

import csv
import numpy as np

#SEE BELOW

def parse_tiny():
    
    #max tiny atom id
#    maxID = 560
    size = 46 #number of unique IDS
    
    #vdw arrays 
    #1d
    vdwR = np.zeros(size)
    vdwEp = np.zeros(size)
    
    #bond length arrays
    #2d
    kL0 = np.zeros((size,size))
    rL0 = np.zeros((size,size))
    
    #bond angle arrays
    #3d
    
    #SOLVE MEMORY PROBLEM
    kA0 = np.zeros((size,size,size))
    theta0 = np.zeros((size, size, size))
    
    #torsion angle arrays
    #2d because only 2 inner IDs matter
    Vtors = np.zeros((size,size))
    gammators = np.zeros((size,size))
    ntors = np.zeros((size,size))
    
    #electric dipole terms
    #2d
    

    reader = csv.reader(open("./tiny.prm"), delimiter=" ", skipinitialspace=True)
    
    for line in reader:
        
        if line:  #if line is not empty
        
            cat = line[0]
            
            if cat == 'vdw':
                
                row = ref(int(line[1]))
                
                vdwR[row] = float(line[2])
                vdwEp[row] = float(line[3])
                
            elif cat == 'bond':
                
                row = ref(int(line[1]))
                col = ref(int(line[2]))
                
                kL0[row,col] = float(line[3])
                kL0[col,row] = float(line[3])
                rL0[row,col] = float(line[4])
                rL0[col,row] = float(line[4])
                
            elif cat == 'angle':
                
                row = ref(int(line[1]))
                col = ref(int(line[2]))
                lay = ref(int(line[3])) #layer
                
                kA0[row,col,lay] = float(line[4])
                kA0[lay,col,row] = float(line[4])
                theta0[row,col,lay] = float(line[5])
                theta0[lay,col,row] = float(line[5])
                
            elif cat == 'torsion':
                
                #check to see if there are actually parameters to store
                if len(line) > 5:
                    
                    row = ref(int(line[2]))
                    col = ref(int(line[3]))
                    
                    Vtors[row,col] = float(line[5])
                    Vtors[col,row] = float(line[5])
                    gammators[row,col] = float(line[6])
                    gammators[col,row] = float(line[6])
                    ntors[row,col] = float(line[7])
                    ntors[col,row] = float(line[7])
                    
                else:
                    pass
                
            elif cat == 'dipole':
                
                #figure out dipole parameters
                pass
                
            else:
                pass
            
        else:
            pass
        
    #save matrices
    
    #vdw arrays
    np.save("vdwR", vdwR)
    np.save("vdwE", vdwEp)
    
    #bond length arrays
    np.save("kb", kL0)
    np.save("rb0", rL0)
    
    #bond angle arrays
    np.save("ka", kA0)
    np.save("theta0", theta0)
    
    #torsion arrays    
    np.save("vtors", Vtors)
    np.save("gammators", gammators)
    np.save("ntors", ntors)
    
    
    
def tinyID_ref():
    "Create a reference array for aliasing parameters"
    
    reader = csv.reader(open("./tiny.prm"), delimiter=" ", skipinitialspace=True)
            
    keyList = []
    for count,line in enumerate(reader):
        if line:    #if line is not empty    
            if line[0] == 'atom':
                keyList.append(int(line[1]))
            else:
                pass  
        else:
            pass
    
    tinyRef = np.zeros(keyList[-1]+1, dtype=np.int8)    
    
    for index, key in enumerate(keyList):
        tinyRef[key] = index
#    print tinyRef
        
    np.save("refArr",tinyRef)
    
def ref(key):
    "Return the corresponding ID for use in tiny parameter arrays"
    
    tinyRef = np.load("refArr.npy")
    return tinyRef[key]
    
        
#need to run tinyID_ref first in order to save the reference array
        
tinyID_ref()
parse_tiny()