# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:06:19 2015

@author: alex
"""

import csv
import numpy as np

#SEE BELOW

def parse_amber():
    
    #max tiny atom id
#    maxID = 560
    size = 50 #number of unique IDS
    size = size + 1
    
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
    #4d
    Vtors = np.zeros((size,size, size, size))
    gammators = np.zeros((size,size, size, size))
    ntors = np.zeros((size,size, size, size))
    
    #electric dipole terms
    #2d
    

    reader = csv.reader(open("./amber99.prm"), delimiter=" ", skipinitialspace=True)
    
    for line in reader:
        
        if line:  #if line is not empty
        
            first_string = line[0]
            
            if first_string == 'vdw':
                
                row = ref(int(line[1]))
                
                vdwR[row] = float(line[2])
                vdwEp[row] = float(line[3])
                
            elif first_string == 'bond':
                
                row = ref(int(line[1]))
                col = ref(int(line[2]))
                
                kL0[row,col] = float(line[3])
                kL0[col,row] = float(line[3])
                rL0[row,col] = float(line[4])
                rL0[col,row] = float(line[4])
                
            elif first_string == 'angle':
                
                row = ref(int(line[1]))
                col = ref(int(line[2]))
                lay = ref(int(line[3])) #layer
                
                kA0[row,col,lay] = float(line[4])
                kA0[lay,col,row] = float(line[4])
                theta0[row,col,lay] = float(line[5])
                theta0[lay,col,row] = float(line[5])
                
            elif first_string == 'torsion':
                
                #check to see if there are actually parameters to store
                if len(line) > 5:
                    row = ref(int(line[1]))
                    col = ref(int(line[2]))
                    lay = ref(int(line[3]))
                    hyp = ref(int(line[4]))
                    
                    Vtors[row,col,lay,hyp] = float(line[5])
                    Vtors[hyp,lay,col,row] = float(line[5])
                    gammators[row,col,lay,hyp] = float(line[6])
                    gammators[hyp,lay,col,row] = float(line[6])
                    ntors[row,col,lay,hyp] = float(line[7])
                    ntors[hyp,lay,col,row] = float(line[7])
                    
                else:
                    pass
                
            elif first_string == 'dipole':
                
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
    
    
    
def amberID_ref():
    "Create a reference array for aliasing parameters"
    
#    reader = csv.reader(open("./amber99.prm"), delimiter=" ", skipinitialspace=True)
#            
#    keyList = []
#    for count,line in enumerate(reader):
#        if line:    #if line is not empty    
#            if line[0] == 'atom':
#                keyList.append(int(line[1]))
#            else:
#                pass  
#        else:
#            pass
#    
#    amberRef = np.zeros(keyList[-1]+1)    
#    
#    for index, key in enumerate(keyList):
#        amberRef[key] = index
    amberRef = range(51)
#    print amberRef
        
    np.save("refArr",amberRef)
    
def ref(key):
    "Return the corresponding ID for use in tiny parameter arrays"
    
    amberRef = np.load("refArr.npy")
    return amberRef[key]
    
        
#need to run tinyID_ref first in order to save the reference array
        
amberID_ref()
parse_amber()