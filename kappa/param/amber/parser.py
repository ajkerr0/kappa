# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:06:19 2015

@author: alex
"""

import os
import csv

import numpy as np

#SEE BELOW

def parse():
    
    #max tiny atom id
#    maxID = 560
    size = 50 #number of unique IDS
    size = size + 1
    
    #vdw arrays 
    #1d
    rvdw0 = np.zeros(size)
    epvdw = np.zeros(size)
    
    #bond length arrays
    #2d
    kr = np.zeros((size,size))
    r0 = np.zeros((size,size))
    
    #bond angle arrays
    #3d
    
    #SOLVE MEMORY PROBLEM
    kt = np.zeros((size,size,size))
    t0 = np.zeros((size, size, size))
    
    #torsion angle arrays
    #4d
    vn = np.zeros((size,size, size, size))
    gn = np.zeros((size,size, size, size))
    nn = np.zeros((size,size, size, size))
    
    #electric dipole terms
    #2d
    
    path = os.path.dirname(os.path.abspath(__file__))
    
    reader = csv.reader(open("%s/amber99.prm" % (path)), delimiter=" ", skipinitialspace=True)
    
    for line in reader:
        
        if line:  #if line is not empty
        
            first_string = line[0]
            
            if first_string == 'vdw':
                
                row = int(line[1])
                
                rvdw0[row] = float(line[2])
                epvdw[row] = float(line[3])
                
            elif first_string == 'bond':
                
                row = int(line[1])
                col = int(line[2])
                
                kr[row,col] = float(line[3])
                kr[col,row] = float(line[3])
                r0[row,col] = float(line[4])
                r0[col,row] = float(line[4])
                
            elif first_string == 'angle':
                
                row = int(line[1])
                col = int(line[2])
                lay = int(line[3]) #layer
                
                kt[row,col,lay] = float(line[4])
                kt[lay,col,row] = float(line[4])
                t0[row,col,lay] = float(line[5])
                t0[lay,col,row] = float(line[5])
                
            elif first_string == 'torsion':
                
                row = int(line[1])
                col = int(line[2])
                lay = int(line[3])
                hyp = int(line[4])
                
                vn[row,col,lay,hyp] = float(line[5])
                vn[hyp,lay,col,row] = float(line[5])
                gn[row,col,lay,hyp] = float(line[6])
                gn[hyp,lay,col,row] = float(line[6])
                nn[row,col,lay,hyp] = float(line[7])
                nn[hyp,lay,col,row] = float(line[7])
                    
                
            elif first_string == 'charge':
                
                #to be finishes
                pass
                
            else:
                pass
        
    #save matrices
    
    #vdw arrays
    np.save("rvdw0", rvdw0)
    np.save("epvdw", epvdw)
    
    #bond length arrays
    np.save("kr", kr)
    np.save("r0", r0)
    
    #bond angle arrays
    np.save("kt", kt)
    np.save("t0", t0)
    
    #torsion arrays    
    np.save("vn", vn)
    np.save("gn", gn)
    np.save("nn", nn)
        
parse()