"""

@author: Alex Kerr
"""

import numpy as np
import kappa

amber = kappa.Amber(dihs=False, angles=True, lengths=True)

# create stubby cnt
cnt = kappa.build(amber, "cnt", radius=1, length=20)

#kappa.plot.bonds(cnt, indices=True)
#kappa.plot.faces(cnt)
indices = [2, 57]

for i in range(1,51):
    
    # generate polyethylene of lunit length
    polyeth = kappa.build(amber, "polyeth", count=i)
    
    func_cnt = kappa.attach(cnt, [polyeth]*2, indices)
    type_list = np.zeros(len(func_cnt.posList))
    for j in range(len(cnt.posList)):
        for k in range(len(func_cnt.posList)):
            if j <= k:
                temp_chk = cnt.posList[j] == func_cnt.posList[k]
                if np.all(temp_chk, axis=0):
                    type_list[k] = 1     # part of CNT
                else:
                    type_list[k] = 2     # part of ends
    kappa.pdb(func_cnt, fn='cnt{}.pdb'.format(i))