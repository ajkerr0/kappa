# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:04:20 2016

@author: alex
"""

import kappa

amber = kappa.Amber(dihs=False)

am = kappa.build(ff=amber, lattice="imine")

kappa.save_file(kappa.top(am), ".", "top_test")