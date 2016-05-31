# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:09:01 2016

@author: alex
"""

# Make selected functionality available directly in the root namespace
#available = [('molecule', ['build', 'lattices']),
#             ('forcefield', ['Amber']),
#             ('plot', ['bonds']),
#             ('operation', ['save', 'load', 'hessian', 'evecs', 'chain'])]
#             
#for module, names in available:
#    exec('from .%s import %s' % (module, ', '.join(names)))
import os

#russ on stackoverflow
package_dir = os.path.dirname(__file__)

from .molecule import *
from .forcefield import *
from .operation import *
from .minimize import minimize_energy
from .antechamber.atomtype import main as atomtype
import plot