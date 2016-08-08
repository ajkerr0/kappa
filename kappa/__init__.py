# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:09:01 2016

@author: Alex Kerr

Bring package functionality to user's script namespace when user imports the package
"""

import os

#russ on stackoverflow
package_dir = os.path.dirname(os.path.abspath(__file__))

#this method may be frowned upon, please someone educate me
from .molecule import *
from .forcefield import *
from .conductivity import Calculation
from .operation import *
from ._minimize import minimize
from .antechamber.atomtype import main as atomtype
from . import plot
from .md.generate import *