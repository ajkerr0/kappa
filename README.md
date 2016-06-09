# kappa
A python package to calculate thermal conductivity across molecular interfaces.

### Goal
Provide the user the tools to build virtual molecules and chemically functionalize them with the hope of overcoming their Kapitza resistance.  Thermal conductivities will be calculated using the Green's function method detailed in:

[Abdellah Ait Moussa and Kieran Mullen, "Using normal modes to calculate and optimize thermal conductivity in functionalized macromolecules," Phys. Rev. E 83, 056708 (2011).](http://journals.aps.org/pre/abstract/10.1103/PhysRevE.83.056708)

We also want to give users ways to export their molecules as input for more robust codes, particularly GROMACS.

### Requirements
kappa requires the numpy, scipy, and matplotlib packages.

### Install
Enter directory of the actual package:

`$ cd PATH/TO/kappa`

~~Install kappa in your python location:~~

~~`$ python setup.py install`~~

~~If you wish to develop the code yourself:~~

`$ python setup.py develop`

We plan on submitting this code to the Python Package Index (PyPI) under a different name.
