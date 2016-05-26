# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:19:51 2016

@author: alex
"""

from distutils.core import setup
from setuptools import find_packages

setup(name='kappa',
      version='0.2.1',
      description='A package to calculate thermal conductivity in molecules',
      author='Alex Kerr',
      author_email='ajkerr0@gmail.com',
      url='https://github.com/ajkerr0/kappa',
      packages=find_packages(),
      )