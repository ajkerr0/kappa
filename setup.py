# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:19:51 2016

@author: Alex Kerr
"""

from setuptools import find_packages, setup

setup(name='kappa',
      version='0.2.1',
      description='A package to calculate thermal conductivity in molecules',
      author='Alex Kerr',
      author_email='ajkerr0@gmail.com',
      url='https://github.com/ajkerr0/kappa',
      packages=find_packages(),
      install_requires=[
      'numpy',
      'matplotlib',
      'scipy'
      ],
      )

