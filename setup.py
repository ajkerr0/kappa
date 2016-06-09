# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:19:51 2016

@author: Alex Kerr
"""
import os,sys
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.develop import develop

def parse(dir_):
    from subprocess import call
    #get path to interpreter
    interpreter = sys.executable
    #get param directories
    param_dir = 'kappa/param/'
    parser_paths = [path for path in os.listdir(param_dir) if os.path.isdir(os.path.join(param_dir, path))]
    for path in parser_paths:
        call([interpreter, "parser.py"],
             cwd=os.path.join(dir_, path))

#Peter Lamut: http://blog.niteoweb.com/setuptools-run-custom-code-in-setup-py/
def customcmd(command_subclass):
    """A decorator for subclasses of the setuptools commands, 
    modifying the run() method so that it parses the force parameter
    files after installation."""
    
    orig_run = command_subclass.run
    
    def modified_run(self):
        orig_run(self)
        self.execute(parse, (self.install_lib,))
        
    command_subclass.run = modified_run
    return command_subclass
        
@customcmd
class CustomInstallCommand(install):
    pass

@customcmd
class CustomDevelopCommand(develop):
    pass

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
      'scipy',
      ],
      cmdclass={
      'install': CustomInstallCommand,
      'develop': CustomDevelopCommand,
      },
      )

