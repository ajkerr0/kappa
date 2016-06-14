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
             cwd=os.path.join(dir_, os.path.join(param_dir,path)))

#Peter Lamut: http://blog.niteoweb.com/setuptools-run-custom-code-in-setup-py/
def path_dec(install_path):
    def customcmd(command_subclass):
        """A decorator for subclasses of the setuptools commands, 
        modifying the run() method so that it parses the force parameter
        files after installation."""
        
        orig_run = command_subclass.run
        
        def modified_run(self):
            orig_run(self)
            self.execute(parse, (getattr(self,install_path),),
                         msg="\nRunning the parameter parsers")
            
        command_subclass.run = modified_run
        return command_subclass
    return customcmd
        
@path_dec("install_lib")
class CustomInstallCommand(install):
    pass

@path_dec("egg_path")
class CustomDevelopCommand(develop):
    pass

setup(name='kappa',
      version='0.2.1',
      description='A package to calculate thermal conductivity in molecules',
      author='Alex Kerr',
      author_email='ajkerr0@gmail.com',
      url='https://github.com/ajkerr0/kappa',
      packages=find_packages(),
      package_data={
      'kappa.antechamber': ['*.txt'],
      'kappa.param': ['*/*.prm']
      },
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
      