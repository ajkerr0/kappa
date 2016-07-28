# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:52:49 2016

@author: Alex Kerr

Contains class definition(s) used to minimize the energy of Molecules
"""

import numpy as np

class Minimizer:
    """A energy minimizer for Molecules
    
    Args:
        n (int): Max number of iterations in minimization run(s).
    Keywords:
        descent (str): Method in which the local energy minimum will be approached;
            must be key in descentDict
        search (str): Line search method; must be key in searchDict
        numgrad (bool): True if the gradients are to be calculated numerically;
            default is False
        eprec (float): Precision for energy changes when an iteration is made 
            signalling convergence has occurred
        fprec (float): Precision for force magnitude signaling
            convergence has occurred.
        efreq (int): Iteration period in which information is printed to the user;
            called frequency despite being inverse frequency."""
    
    def __init__(self, n, descent="cg", search="backtrack", 
                 numgrad=False, eprec=1e-5, fprec=1e-3,
                 efreq=1):
        self.n = n
        self.descent = descent
        self.search = search
        self.numgrad = numgrad
        self.eprec = eprec
        self.fprec = fprec
        self.efreq =  efreq
        
    def __call__(self, molecule):
        if self.numgrad:
            grad_routine = molecule.define_gradient_routine()
        else:
            grad_routine = molecule.define_gradient_routine()
        descentDict[self.descent](molecule, self.n, searchDict[self.search], molecule.define_energy_routine(),
                                  grad_routine, self.efreq,
                                  self.eprec*molecule.ff.eunits, self.fprec*molecule.ff.eunits/molecule.ff.lunits)
        
def steepest_descent(mol, n, search, calc_e, calc_grad, efreq, eprec, fprec):
    """Minimize the energy of the inputted molecule via the steepest descent approach."""
    
    energy = calc_e()
    gradient, maxForce, totalMag, = calc_grad()
    eList = [energy]
    print('energy:   $s' % energy)
    print('maxforce: $s' % maxForce)
    
    for step in range(1, n+1):
        
        #calculate the stepsize
        stepSize = search(mol, -gradient/totalMag, energy)
        
        #take the step
        mol.posList += stepSize*(-gradient/totalMag)
        
        #reset quantities
        energy = calc_e()
        gradient, maxForce, totalMag = calc_grad()
        
        eList.append(energy)
        #for every multiple of efreq, print the status
        if step % efreq == 0:
            print('step:     %s' % step)
            print('energy:   %s' % energy)
            print('maxforce: %s' % maxForce)
            
        #break the iteration if our forces are small enough
        if maxForce < fprec:
            print('#########\nFinished!\n#########')
            print('step:     %s' % step)
            print('energy:   %s' % energy)
            print('maxforce: %s' % maxForce)
            break
    
    return mol, eList
    
def conjugate_gradient(mol, n, search, calc_e, calc_grad, efreq, eprec, fprec):
    """Minimize the energy of the inputted molecule via the conjugate gradient approach."""
    
    energy = calc_e()
    gradient, maxForce, totalMag, = calc_grad()
    eList = [energy]
    print('energy:   $s' % energy)
    print('maxforce: $s' % maxForce)
    
    gamma = 0.0
    prevH = np.zeros([len(mol), 3])
    
    for step in range(1, n+1):
        
        #get the step direction
        h = -gradient + gamma*prevH
        normH = h/np.linalg.norm(np.hstack(h))
        
        #calculate the stepsize
        stepSize = search(mol, normH, energy)
        
        #take the step
        mol.posList += stepSize*(-gradient/totalMag)
        
        #reset quantities
        prevH = h
        prevGrad = gradient
        energy = calc_e()
        gradient, maxForce, totalMag = calc_grad()
        gamma = calculate_gamma(gradient, prevGrad)
        
        eList.append(energy)
        #for every multiple of efreq, print the status
        if step % efreq == 0:
            print('step:     %s' % step)
            print('energy:   %s' % energy)
            print('maxforce: %s' % maxForce)
            
        #break the iteration if our forces are small enough
        if maxForce < fprec:
            print('#########\nFinished!\n#########')
            print('step:     %s' % step)
            print('energy:   %s' % energy)
            print('maxforce: %s' % maxForce)
            break
    
    return mol, eList

def line_search_backtrack(mol, stepList, e, grad, calc_e, calc_grad):
    """Return the stepsize determined by the backtracking strategies of
    Armijo and Goldstein."""
    
    n = 25
    alpha = 1e-2
    tau = 0.5
    c = 0.5
    
    for j in range(n):
        
        m = np.dot(np.hstack(stepList),np.hstack(grad))
        t = -c*m
        mol.posList += alpha*stepList
        newE = calc_e()
        if e - newE >= alpha*t:
            return alpha
        else:
            mol.posList += -alpha*stepList
            e, grad = newE, calc_grad()
            alpha *= tau        
        
descentDict = {"sd":steepest_descent, "cg":conjugate_gradient}
searchDict = {"backtrack":line_search_backtrack}

def calculate_gamma(grad, pgrad):
    """Return the 'gamma' factor in the conjugate gradient method."""
    grad = np.hstack(grad)
    pgrad = np.hstack(pgrad)
#    return (np.dot(grad-pGrad,grad))/(np.dot(pGrad,pGrad))
    return (np.dot(grad,grad))/(np.dot(pgrad,pgrad))