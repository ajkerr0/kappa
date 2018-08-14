# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:52:49 2016

@author: Alex Kerr

Contains class definition(s) used to minimize the energy of Molecules
"""

import sys
from math import copysign

import numpy as np
import scipy.optimize

EP = sys.float_info.epsilon
SQRTEP = EP**.5
GR = 1.618
                                  
def minimize(mol, n=2500, descent="cg", search="backtrack", numgrad=False,
             eprec=1e-2, fprec=1e-2,
             efreq=1000, nbnfreq=15, print_=True):
    """Minimize the energy of the inputted molecule.
    
    Args:
        mol (Molecule): Molecule to be energy-minimized.
    Keywords:
        n (int): Max number of iterations in minimization run(s).
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
    
    if numgrad:
        grad_routine = mol.define_gradient_routine_numerical()
    else:
        grad_routine = mol.define_gradient_routine_analytical()
        
    return descentDict[descent](mol, n, searchDict[search], mol.define_energy_routine(), grad_routine,
                         efreq, nbnfreq, eprec*mol.ff.eunits, fprec*mol.ff.eunits/mol.ff.lunits,
                         print_=print_)
        
def steepest_descent(mol, n, search, calc_e, calc_grad, efreq, nbn, eprec, fprec):
    """Minimize the energy of the inputted molecule via the steepest descent approach."""
    
    #initial guess for stepsize
    stepSize = 1e-1
    
    energy = calc_e()
    gradient, maxForce, totalMag, = calc_grad()
    eList = [energy]
    print('energy:   %s' % energy)
    print('maxforce: %s' % maxForce)
    
    for step in range(1, n+1):
        
        #calculate the stepsize
        stepSize = search(mol, -gradient/totalMag, energy, gradient, calc_e, alpha=stepSize)
        
        #take the step
        mol.posList += stepSize*(-gradient/totalMag)
        
        #reset nonbonded neighbors
        if mol.ff.lj:
            if step % nbn == 0:
                mol._configure_nonbonded_neighbors()
        
        #reset quantities
        energy = calc_e()
        gradient, maxForce, totalMag = calc_grad()
        
        #for every multiple of efreq, print the status
        if step % efreq == 0:
            print('step:     %s' % step)
            print('energy:   %s' % energy)
            print('maxforce: %s' % maxForce)
            eList.append(energy)
            
        #break the iteration if our forces are small enough
        if maxForce < fprec:
            print('###########\n Finished! \n###########')
            print('step:     %s' % step)
            print('energy:   %s' % energy)
            print('maxforce: %s' % maxForce)
            break
    
    return mol, eList
    
def conjugate_gradient(mol, n, search, calc_e, calc_grad, efreq, nbn, eprec, fprec,
                       print_=True):
    """Minimize the energy of the inputted molecule via the conjugate gradient approach."""
    
    #initial guess for stepsize
    stepSize = 1e-1
    
    #starting values
    energy = calc_e()
    gradient, maxForce, totalMag, = calc_grad()
    eList = [energy]
    if print_:
        print('energy:   %s' % energy)
        print('maxforce: %s' % maxForce)
    
    gamma = 0.0
    prevH = np.zeros([len(mol), 3])
    
    for step in range(1, n+1):
        
        #get the step direction
        h = -gradient + gamma*prevH
        normH = h/np.linalg.norm(np.hstack(h))
        
        #calculate the stepsize
        stepSize = search(mol, normH, energy, gradient, calc_e, alpha=stepSize)
        
        #take the step
        mol.posList += stepSize*(normH)
        
        #reset nonbonded neighbors
        if mol.ff.lj:
            if step % nbn == 0:
                mol._configure_nonbonded_neighbors()
        
        #reset quantities
        prevH = h
        prevGrad = gradient
        energy = calc_e()
        gradient, maxForce, totalMag = calc_grad()
        gamma = calculate_gamma(gradient, prevGrad)
        
        #for every multiple of efreq, print the status
        if (step % efreq == 0) & print_:
            print('step:     %s' % step)
            print('energy:   %s' % energy)
            print('maxforce: %s' % maxForce)
            eList.append(energy)
            
        #break the iteration if our forces are small enough
        if maxForce < fprec:
            if print_:
                print('###########\n Finished! \n###########')
                print('step:     %s' % step)
                print('energy:   %s' % energy)
                print('maxforce: %s' % maxForce)
            break
    
    return mol, eList

def scipy_method(mol, n, search, calc_e, calc_grad, efreq, nbn, eprec, fprec,
                 print_=True):
    
    def func(pos):
        pos = pos.reshape(pos.shape[0]//3, 3)
        mol.posList = pos
        return calc_e()
    
    op_result = scipy.optimize.minimize(func, np.hstack(mol.posList), method='BFGS', options={'gtol':1e-8})
    pos = op_result.x
    pos = pos.reshape(pos.shape[0]//3, 3)
    mol.posList = pos
    return mol,None
    
def parabolic_interpolation(a,b,c,fa,fb,fc):
    
    m = (b-a)*(fb-fc)
    n = (b-c)*(fb-fa)
    return b - ((b-c)*n - (b-a)*m)/(2.*copysign(max(abs(n-m), EP), n-m))

def line_search_backtrack(mol, stepList, e, grad, calc_e, alpha=None):
    """Return the stepsize determined by the backtracking strategies of
    Armijo and Goldstein.
    
    Args:
        mol (Molecule): Molecule to be minimized.  mol.posList is the coordinate array of the atoms
        stepList (ndarray):  A N x 3 array of the step direction.
        e (float): The energy of the molecule before the step is taken.
        grad (ndarray): A N x 3 array of the forces on the atoms before the step is taken.
        calc_e (function): Callable function that returns the energy of the molecule
        alpha (float): An initial guess for the step size"""
    
    tau = 0.5
    c = 0.33
    alpha /= tau
    
    m = np.dot(np.hstack(stepList),np.hstack(grad))
    if m > 0.:
        raise ValueError("Step isn't a descent!")
        
    t = -c*m
    
    counter = 0
    while alpha > EP:
        
        mol.posList += alpha*stepList
        newE = calc_e()
        mol.posList += -alpha*stepList
        if e - newE >= alpha*t:
#            print('counter:  %s' % counter)
            return alpha
        else:
            alpha *= tau
            counter += 1
    
#    print('counter:  %s' % counter)        
    return EP
    
def line_search_brent(mol, stepList, e, grad, calc_e, alpha=None):
    """Return the stepsize determined by Brent's method
    
    Args:
        mol (Molecule): Molecule to be minimized.  mol.posList is the coordinate array of the atoms
        stepList (ndarray):  A N x 3 array of the step direction.
        e (float): The energy of the molecule before the step is taken.
        grad (ndarray): A N x 3 array of the forces on the atoms before the step is taken.
        calc_e (function): Callable function that returns the energy of the molecule"""
        
    a, b, c = bracket_minimum(mol, stepList, e, calc_e)
        
    return 0.
    
def bracket_minimum(mol, stepList, ea, calc_e):
    """Return a tuple of 3 points a,b,c such that
    f(a) > f(b) < f(c).  These points are stepsizes along the step direction
    along the potential energy surface of the molecule."""
    
    #a will be the zero point
    a = 0.
    
    #b will be some small distance away (we assume our step direction will minimize the energy within a finite range)
    b = a + SQRTEP
    mol.posList += b*stepList
    eb = calc_e()
    mol.posList += -b*stepList
    
    if eb > ea:
        raise ValueError("There was an error in bracketing the minimum!")
    
    #c will be found through an iterative process
    def ef(c):
        mol.posList += c*stepList
        e = calc_e()
        mol.posList += -c*stepList
        return e
        
    c = b + GR*(b-a)
    ec = ef(c)
    
    while ec <= eb:
        
        x = parabolic_interpolation(a,b,c,ea,eb,ec)
        xedge = b + 100.*(c-b)
        
        #if x is between b,c...
        if ((b-x)*(x-c)) > 0.:
            #check f(x)
            ex = ef(x)
            if ex < ec:
                a,b = b,x
                ea,eb = eb, ex
                continue
            elif ex > eb:
                c,ec = x,ex
                continue
            x = c + GR*(c-b)
            ex = ef(x)
            
        #if x is between c and our defined edge
        elif ((c-x)*(x-xedge)) > 0.:
            #check f(x)
            ex=ef(x)
            if ex < ec:
                b,c,x = c,x,c+GR*(c-b)
                eb,ec,ex = ec,ex,ef(x)
        
        #put x on the edge        
        elif ((x-xedge)*(xedge-c)) > 0.:
            x = xedge
            ex = ef(x)
         
        #just slide along the search direction
        else:
            x = c + GR*(c-b)
            ex = ef(x)
        
        a,b,c = b,c,x
        ea,eb,ec = eb,ec,ex
        
    return a, b, c
        
descentDict = {"sd":steepest_descent, "cg":conjugate_gradient, "scipy":scipy_method}
searchDict = {"backtrack":line_search_backtrack, "brent":line_search_brent}

def calculate_gamma(grad, pgrad):
    """Return the 'gamma' factor in the conjugate gradient method
    according to Fletcher and Reeves."""
    grad = np.hstack(grad)
    pgrad = np.hstack(pgrad)
    return (np.dot(grad,grad))/(np.dot(pgrad,pgrad))