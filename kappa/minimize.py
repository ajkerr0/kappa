# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:52:49 2016

@author: alex
"""

import numpy as np
from copy import deepcopy

GOLD = 1.618034
GLIMIT = 100.0
TINY = 1e-20

n = 1000
forcePrec = 0.001 #kcal/mol/angstroms
stepPrec = 1e-18
eFreq = 100

def minimize_energy(molecule):
    """Minimize the energy of a given molecule within a certain precision"""
    
    print "Preparing " + molecule.name + " for energy minimization..."
    
    calculate_energy = molecule.ff.define_energy_routine(molecule)
    
#    minimizer = steepest_descent
    minimizer = conjugate_gradient
    
    minimizer(molecule, calculate_energy)
    
def steepest_descent(molecule, calculate_energy):
    """Return minimized molecule using steepest descent method."""
    
    prec = forcePrec/molecule.ff.eUnits
    energy, gradient, maxForce, totalMag = calculate_energy()
    eList = [energy]
    print 'energy: ' + str(energy)
    print 'maxforce: ' + str(maxForce)

    for stepNo in range(n):
        
        #calculate the stepsize
        stepSize = calculate_stepsize(molecule, -gradient/totalMag, energy)
        
        #take step
        molecule.posList += stepSize*-gradient/totalMag        
        
        energy, gradient, maxForce, totalMag = calculate_energy()
        
        eList.append(energy)
        #for every multiple of eFreq print status
        if (stepNo+1) % eFreq == 0:
            print 'step: ' + str(stepNo+1)
            print 'energy: ' + str(energy)
            print 'maxforce: ' + str(maxForce)
        else:
            pass
        
        #break the iteration if your forces are small enough
        if maxForce < prec:
            print "Finished!"
            print 'step: ' + str(stepNo+1)
            print 'energy: ' + str(energy)
            print 'maxforce: ' + str(maxForce)
            break
        else:
            pass

    return molecule, eList
    
def conjugate_gradient(molecule, calculate_energy):
    
    prec = forcePrec/molecule.ff.eUnits
    energy, gradient, maxForce, totalMag = calculate_energy()
    eList = [energy]
    print 'energy: ' + str(energy)
    print 'maxforce: ' + str(maxForce)
    
    gamma = 0.0
    prevH = np.zeros((len(molecule.posList),3))
    
    for stepNo in range(n):
        
        #step direction
        h = -gradient + gamma*prevH
        normH = h/np.linalg.norm(np.hstack(h))
        
        #calculate the stepsize
        stepSize = calculate_stepsize(molecule, normH, energy)
        
        prevH = h
        prevGrad = gradient
        
        #take step
        molecule.posList += stepSize*normH
        
        energy, gradient, maxForce, totalMag = calculate_energy()
        
        eList.append(energy)
        #for every multiple of eFreq print status
        if (stepNo+1) % eFreq == 0:
            print 'step: ' + str(stepNo+1)
            print 'energy: ' + str(energy)
            print 'maxforce: ' + str(maxForce)
        else:
            pass
        
        #break the iteration if your forces are small enough
        if maxForce < prec or stepSize < stepPrec:
#        if maxForce < prec:
            print "Finished!"
            print 'step: ' + str(stepNo+1)
            print 'energy: ' + str(energy)
            print 'maxforce: ' + str(maxForce)
            break
        
        gamma = calculate_gamma(gradient,prevGrad)

    return molecule, eList
    
def min_parabola(a,b,c,fa,fb,fc):
    r = (b-a)*(fb-fc)
    q = (b-c)*(fb-fa)
    return b - .5*((b-a)*r - (b-c)*q)/max(abs(r - q),TINY)*np.sign(r-q)
#    print a,b,c,fa,fb,fc
#    return b - .5*((b-a)*r - (b-c)*q)/(r-q)
    
def calculate_gamma(grad,pGrad):
    grad = np.hstack(grad)
    pGrad = np.hstack(pGrad)
#    return (np.dot(grad-pGrad,grad))/(np.dot(pGrad,pGrad))
    return (np.dot(grad,grad))/(np.dot(pGrad,pGrad))
    
def calculate_stepsize(molecule, stepList, e):
    
    n = 3
    
    testMolecule = deepcopy(molecule)
    calculate_energy = testMolecule.ff.define_energy_routine(testMolecule,grad=False)
    a,b,c,va,vb,vc = initial_bracket(testMolecule, stepList, e)
    if a is False:
        print a
#        print 'false'
        return 0.0
    step = min_parabola(a,b,c,va,vb,vc)
    
    for i in range(n):
        if step < b:
            c = b
            vc = vb
        else:
            a = b
            va = vb
        b = step
        testMolecule.posList += b*stepList
        vb,_,_,_ = calculate_energy()
        testMolecule.posList += -b*stepList
        step = min_parabola(a,b,c,va,vb,vc)
        if step < 0.0:
#            print 'zero'
            return 0.0
    
    return step
    
def initial_bracket(testMolecule, stepList, e):
    
    calculate_energy = testMolecule.ff.define_energy_routine(testMolecule,grad=False)
    
    #INITIAL BRACKET PHASE (Numerical Recipes 10.1)
    a = 0.
    initb = 1e-6
    b = initb
    va = e
    testMolecule.posList += b*stepList
    vb,_,_,_ = calculate_energy()
    testMolecule.posList += -b*stepList
    while vb > va:  #if b isn't a center for the bracket make it close to a
        if initb < 1e-20:
#            print "too small"
            return False,_,_,_,_,_
        initb = initb*0.1
        b = initb
        testMolecule.posList += b*stepList
        vb,_,_,_ = calculate_energy()
        testMolecule.posList += -b*stepList
#        print "too large"
    c = b + GOLD*(b-a) #initial guess for c (it's at least as big as b)
    testMolecule.posList += c*stepList
    vc,_,_,_ = calculate_energy()
    testMolecule.posList += -c*stepList
    while vb > vc:
        u = min_parabola(a,b,c,va,vb,vc)
        ulim = b + GLIMIT*(c-b)
        if (b-u)*(u-c) > 0.0:
            testMolecule.posList += u*stepList
            vu,_,_,_ = calculate_energy()
            testMolecule.posList += -u*stepList
            if vu < vc:
                a,b = b,u
                va,vb = vb,vu
                break
            elif vu > vb:
                c,vc = u,vu
                break
            u = c + GOLD*(c-b)
            testMolecule.posList += u*stepList
            vu,_,_,_ = calculate_energy()
            testMolecule.posList += -u*stepList
        elif (c-u)*(u-ulim) > 0.0:
            testMolecule.posList += u*stepList
            vu,_,_,_ = calculate_energy()
            testMolecule.posList += -u*stepList
            if vu < vc:
                b,c,u = c,u,u + GOLD*(u-c)
        elif (u-ulim)*(ulim-c) >= 0.0:
            u = ulim
            testMolecule.posList += u*stepList
            vu,_,_,_ = calculate_energy()
            testMolecule.posList += -u*stepList
        else:
            u = c + GOLD*(c*b)
            testMolecule.posList += u*stepList
            vu,_,_,_ = calculate_energy()
            testMolecule.posList += -u*stepList
        a,b,c = b,c,u
        va,vb,vc = vb,vc,vu
            
    return a,b,c,va,vb,vc