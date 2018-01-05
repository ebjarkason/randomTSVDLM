# Script for running forward simulation and evaluting observation residuals
# Coded by: Elvar K. Bjarkason (2017)

import time
import os
import scipy as sp
from t2data import *
import SaveLoadSparseCSRmatrix as slspCSR

# resFromSim:
# Function for running the forward simulations and evaluating the 
# observation and regularization residuals
# ---------------------------------------------------------------------
# INPUTS:
# parameters: vector of length Nm containing the adjustable permeability parameters.
# ----------------------------------------------------------------------
# RETURNS:
# rw: (vector of length Nd) the weighted observation residual vector 
# (weighted by CD^(-0.5) where CD is the observation covariance matrix)
# ----------------------------------------------------------------------
def resFromSim(parameters):

    # Name of datfile:
    fdat = open('datfileFWDextension.txt')
    line = fdat.readline()
    datfile = line.split()
    datfile = datfile[0]
    fdat.close()

    # Load grid and param information:
    numgridbal = sp.load('numgridbal.npy') # [NEL, NCON, NK, NEQ, Nm, Nd, Nobstimes, NRadj, permislog]
    Nd = numgridbal[5]
    NRadj = numgridbal[7]
    permislog = numgridbal[8]
    
    if (permislog == 1 ):
        # Transform from log base 10 to untransformed permeabilities:
        xparameters = 10.0**parameters
    else:
        xparameters = sp.copy(parameters)
    
    # Rewrite natural state .dat file with the most up to date permeabilities:
    fNATtpl = open('nat'+datfile,'r')
    fNATrun = open('natFWD.dat','w')    
    # Read template file until parameter section,
    # and update permeabilites in natural state .dat file used for forward similation
    k = -5
    x = 0
    lines = fNATtpl.readlines()
    # Would be better to generalize this e.g. using PyTOUGH, and the 
    # specified adjustable rock-types and principal axis of 
    # each adjustable permeability parameter.
    # But this works fine for present test problem
    for line in lines:
        if (k > -1)and(k < 8000):
            fNATrun.write(line[0:30])
            fNATrun.write("{:.4e}".format(xparameters[k]))
            fNATrun.write("{:.4e}".format(xparameters[k]))
            fNATrun.write("{:.4e}".format(xparameters[k+NRadj/2]))
            fNATrun.write(line[60::])
        else:
            fNATrun.write(line)
        k += 1
    fNATtpl.close()
    fNATrun.close()
    
    # Rewrite natural state .dat file with the most up to date permeabilities:
    fPRODtpl = open('prod'+datfile,'r')
    fPRODrun = open('prodFWD.dat','w')
    # Read template file until parameter section,
    # and update permeabilites in production .dat file used for forward similation
    k = -5
    x = 0
    lines = fPRODtpl.readlines()
    # Would be better to generalize this e.g. using PyTOUGH, and the 
    # specified adjustable rock-types and principal axis of 
    # each adjustable permeability parameter.
    # But this works fine for present test problem
    for line in lines:
        if (k > -1)and(k < 8000):
            fPRODrun.write(line[0:30])
            fPRODrun.write("{:.4e}".format(xparameters[k]))
            fPRODrun.write("{:.4e}".format(xparameters[k]))
            fPRODrun.write("{:.4e}".format(xparameters[k+NRadj/2]))
            fPRODrun.write(line[60::])
        else:
            fPRODrun.write(line)
        k += 1
    fPRODtpl.close()
    fPRODrun.close()
    
    # Run forward model
    os.system('python runModelFWD.py')

    # Create residual vector r:
    data = sp.loadtxt('2D001NoisyData.txt') # Load noisy observations
    # Read simulated observations:
    simdata = sp.zeros(Nd)
    # Read simulated natural state observations:
    dNat = sp.loadtxt('simulobservNat.txt')
    NdNat = len(dNat)
    simdata[0:NdNat] = dNat
    # Read simulated production observations:
    dProd = sp.loadtxt('simulobservProd.txt')
    simdata[NdNat::] = dProd
    
    # Compute residual vector:
    r = simdata - data # data mismatch or residual vector
    wmatrix = sp.load('CdInvSQRT.npy')
    rw = sp.dot(wmatrix,r) # weighted residual vector
    
    sp.save('parameters.npy',parameters)
    
    return rw  
    
    