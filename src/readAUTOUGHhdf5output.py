# Reading HDF5 run files from a steady state AUTOUGH run for assembling 
# FWD matrix, ACC matrix, liquid/vapour fluxes and enthalpies
# Coded by: Elvar K. Bjarkason (2017)

import scipy as sp
import h5py
from scipy.sparse import csr_matrix


# AandBmatZeroOneFill: 
# Generate sparsity information (and fill-in) for the TOUGH2 forward (FWD) and
# accumulation (ACC) matrices.    
# ----------------------------------------------------------------------
# INPUTS:
# h5filename: .h5 binary file name from TOUGH2 output (use nat state
# since it should be smaller than production bonary output)
# ----------------------------------------------------------------------
# SAVES:
# fwdrows.npy: row indices for elements of FWD matrix
# fwdcols.npy: column indices for elements of FWD matrix
# accrows.npy: row indices for elements of ACC matrix
# acccols.npy: column indices for elements of ACC matrix
# zerosAmat.npy: Indices for FWD matrix elements to be set to zero
# onesAmat.npy:  Indices for FWD matrix elements to be set to one
# zerosBmat.npy: Indices for ACC matrix elements to be set to zero 
# ----------------------------------------------------------------------
def AandBmatZeroOneFill(h5filename):
    # Open and read HDF5 output file:
    # (the natural state file may typically be smaller than the production file)
    adjfile = h5py.File(h5filename,'r')
    
    # IRN (forward matrix row indices), ICN (forward matrix column indices) as an array
    fwdrows = adjfile['adjoint/fwdmat/irnicn'][:,0] - 1 # Minus 1 to convert indexing from FORTRAN to Python
    fwdcols = adjfile['adjoint/fwdmat/irnicn'][:,1] - 1 # Minus 1 to convert indexing from FORTRAN to Python
    sp.save('fwdrows.npy',fwdrows)
    sp.save('fwdcols.npy',fwdcols)
    # IRN (accumulation matrix row indices), ICN (accumulation matrix column indices) as an array
    accrows = adjfile['adjoint/accmat/irnicn'][:,0] - 1 # Minus 1 to convert indexing from FORTRAN to Python
    acccols = adjfile['adjoint/accmat/irnicn'][:,1] - 1 # Minus 1 to convert indexing from FORTRAN to Python
    sp.save('accrows.npy',accrows)
    sp.save('acccols.npy',acccols)
    
    # Indices for large constant boundary blocks:
    VOLS = sp.load('Volumes.npy')
    zerosAmat = [] # Indices for FWD matrix elements to be set to zero
    onesAmat = [] # Indices for FWD matrix elements to be set to one
    for i in range(0,len(fwdrows)):
        irn = fwdrows[i]
        icn = fwdcols[i]
        rVol = VOLS[int(irn/2)]  # SHOULD PROBABLY USE NK1 instead of 2 ???? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        cVol = VOLS[int(icn/2)]
        if (rVol > 1.E20) or (rVol < 0.) or (cVol > 1.E20) or (cVol < 0.):
            if (irn==icn):
                onesAmat.extend([i])
            else:
                zerosAmat.extend([i])
    zerosBmat = [] #  Indices for ACC matrix elements to be set to zero
    for i in range(0,len(accrows)):
        irn = accrows[i]
        icn = acccols[i]
        rVol = VOLS[int(irn/2)] # SHOULD PROBABLY USE NK1 instead of 2 ???? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        cVol = VOLS[int(icn/2)]
        if (rVol > 1.E20) or (rVol < 0.) or (cVol > 1.E20) or (cVol < 0.):
            zerosBmat.extend([i])
    sp.save('zerosAmat.npy',zerosAmat)
    sp.save('onesAmat.npy',onesAmat)
    sp.save('zerosBmat.npy',zerosBmat)
    adjfile.close()
    return
    
# returnABmatsConFluxesEnthalpies: 
#   
# ----------------------------------------------------------------------
# INPUTS:
# NEQ: number of balance equations
# strlcount: file name extension
# timestepnr: index for current time-step
# timestepnrB: index for time-step used to generate ACC matrix (could be 
# the following or previous time-step dempending on whether we are using 
# the adjoint or direct method)
# ----------------------------------------------------------------------
# RETURNS:
# dt: time-step [s]
# A: FWD Jacobian matrix
# B: ACC matrix
# fluxw: liquid fluxes at block connections
# enthw: liquid enthalpies at block connections
# fluxg: vapour fluxes at block connections
# enthg: vapour enthalpies at block connections
# ----------------------------------------------------------------------
def returnABmatsConFluxesEnthalpies(NEQ, strlcount, timestepnr, timestepnrB):
    # Open and read adjoint file:
    if timestepnr == -1: # Read nat state file
        adjfile = h5py.File('natFWD'+strlcount+'.h5','r')
    else: # Read production file
        adjfile = h5py.File('prodFWD'+strlcount+'.h5','r')
    # Read current time-step from file:
    dt = adjfile['fulltimes']['DELTEX'][timestepnr]
    
    fwdrows = sp.load('fwdrows.npy')
    fwdcols = sp.load('fwdcols.npy')
    accrows = sp.load('accrows.npy')
    acccols = sp.load('acccols.npy')
    zerosAmat = sp.load('zerosAmat.npy')
    onesAmat = sp.load('onesAmat.npy')
    zerosBmat = sp.load('zerosBmat.npy')
        
    # Assemble forward matrix from file for the latest time-step:
    fwdels = adjfile['adjoint/fwdmat/co'][:,timestepnr]
    # Adjust values for large or constant boundary blocks:
    fwdels[zerosAmat] = 0.
    fwdels[onesAmat] = 1.
    A = csr_matrix((fwdels, (fwdrows,fwdcols)), shape=(NEQ,NEQ)) # FWD matrix
    
    if (timestepnrB == -1) and (timestepnr == 0): 
        # First production time-step using direct method
        # Read B matrix from last natural state time:
        natadjfile = h5py.File('natFWD'+strlcount+'.h5','r')
        accels = natadjfile['adjoint/accmat/co'][:,timestepnrB]
        natadjfile.close()
    else:
        accels = adjfile['adjoint/accmat/co'][:,timestepnrB]
    # Adjust values for large or constant boundary blocks:
    zerosBmat = sp.load('zerosBmat.npy')
    # Adjust values for large or constant boundary blocks:
    accels[zerosBmat] = 0.
    B = csr_matrix((accels, (accrows,acccols)), shape=(NEQ,NEQ)) # ACC matrix
    
    # Read flux information at block interfaces:
    fluxw = adjfile['adjoint/flowenth'][timestepnr,:,2] # for storing liquid phase interface fluxes
    enthw = adjfile['adjoint/flowenth'][timestepnr,:,3] # for storing liquid phase interface enthalpies
    fluxg = adjfile['adjoint/flowenth'][timestepnr,:,0] # for storing vapour phase interface fluxes
    enthg = adjfile['adjoint/flowenth'][timestepnr,:,1] # for storing vapour phase interface enthalpies 
    
    adjfile.close() 
    
    return dt, A, B, fluxw, enthw, fluxg, enthg
    
