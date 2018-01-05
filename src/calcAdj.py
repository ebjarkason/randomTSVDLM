# Scripts implementing adjoint method to evaluate the sensitivity matrix
# times a vector or matrix.
# Coded by: Elvar K. Bjarkason (2017)

import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import satPress
from t2incons import *
from t2thermo import *
import h5py
import readAUTOUGHhdf5output as readh5
from numba import jit


@jit
def evalSTH4harmonicpermeabilitiesVEC(STH, M, massfluxDt, energyfluxDt, NRadj, NCON, NK, permislog, coninfo, conDists, VOLS, RTperms):
    # For adjustable permeabilities add GT*M to ST*H:
    if NRadj > 0:
        if permislog == 1:
            logten = sp.log(10)
            for k in range(0,NCON):
                NRadjIndn = coninfo[k,5]
                NRadjIndm = coninfo[k,6]
                if (NRadjIndn>-1)or(NRadjIndm>-1):
                    n = coninfo[k,0]
                    m = coninfo[k,1]
                    massfac = massfluxDt[k]
                    enfac = energyfluxDt[k]
                    VOLn = VOLS[n]
                    VOLm = VOLS[m]
                    if NRadjIndn==NRadjIndm:
                        if (VOLn <= 1.E20) and (VOLn > 0.):
                            STH[NRadjIndn] += ( (logten/VOLn)*massfac ) * M[n*NK]
                            STH[NRadjIndn] += ( (logten/VOLn)*enfac   ) * M[n*NK+1]
                        
                        if (VOLm <= 1.E20) and (VOLm > 0.):
                            STH[NRadjIndn] += (-(logten/VOLm)*massfac ) * M[m*NK]
                            STH[NRadjIndn] += (-(logten/VOLm)*enfac   ) * M[m*NK+1]
                    else:
                        Dn = conDists[k,0]
                        Dm = conDists[k,1]
                        Dnm = Dn + Dm
                        axis = coninfo[k,4]
                        kn = RTperms[coninfo[k,2],axis-1]
                        km = RTperms[coninfo[k,3],axis-1]
                        harmk = Dnm/(Dn/kn + Dm/km)
                        if (NRadjIndn>-1):
                            factor = logten*((Dn/Dnm)*(harmk/kn))
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                STH[NRadjIndn] += ( (factor/VOLn)*massfac ) * M[n*NK]
                                STH[NRadjIndn] += ( (factor/VOLn)*enfac   ) * M[n*NK+1]
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                STH[NRadjIndn] += (-(factor/VOLm)*massfac ) * M[m*NK]
                                STH[NRadjIndn] += (-(factor/VOLm)*enfac   ) * M[m*NK+1] 
                        if(NRadjIndm>-1):
                            factor = logten*((Dm/Dnm)*(harmk/km))
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                STH[NRadjIndm] += ( (factor/VOLn)*massfac ) * M[n*NK]
                                STH[NRadjIndm] += ( (factor/VOLn)*enfac   ) * M[n*NK+1]
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                STH[NRadjIndm] += (-(factor/VOLm)*massfac ) * M[m*NK]
                                STH[NRadjIndm] += (-(factor/VOLm)*enfac   ) * M[m*NK+1] 
        else:
            for k in range(0,NCON):
                NRadjIndn = coninfo[k,5]
                NRadjIndm = coninfo[k,6]
                if (NRadjIndn>-1)or(NRadjIndm>-1):
                    n = coninfo[k,0]
                    m = coninfo[k,1]
                    massfac = massfluxDt[k]
                    enfac = energyfluxDt[k]
                    Dn = conDists[k,0]
                    Dm = conDists[k,1]
                    Dnm = Dn + Dm
                    axis = coninfo[k,4]
                    kn = RTperms[coninfo[k,2],axis-1]
                    km = RTperms[coninfo[k,3],axis-1]
                    harmk = Dnm/(Dn/kn + Dm/km)
                    VOLn = VOLS[n]
                    VOLm = VOLS[m]
                    if NRadjIndn==NRadjIndm:
                        if (VOLn <= 1.E20) and (VOLn > 0.):
                            STH[NRadjIndn] += ( massfac/(VOLn*harmk) ) * M[n*NK]
                            STH[NRadjIndn] += ( enfac/(VOLn*harmk)   ) * M[n*NK+1]
                        
                        if (VOLm <= 1.E20) and (VOLm > 0.):
                            STH[NRadjIndn] += (-massfac/(VOLS[m]*harmk) ) * M[m*NK]
                            STH[NRadjIndn] += (-enfac/(VOLS[m]*harmk)   ) * M[m*NK+1]
                    else:
                        if (NRadjIndn>-1):
                            factor = ((Dn/Dnm)*(harmk/(kn**2)))
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                STH[NRadjIndn] += ( (factor/VOLn)*massfac ) * M[n*NK]
                                STH[NRadjIndn] += ( (factor/VOLn)*enfac   ) * M[n*NK+1]
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                STH[NRadjIndn] += (-(factor/VOLm)*massfac ) * M[m*NK]
                                STH[NRadjIndn] += (-(factor/VOLm)*enfac   ) * M[m*NK+1] 
                                
                        if(NRadjIndm>-1):
                            factor = ((Dm/Dnm)*(harmk/(km**2)))
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                STH[NRadjIndm] += ( (factor/VOLn)*massfac ) * M[n*NK]
                                STH[NRadjIndm] += ( (factor/VOLn)*enfac   ) * M[n*NK+1]
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                STH[NRadjIndm] += (-(factor/VOLm)*massfac ) * M[m*NK]
                                STH[NRadjIndm] += (-(factor/VOLm)*enfac   ) * M[m*NK+1]


    return STH          
         

@jit
def evalSTH4harmonicpermeabilitiesMAT(STH, M, massfluxDt, energyfluxDt, NRadj, NCON, NK, permislog, coninfo, conDists, VOLS, RTperms):
    # For adjustable permeabilities add GT*M to ST*H:
    if NRadj > 0:
        if permislog == 1:
            logten = sp.log(10)
            for k in range(0,NCON):
                NRadjIndn = coninfo[k,5]
                NRadjIndm = coninfo[k,6]
                if (NRadjIndn>-1)or(NRadjIndm>-1):
                    n = coninfo[k,0]
                    m = coninfo[k,1]
                    massfac = massfluxDt[k]
                    enfac = energyfluxDt[k]
                    VOLn = VOLS[n]
                    VOLm = VOLS[m]
                    if NRadjIndn==NRadjIndm:
                        if (VOLn <= 1.E20) and (VOLn > 0.):
                            STH[NRadjIndn,:] += ( (logten/VOLn)*massfac ) * M[n*NK,:]
                            STH[NRadjIndn,:] += ( (logten/VOLn)*enfac   ) * M[n*NK+1,:]
                        
                        if (VOLm <= 1.E20) and (VOLm > 0.):
                            STH[NRadjIndn,:] += (-(logten/VOLm)*massfac ) * M[m*NK,:]
                            STH[NRadjIndn,:] += (-(logten/VOLm)*enfac   ) * M[m*NK+1,:]
                    else:
                        Dn = conDists[k,0]
                        Dm = conDists[k,1]
                        Dnm = Dn + Dm
                        axis = coninfo[k,4]
                        kn = RTperms[coninfo[k,2],axis-1]
                        km = RTperms[coninfo[k,3],axis-1]
                        harmk = Dnm/(Dn/kn + Dm/km)
                        if (NRadjIndn>-1):
                            factor = logten*((Dn/Dnm)*(harmk/kn))
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                STH[NRadjIndn,:] += ( (factor/VOLn)*massfac ) * M[n*NK,:]
                                STH[NRadjIndn,:] += ( (factor/VOLn)*enfac   ) * M[n*NK+1,:]
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                STH[NRadjIndn,:] += (-(factor/VOLm)*massfac ) * M[m*NK,:]
                                STH[NRadjIndn,:] += (-(factor/VOLm)*enfac   ) * M[m*NK+1,:] 
                        if(NRadjIndm>-1):
                            factor = logten*((Dm/Dnm)*(harmk/km))
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                STH[NRadjIndm,:] += ( (factor/VOLn)*massfac ) * M[n*NK,:]
                                STH[NRadjIndm,:] += ( (factor/VOLn)*enfac   ) * M[n*NK+1,:]
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                STH[NRadjIndm,:] += (-(factor/VOLm)*massfac ) * M[m*NK,:]
                                STH[NRadjIndm,:] += (-(factor/VOLm)*enfac   ) * M[m*NK+1,:] 
        else:
            for k in range(0,NCON):
                NRadjIndn = coninfo[k,5]
                NRadjIndm = coninfo[k,6]
                if (NRadjIndn>-1)or(NRadjIndm>-1):
                    n = coninfo[k,0]
                    m = coninfo[k,1]
                    massfac = massfluxDt[k]
                    enfac = energyfluxDt[k]
                    Dn = conDists[k,0]
                    Dm = conDists[k,1]
                    Dnm = Dn + Dm
                    axis = coninfo[k,4]
                    kn = RTperms[coninfo[k,2],axis-1]
                    km = RTperms[coninfo[k,3],axis-1]
                    harmk = Dnm/(Dn/kn + Dm/km)
                    VOLn = VOLS[n]
                    VOLm = VOLS[m]
                    if NRadjIndn==NRadjIndm:
                        if (VOLn <= 1.E20) and (VOLn > 0.):
                            STH[NRadjIndn,:] += ( massfac/(VOLn*harmk) ) * M[n*NK,:]
                            STH[NRadjIndn,:] += ( enfac/(VOLn*harmk)   ) * M[n*NK+1,:]
                        
                        if (VOLm <= 1.E20) and (VOLm > 0.):
                            STH[NRadjIndn,:] += (-massfac/(VOLS[m]*harmk) ) * M[m*NK,:]
                            STH[NRadjIndn,:] += (-enfac/(VOLS[m]*harmk)   ) * M[m*NK+1,:]
                    else:
                        if (NRadjIndn>-1):
                            factor = ((Dn/Dnm)*(harmk/(kn**2)))
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                STH[NRadjIndn,:] += ( (factor/VOLn)*massfac ) * M[n*NK,:]
                                STH[NRadjIndn,:] += ( (factor/VOLn)*enfac   ) * M[n*NK+1,:]
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                STH[NRadjIndn,:] += (-(factor/VOLm)*massfac ) * M[m*NK,:]
                                STH[NRadjIndn,:] += (-(factor/VOLm)*enfac   ) * M[m*NK+1,:] 
                                
                        if(NRadjIndm>-1):
                            factor = ((Dm/Dnm)*(harmk/(km**2)))
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                STH[NRadjIndm,:] += ( (factor/VOLn)*massfac ) * M[n*NK,:]
                                STH[NRadjIndm,:] += ( (factor/VOLn)*enfac   ) * M[n*NK+1,:]
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                STH[NRadjIndm,:] += (-(factor/VOLm)*massfac ) * M[m*NK,:]
                                STH[NRadjIndm,:] += (-(factor/VOLm)*enfac   ) * M[m*NK+1,:]


    return STH         
                      
# CmatrixNatState: 
# Generate C matrix (contains derivatives of observation outputs w.r.t state
# variables) for natrual state. 
# ----------------------------------------------------------------------
# INPUTS:
# Natobsadr: Vector of natural state observation addresses
# NNatloc: number natural state observation addresses
# Nd: number of observations
# NEQ: number of balance or state equations
# NK: Number of components PLUS 1 for energy (NK = 2 for single water EOS)
# ----------------------------------------------------------------------
# RETURNS:
# C: (Nd by NEQ) matrix
# ---------------------------------------------------------------------- 
def CmatrixNatState(Natobsadr, NNatloc, Nd, NEQ, NK):
    # Read block temperatures or saturations:
    inc = t2incon('incbest.incon')
    # Get block temperatures or saturations at nat state observation locations:
    TorS = sp.zeros(NNatloc)
    for i in range(0,NNatloc):
        x = inc[int(Natobsadr[i])]
        # Temperature observation:
        TorS[i] = float(x[1])
    
    # Construct C matrix:
    # For now C is specified as a dense matrix, but should consider using
    # sparse format instead.
    C = sp.zeros((Nd, NEQ))
    for j in range(0,NNatloc):
        adr = Natobsadr[j]
        if TorS[j] > 1.0: # Single-phase block
            C[j,NK*adr+1] = 1.
        else: # Two-phase block
            Pdat = inc[int(Natobsadr[j])]
            P = float(Pdat[0])
            Temp = tsat( P )
            DPDT = satPress.derivDPDT(Temp)
            C[j,NK*adr] = 1./DPDT
    return C


# STtimesMatrix: 
# Using the adjoint method, function to evaluate the sensitivity matrix transposed
# times a a vector (Mat=FALSE) or matrix (Mat=TRUE) ST*H. 
# ----------------------------------------------------------------------
# INPUTS:
# H: (Nm by s) matrix or vector
# parameters: adjustable permeability parameters
# Mat: use Mat=False when H is a vector and Mat=True when H is a matrix
# ----------------------------------------------------------------------
# RETURNS:
# STH: ST*H where S is the sensitivity matrix
# ----------------------------------------------------------------------
def STtimesMatrix(H, parameters, Mat=False):
    
    # Load grid and param information:
    Prodobsadr = sp.load('Prodobsadr.npy')  
    Prodobstype = sp.load('Prodobstype.npy')
    ObsFlowGENERSindx = sp.load('ObsFlowGENERSindx.npy')
    NProdloc = len(Prodobsadr)
    Natobsadr = sp.load('Natobsadr.npy')  
    NNatloc = len(Natobsadr)
    PAX = sp.load('PAX.npy')
    RTadjIndex = sp.load('RTadjIndex.npy')
    RTperms = sp.load('RTperms.npy')
    VOLS = sp.load('Volumes.npy')
    coninfo = sp.load('coninfo.npy')
    conDists = sp.load('conDists.npy')
    numgridbal = sp.load('numgridbal.npy') # [NEL, NCON, NK, NEQ, Nm, Nd, Nobstimes, NRadj, permislog]
    NEL = numgridbal[0]
    NCON = numgridbal[1]
    NK = numgridbal[2]
    NEQ = numgridbal[3]
    Nm = numgridbal[4]
    Nd = numgridbal[5]
    Nobstimes = numgridbal[6]
    NRadj = numgridbal[7]
    permislog = numgridbal[8]
    
     
    # Current values of adjustable permeabilities:
    if (permislog == 1 ):
        # Transform from log base 10 to untransformed permeabilities:
        for j in range(0,NRadj):
            RTperms[RTadjIndex[j],PAX[j]-1] = 10.**(parameters[j]) 
    else:
        for j in range(0,NRadj):
            RTperms[RTadjIndex[j],PAX[j]-1] = parameters[j] 
    
      
    strlcount = ''
        
    # Check for matrix or vector version:
    if Mat:
        STH = sp.zeros((Nm,H.shape[1]))
    else:
        STH = sp.zeros(Nm)
    
    
    LMiterc = sp.load('LMiter.npy')
    if (LMiterc == 1):
        # Find the fixed sparsity pattern of the A and B matrices:
        readh5.AandBmatZeroOneFill(h5filename = 'natFWD'+strlcount+'.h5')
    
    
    
    
    
    # Begin tracking backward in time for production period:
    
    # Open and read adjoint file:
    adjfile = h5py.File('prodFWD'+strlcount+'.h5','r')
    # Time-steps:
    dt = adjfile['fulltimes']['DELTEX']
    Nt = len(dt)
    # Simulation times:
    times = adjfile['fulltimes']['TIME'][:]
    timeEnd = times[-1]
    # Observation times:
    obstimes = sp.load('obstimes.npy')
    
    kobtim = 0
    kobtimBegin = 0
    # In case the production runs exits before the final observation time:
    timcarryon = True
    while (kobtim < Nobstimes)and(timcarryon):
        if ((obstimes[Nobstimes-kobtim-1] - timeEnd) >= 2.0e-6):
            kobtim += 1
            kobtimBegin += 1
        else:
            timcarryon = False
    
    # Tracking backward in time, solve adjoint problem for the production time-steps:
    for timestepnr in range(Nt-1,-1,-1):
        [dt, A, B, fluxw, enthw, fluxg, enthg] = readh5.returnABmatsConFluxesEnthalpies(NEQ, strlcount, timestepnr, timestepnr)
        
        # Construct C matrix for production pressure and enthalpy observations:
        # For now C is specified as a dense matrix, but should consider using
        # sparse format instead.
        C = sp.zeros((Nd,NEQ))
        
        isobstime = False
        simtime = times[timestepnr]
        if (simtime <= timeEnd) and (kobtim < Nobstimes):
            ObsTime = obstimes[-kobtim-1]
            tcheck = sp.absolute((simtime-ObsTime)/ObsTime)
            cond1 = (tcheck < 2.0e-6)
            if (timestepnr == 0):
                cond2 = True
            else:
                simtime2 = times[timestepnr-1]
                tcheck2 = sp.absolute((simtime2-ObsTime)/ObsTime)
                cond2 = (tcheck < tcheck2)
            if cond1 and cond2:
                isobstime = True
                for j in range(0,NProdloc):
                    adr = Prodobsadr[j]
                    if Prodobstype[j] == 'P':
                        # Pressure observation:
                        C[j + (Nobstimes-1-kobtim)*NProdloc + NNatloc,NK*adr] = 1.
                    elif Prodobstype[j] == 'E':
                        C[j + (Nobstimes-1-kobtim)*NProdloc + NNatloc,NK*adr] = adjfile['adjoint/gener_enth_der'][timestepnr,int(ObsFlowGENERSindx[j]),0]
                        C[j + (Nobstimes-1-kobtim)*NProdloc + NNatloc,NK*adr+1] = adjfile['adjoint/gener_enth_der'][timestepnr,int(ObsFlowGENERSindx[j]),1]
                    
                kobtim += 1   

        # Solve adjoint problem for current time-step (number of RHS equal to the number of columns in matrix H):
        CT = C.transpose()
        RHS = -CT.dot(H)
        if (kobtim == (kobtimBegin+1))and(isobstime):
            M = spsolve(csr_matrix(A.transpose()), RHS )
        elif (kobtim > 0)and(kobtim > kobtimBegin):
            BT = B.transpose()
            RHS += BT.dot(M)
            M = spsolve(csr_matrix(A.transpose()), RHS )
            
        
        if (kobtim > kobtimBegin):
            # Evaluate GT*M and add to ST*H :
            # Precalculate the interface mass and energy flux terms:
            massfluxDt = (fluxw + fluxg)*dt
            energyfluxDt = (fluxw*enthw + fluxg*enthg)*dt
            # For adjustable permeabilities:
            if Mat:
                STH = evalSTH4harmonicpermeabilitiesMAT(STH,M,massfluxDt,energyfluxDt,NRadj,NCON,NK,permislog,coninfo,conDists,VOLS,RTperms)
            else:
                STH = evalSTH4harmonicpermeabilitiesVEC(STH,M,massfluxDt,energyfluxDt,NRadj,NCON,NK,permislog,coninfo,conDists,VOLS,RTperms)
            

    # Close .h5 file:
    adjfile.close()  
    
    
    # End with solving for the natural state:
    [dt, A, B, fluxw, enthw, fluxg, enthg] = readh5.returnABmatsConFluxesEnthalpies(NEQ, strlcount, -1, -1)
    # Remove accumulation terms from forward Jacobian:
    A = A-B   
    
    # Generate matrix C for natural state:
    C = CmatrixNatState(Natobsadr, NNatloc, Nd, NEQ, NK)

    # Solve adjoint equations for the steady state:
    CT = C.transpose()
    RHS = -CT.dot(H)
    if (kobtim > kobtimBegin):
        B1T = B.transpose()
        RHS += B1T.dot(M)
    M = spsolve(csr_matrix(A.transpose()), RHS )
    
    
    # Evaluate GT*M and add to ST*H :
    
    # Precalculate the interface mass and energy flux terms:
    massfluxDt = (fluxw + fluxg)*dt
    energyfluxDt = (fluxw*enthw + fluxg*enthg)*dt
    # For adjustable permeabilities:
    if Mat:
        STH = evalSTH4harmonicpermeabilitiesMAT(STH,M,massfluxDt,energyfluxDt,NRadj,NCON,NK,permislog,coninfo,conDists,VOLS,RTperms)
    else:
        STH = evalSTH4harmonicpermeabilitiesVEC(STH,M,massfluxDt,energyfluxDt,NRadj,NCON,NK,permislog,coninfo,conDists,VOLS,RTperms)
            
    return STH
















        

