# Scripts implementing direct method to evaluate the sensitivity matrix
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
def evalGH4harmonicpermeabilitiesVEC(GH, H, massfluxDt, energyfluxDt, NRadj, NCON, NK, permislog, coninfo, conDists, VOLS, RTperms):
    # For adjustable permeabilities add terms to the matrix G*H:
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
                        Hrow = H[NRadjIndn]
                        if (VOLn <= 1.E20) and (VOLn > 0.):
                            GH[n*NK]   += ( (logten/VOLn)*massfac ) * Hrow
                            GH[n*NK+1] += ( (logten/VOLn)*enfac   ) * Hrow
                        
                        if (VOLm <= 1.E20) and (VOLm > 0.):
                            GH[m*NK]   += (-(logten/VOLm)*massfac ) * Hrow
                            GH[m*NK+1] += (-(logten/VOLm)*enfac   ) * Hrow 
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
                            Hrow = H[NRadjIndn]
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                GH[n*NK]   += ( (factor/VOLn)*massfac ) * Hrow
                                GH[n*NK+1] += ( (factor/VOLn)*enfac   ) * Hrow
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                GH[m*NK]   += (-(factor/VOLm)*massfac ) * Hrow
                                GH[m*NK+1] += (-(factor/VOLm)*enfac   ) * Hrow 
                        if(NRadjIndm>-1):
                            factor = logten*((Dm/Dnm)*(harmk/km))
                            Hrow = H[NRadjIndm]
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                GH[n*NK]   += ( (factor/VOLn)*massfac ) * Hrow
                                GH[n*NK+1] += ( (factor/VOLn)*enfac   ) * Hrow
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                GH[m*NK]   += (-(factor/VOLm)*massfac ) * Hrow
                                GH[m*NK+1] += (-(factor/VOLm)*enfac   ) * Hrow 
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
                        Hrow = H[NRadjIndn]
                        if (VOLn <= 1.E20) and (VOLn > 0.):
                            GH[n*NK]   += ( massfac/(VOLn*harmk) ) * Hrow
                            GH[n*NK+1] += ( enfac/(VOLn*harmk)   ) * Hrow
                        
                        if (VOLm <= 1.E20) and (VOLm > 0.):
                            GH[m*NK]   += (-massfac/(VOLS[m]*harmk) ) * Hrow
                            GH[m*NK+1] += (-enfac/(VOLS[m]*harmk)   ) * Hrow 
                    else:
                        if (NRadjIndn>-1):
                            factor = ((Dn/Dnm)*(harmk/(kn**2)))
                            Hrow = H[NRadjIndn]
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                GH[n*NK]   += ( (factor/VOLn)*massfac ) * Hrow
                                GH[n*NK+1] += ( (factor/VOLn)*enfac   ) * Hrow
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                GH[m*NK]   += (-(factor/VOLm)*massfac ) * Hrow
                                GH[m*NK+1] += (-(factor/VOLm)*enfac   ) * Hrow 
                                
                        if(NRadjIndm>-1):
                            factor = ((Dm/Dnm)*(harmk/(km**2)))
                            Hrow = H[NRadjIndm]
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                GH[n*NK]   += ( (factor/VOLn)*massfac ) * Hrow
                                GH[n*NK+1] += ( (factor/VOLn)*enfac   ) * Hrow
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                GH[m*NK]   += (-(factor/VOLm)*massfac ) * Hrow
                                GH[m*NK+1] += (-(factor/VOLm)*enfac   ) * Hrow

    return GH
                                
@jit
def evalGH4harmonicpermeabilitiesMAT(GH, H, massfluxDt, energyfluxDt, NRadj, NCON, NK, permislog, coninfo, conDists, VOLS, RTperms):
    # For adjustable permeabilities add terms to the matrix G*H:
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
                        Hrow = H[NRadjIndn,:]
                        if (VOLn <= 1.E20) and (VOLn > 0.):
                            GH[n*NK,:]   += ( (logten/VOLn)*massfac ) * Hrow
                            GH[n*NK+1,:] += ( (logten/VOLn)*enfac   ) * Hrow
                        
                        if (VOLm <= 1.E20) and (VOLm > 0.):
                            GH[m*NK,:]   += (-(logten/VOLm)*massfac ) * Hrow
                            GH[m*NK+1,:] += (-(logten/VOLm)*enfac   ) * Hrow 
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
                            Hrow = H[NRadjIndn,:]
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                GH[n*NK,:]   += ( (factor/VOLn)*massfac ) * Hrow
                                GH[n*NK+1,:] += ( (factor/VOLn)*enfac   ) * Hrow
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                GH[m*NK,:]   += (-(factor/VOLm)*massfac ) * Hrow
                                GH[m*NK+1,:] += (-(factor/VOLm)*enfac   ) * Hrow 
                        if(NRadjIndm>-1):
                            factor = logten*((Dm/Dnm)*(harmk/km))
                            Hrow = H[NRadjIndm,:]
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                GH[n*NK,:]   += ( (factor/VOLn)*massfac ) * Hrow
                                GH[n*NK+1,:] += ( (factor/VOLn)*enfac   ) * Hrow
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                GH[m*NK,:]   += (-(factor/VOLm)*massfac ) * Hrow
                                GH[m*NK+1,:] += (-(factor/VOLm)*enfac   ) * Hrow 
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
                        Hrow = H[NRadjIndn,:]
                        if (VOLn <= 1.E20) and (VOLn > 0.):
                            GH[n*NK,:]   += ( massfac/(VOLn*harmk) ) * Hrow
                            GH[n*NK+1,:] += ( enfac/(VOLn*harmk)   ) * Hrow
                        
                        if (VOLm <= 1.E20) and (VOLm > 0.):
                            GH[m*NK,:]   += (-massfac/(VOLS[m]*harmk) ) * Hrow
                            GH[m*NK+1,:] += (-enfac/(VOLS[m]*harmk)   ) * Hrow 
                    else:
                        if (NRadjIndn>-1):
                            factor = ((Dn/Dnm)*(harmk/(kn**2)))
                            Hrow = H[NRadjIndn,:]
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                GH[n*NK,:]   += ( (factor/VOLn)*massfac ) * Hrow
                                GH[n*NK+1,:] += ( (factor/VOLn)*enfac   ) * Hrow
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                GH[m*NK,:]   += (-(factor/VOLm)*massfac ) * Hrow
                                GH[m*NK+1,:] += (-(factor/VOLm)*enfac   ) * Hrow 
                                
                        if(NRadjIndm>-1):
                            factor = ((Dm/Dnm)*(harmk/(km**2)))
                            Hrow = H[NRadjIndm,:]
                            if (VOLn <= 1.E20) and (VOLn > 0.):
                                GH[n*NK,:]   += ( (factor/VOLn)*massfac ) * Hrow
                                GH[n*NK+1,:] += ( (factor/VOLn)*enfac   ) * Hrow
                            
                            if (VOLm <= 1.E20) and (VOLm > 0.):
                                GH[m*NK,:]   += (-(factor/VOLm)*massfac ) * Hrow
                                GH[m*NK+1,:] += (-(factor/VOLm)*enfac   ) * Hrow

    return GH                               
 
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

# StimesMatrix: 
# Using the direct method, function to evaluate the sensitivity matrix 
# times a a vector (Mat=FALSE) or matrix (Mat=TRUE) S*H. 
# ----------------------------------------------------------------------
# INPUTS:
# H: (Nm by s) matrix or vector
# parameters: adjustable permeability parameters
# Mat: use Mat=False when H is a vector and Mat=True when H is a matrix
# ----------------------------------------------------------------------
# RETURNS:
# SH: S*H where S is the sensitivity matrix
# ----------------------------------------------------------------------
def StimesMatrix(H, parameters, Mat=False):
    
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
    
    LMiterc = sp.load('LMiter.npy')
    if (LMiterc == 1):
        # Find the fixed sparsity pattern of the A and B matrices:
        readh5.AandBmatZeroOneFill(h5filename = 'natFWD'+strlcount+'.h5')
    
    
    # Start with solving for the natural state:
    [dt, A, B, fluxw, enthw, fluxg, enthg] = readh5.returnABmatsConFluxesEnthalpies(NEQ, strlcount, -1, -1)
    
    # Create information pertaining to G matrix:
    # Only use the G matrix implicitly for G*H:
    if Mat:
        GH = sp.zeros((NEQ,H.shape[1]))
    else:
        GH = sp.zeros(NEQ)
    
    # Precalculate the interface mass and energy flux terms:
    massfluxDt = (fluxw + fluxg)*dt
    energyfluxDt = (fluxw*enthw + fluxg*enthg)*dt
    # For adjustable permeabilities:
    if Mat:
        GH = evalGH4harmonicpermeabilitiesMAT(GH, H, massfluxDt, energyfluxDt, NRadj, NCON, NK, permislog, coninfo, conDists, VOLS, RTperms)
    else:
        GH = evalGH4harmonicpermeabilitiesVEC(GH, H, massfluxDt, energyfluxDt, NRadj, NCON, NK, permislog, coninfo, conDists, VOLS, RTperms)

    
    # Remove accumulation terms from forward Jacobian:
    A = A-B
    # Solve direct problem for the natural state (number of RHS equal to the number of columns in matrix H):
    X = spsolve(csr_matrix(A), -GH)
    
    # Generate matrix C for natural state:
    C = CmatrixNatState(Natobsadr, NNatloc, Nd, NEQ, NK)
    # Update Sensitivity matrix product SH:        
    SH = C.dot(X)
    

    
    # Now solve for the following production period:
    # Open and read adjoint file:
    adjfile = h5py.File('prodFWD'+strlcount+'.h5','r')
    # Simulation times:
    times = adjfile['fulltimes']['TIME'][:]
    Nt = len(times)
    timeEnd = times[-1]
    # Observation times:
    obstimes = sp.load('obstimes.npy')
    
    kobtim = 0
    # Lower the required number of time-steps Nt required by the direct method in case the 
    # production run did not run fully:
    if ((obstimes[-1] - timeEnd) >= 2.0e-6):
        chkTimeEnd = True
        kObsEnough = 0
        while (chkTimeEnd)and(kObsEnough < (Nobstimes-1)):####(chkTimeEnd)and(kObsEnough < Nobstimes):
            kObsEnough += 1
            if ((obstimes[-1-kObsEnough] - timeEnd) < 2.0e-6):
                chkTimeEnd = False
        
#        chkTimeEnd = True
        Ntnew = 0
        if chkTimeEnd == True:
            chkTimeEnd = False
        else:
            chkTimeEnd = True
        while chkTimeEnd:
            if (Ntnew < (Nt-1)):
                dtcheck1 = obstimes[-1-kObsEnough] - times[Ntnew]
                dtcheck2 = obstimes[-1-kObsEnough] - times[Ntnew + 1]
                cond1 = dtcheck1  < 2.0e-6
                cond2 = sp.absolute(dtcheck1) < sp.absolute(dtcheck2)
                if (cond1 and cond2):
                    chkTimeEnd = False
            else:
                chkTimeEnd = False
            Ntnew += 1
        Nt = Ntnew
            
    
    # Tracking forward in time, solve direct problem for the production time-steps:
    for timestepnr in range(0,Nt):
        [dt, A, B, fluxw, enthw, fluxg, enthg] = readh5.returnABmatsConFluxesEnthalpies(NEQ, strlcount, timestepnr, timestepnr-1)
        
        # Create information pertaining to G matrix:
        # Only use the G matrix implicitly for G*H:
        if Mat:
            GH = sp.zeros((NEQ,H.shape[1]))
        else:
            GH = sp.zeros(NEQ)
        
        # Precalculate the interface mass and energy flux terms:
        massfluxDt = (fluxw + fluxg)*dt
        energyfluxDt = (fluxw*enthw + fluxg*enthg)*dt
        # For adjustable permeabilities:
        if Mat:
            GH = evalGH4harmonicpermeabilitiesMAT(GH,H,massfluxDt,energyfluxDt,NRadj,NCON,NK,permislog,coninfo,conDists,VOLS,RTperms)
        else:
            GH = evalGH4harmonicpermeabilitiesVEC(GH,H,massfluxDt,energyfluxDt,NRadj,NCON,NK,permislog,coninfo,conDists,VOLS,RTperms)    
                
        
        # Solve direct problem for current time-step (number of RHS equal to the number of columns in matrix H):
        X = spsolve(csr_matrix(A), -GH + B.dot(X))
         
        # Construct C matrix for production pressure and enthalpy observations:
        # For now C is specified as a dense matrix, but should consider using
        # sparse format instead.
        C = sp.zeros((Nd, NEQ))
        
        simtime = times[timestepnr]
        if (simtime < timeEnd) and (kobtim < (Nobstimes)):
            ObsTime = obstimes[kobtim]
            tcheck = sp.absolute((simtime-ObsTime)/ObsTime)
            simtime2 = times[timestepnr+1]
            tcheck2 = sp.absolute((simtime2-ObsTime)/ObsTime)
            if (tcheck < 2.0e-6) and (tcheck <= tcheck2):
                for j in range(0,NProdloc):
                    adr = Prodobsadr[j]
                    if Prodobstype[j] == 'P':
                        # Pressure observation:
                        C[j + kobtim*NProdloc + NNatloc,NK*adr] = 1.
                    elif Prodobstype[j] == 'E':
                        C[j + kobtim*NProdloc + NNatloc,NK*adr] = adjfile['adjoint/gener_enth_der'][timestepnr,int(ObsFlowGENERSindx[j]),0]
                        C[j + kobtim*NProdloc + NNatloc,NK*adr+1] = adjfile['adjoint/gener_enth_der'][timestepnr,int(ObsFlowGENERSindx[j]),1]
                    
                kobtim += 1
        elif (kobtim < (Nobstimes)):
            ObsTime = obstimes[kobtim]
            tcheck = sp.absolute((simtime-ObsTime)/ObsTime)
            if (tcheck < 2.0e-6):
                for j in range(0,NProdloc):
                    adr = Prodobsadr[j]
                    if Prodobstype[j] == 'P':
                        # Pressure observation:
                        C[j + kobtim*NProdloc + NNatloc,NK*adr] = 1.
                    elif Prodobstype[j] == 'E':
                        # Flowing Enthalpy observation
                        C[j + kobtim*NProdloc + NNatloc,NK*adr] = adjfile['adjoint/gener_enth_der'][timestepnr,int(ObsFlowGENERSindx[j]),0]
                        C[j + kobtim*NProdloc + NNatloc,NK*adr+1] = adjfile['adjoint/gener_enth_der'][timestepnr,int(ObsFlowGENERSindx[j]),1]
                
                kobtim += 1    
                     
        SH += C.dot(X)
        
    # Close .h5 file:
    adjfile.close()  
    
    return SH
    
    
                
    
    