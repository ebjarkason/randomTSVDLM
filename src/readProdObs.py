# Script for reading simulation outputs from production simulation.
# Coded by: Elvar K. Bjarkason (2017)

from t2incons import *
import scipy as sp
from t2thermo import *
import shutil
import h5py


# Load observation addresses:
obsadr = sp.load('Prodobsadr.npy') 
Nloc = len(obsadr)
# Load observation types:
Prodobstype = sp.load('Prodobstype.npy')
ObsFlowGENERSindx = sp.load('ObsFlowGENERSindx.npy')

# Simulation times:
adjfile = h5py.File('prodFWD.h5','r')
times = adjfile['fulltimes']['TIME'][:]
Nt = len(times)
# Observation times:
obstimes = sp.load('obstimes.npy')


# Read production observations:
numgridbal = sp.load('numgridbal.npy') # [NEL, NCON, NK, NEQ, Nm, Nd, Nobstimes, NRadj, NRadjHalf, permislog]
Nobstimes = numgridbal[6]
obs = sp.zeros(Nloc*Nobstimes)

kobtim = 0
for j in range(0, Nt):
    simtime = times[j]
    if (j < (Nt-1)) and (kobtim < (Nobstimes)):
        ObsTime = obstimes[kobtim]
        tcheck = sp.absolute((simtime-ObsTime)/ObsTime)
        simtime2 = times[j+1]
        tcheck2 = sp.absolute((simtime2-ObsTime)/ObsTime)
        if (tcheck < 2.0e-6) and (tcheck <= tcheck2):
            for i in range(0,Nloc):
                if Prodobstype[i] == 'P':
                    # Pressure observation:
                    obs[i + kobtim*Nloc] = adjfile['primary'][j,int(obsadr[i]),0]
                elif Prodobstype[i] == 'E':
                    # Flowing Enthalpy observation
                    obs[i + kobtim*Nloc] = adjfile['generation'][j,int(ObsFlowGENERSindx[i]),1]
            
            kobtim += 1
    elif (kobtim < (Nobstimes)):
        ObsTime = obstimes[kobtim]
        tcheck = sp.absolute((simtime-ObsTime)/ObsTime)
        if (tcheck < 2.0e-6):
            for i in range(0,Nloc):
                if Prodobstype[i] == 'P':
                    # Pressure observation:
                    obs[i + kobtim*Nloc] = adjfile['primary'][j,int(obsadr[i]),0]
                elif Prodobstype[i] == 'E':
                    # Flowing Enthalpy observation
                    obs[i + kobtim*Nloc] = adjfile['generation'][j,int(ObsFlowGENERSindx[i]),1]
            
            kobtim += 1
        
        
# Save files with the appropriate lambda counter extension:"
lcount = sp.load('lamcount.npy')
strlcount = str(lcount[0])

# Save observations:
sp.savetxt('simulobservProd.txt', obs)









