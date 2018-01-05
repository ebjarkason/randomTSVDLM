# Script for reading simulation outputs from natural state simulation.
# Coded by: Elvar K. Bjarkason (2017)

from t2incons import *
import scipy as sp
from t2thermo import *
import shutil
import h5py


# Load observation addresses:
obsadr = sp.load('Natobsadr.npy') 
Nloc = len(obsadr)
obs = sp.zeros(Nloc)

# Load natural state SAVE file:
inc = t2incon('savNat.save')

# Read natural state observation temperatures:
# Note!!!!!!!!!!!!!!!!!!
# Would be better to read temperature from new .h5 output rather than using the SAVE file
#!!!!!!!!!!!!!!!!!!!!!!!!!!!
for i in range(0,Nloc):
    x = inc[int(obsadr[i])] # State variables of observation location
    T = x[1] # Second primary variable
    if T > 1.0: # Second primary variable is temperature
        obs[i] = T
    else: # Second primary variable is saturation
        # Two-phase conditions; 
        # calculate temperature from saturation pressure
        obs[i] = tsat(x[0])        
        
# Save files with the appropriate lambda counter extension:"
lcount = sp.load('lamcount.npy')
lcount[0] += 1
sp.save('lamcount.npy',lcount)
strlcount = str(lcount[0])

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!
# Delete porosity values from the INCON:
inc.porosity = None
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Create an incon for the following production simulation:
inc.write('incFromNat.incon')
# Save observations:
sp.savetxt('simulobservNat.txt', obs)








