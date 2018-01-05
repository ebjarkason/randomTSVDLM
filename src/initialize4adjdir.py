# Script used to generate parameter, observation and grid information for
# the adjoint and direct solves
# Coded by: Elvar K. Bjarkason (2017)

from t2data import *
import scipy as sp
import os

# Delete files to make sure that the problem is running correctly:
remfiles = ['simulobservNat.txt','simulobservProd.txt','natFWD.dat','prodFWD.dat','savNat.save','savProd.save','incFromNat.incon','incbest.incon','Natobsadr.npy','datfileNameFWD.txt','numgridbal.npy','RTperms.npy','Volumes.npy','coninfo.npy','conDists.npy','PAX.npy','RTadjIndex.npy']
for afile in remfiles:
    if os.path.isfile(afile):
        os.remove(afile)

########################################################################
# EDIT this part to specify observation locations and names of TOUGH2
# input files used to run simulations.

permislog = 1 # Use when adjustable permeability parameters are log-transformed (base 10)
#permislog = 0 # Use when adjustable permeability parameters are NOT log-transformed

# For Natural state, the Temperature Observation Block names:

# Name of baseline TOUGH2 .dat file used to generate grid information used
# by the adjoint code.
# The baseline file should have the exact same structure as the file used for 
# the forward simulations. The values of adjustable parameters 
# (i.e. adjustable permeabilities) can have some default values. However,
# other parameters should be the same as those used during forward simulations.
# E.g. permeability values of non-adjustable rock-types are stored
# for later use within the adjoint code.
datfile = 'nat2002.dat' # Name of natural state .dat file - USED AS BASELINE FILE FOR GRID INFORMATION
# datfile = prod2002.dat # Name of production .dat file
# The naming convention is 
# 'nat' + extension +'.dat' for natural state .dat file
# 'prod' + extension +'.dat' for production .dat file
# Save name extension for later use:
fdat = open('datfileFWDextension.txt','w')
fdat.write(datfile[3::])
fdat.close()
T2file = t2data(datfile)

# ---------------------------------------------------------------------
# For Natural state observations:
# ---------------------------------------------------------------------
# The downhole Observation Block names:
Natoblocks = ['aaa 1','aaa 4','aaa 7','aaa10','aaa13','aaa16','aaa19','aaa22','aaa25','aaa28','aaa31','aaa34','aaa37','aaa40','aaa43','aaa46','aaa49','aaa52','aaa55','aaa58','aaa61','aaa64','aaa67','aaa70','aaa73','aaa76','aaa79','aau 1','aau 4','aau 7','aau10','aau13','aau16','aau19','aau22','aau25','aau28','aau31','aau34','aau37','aau40','aau43','aau46','aau49','aau52','aau55','aau58','aau61','aau64','aau67','aau70','aau73','aau76','aau79','bao 1','bao 4','bao 7','bao10','bao13','bao16','bao19','bao22','bao25','bao28','bao31','bao34','bao37','bao40','bao43','bao46','bao49','bao52','bao55','bao58','bao61','bao64','bao67','bao70','bao73','bao76','bao79','cai 1','cai 4','cai 7','cai10','cai13','cai16','cai19','cai22','cai25','cai28','cai31','cai34','cai37','cai40','cai43','cai46','cai49','cai52','cai55','cai58','cai61','cai64','cai67','cai70','cai73','cai76','cai79','dac 1','dac 4','dac 7','dac10','dac13','dac16','dac19','dac22','dac25','dac28','dac31','dac34','dac37','dac40','dac43','dac46','dac49','dac52','dac55','dac58','dac61','dac64','dac67','dac70','dac73','dac76','dac79']
# Natural state observation types are all block temperatures:
Natobstype = ['T']*len(Natoblocks)
# Natural state observation addresses:
NNatloc = len(Natoblocks)
Natobsadr = sp.zeros(NNatloc,dtype=sp.int64)
for j in range(0,NNatloc):
    LOCindex = T2file.grid.block_index(Natoblocks[j])
    Natobsadr[j] = LOCindex
# Save natural state observation addresses    
sp.save('Natobsadr.npy',Natobsadr)  
# Save natural state observation types:
sp.save('Natobstype.npy',Natobstype)  


# ---------------------------------------------------------------------
# For production observations:
# ---------------------------------------------------------------------
# For now we make the simplified assumption that each production 
# well gives pressure and enthalpy observations at the specified 
# observation times.
# Here the production observations at each observation time are:
# a pressure observation and an enthalpy observation at each producer.

# Observation times for production simulation:
obstimes = [7.88400E+6, 1.57680E+7, 2.36520E+7, 3.15360E+7, 3.94200E+7, 4.73040E+7, 5.51880E+7, 6.30720E+7, 7.09560E+7, 7.88400E+7, 8.67240E+7, 9.46080E+7]
# Save production observation times:
sp.save('obstimes.npy', obstimes)
Nobstimes = len(obstimes)
# Names of observation blocks for production observations,
# corresponding to the production zones of the 3 production wells:
Prodoblocks = ['aab12', 'aap14', 'aaf18']
# Number of production observation locations:
NProdloc = len(Prodoblocks)

# Specify observation types at each production observation time:
Prodobstype = []
# Here the production observations (at each observation time) are 
# pressure for each production well and also enthalpy for each production well.
# Add production pressure observations for each observation location:
for j in range(0,NProdloc):
    Prodobstype.extend('P')
# Add production enthalpy observations for each observation location:
for j in range(0, NProdloc):
    Prodobstype.extend('E')
sp.save('Prodobstype.npy',Prodobstype)

# Production observation addresses:
Prodobsadr = sp.zeros(2*NProdloc,dtype=sp.int64)
for j in range(0,NProdloc):
    LOCindex = T2file.grid.block_index(Prodoblocks[j])
    Prodobsadr[j] = LOCindex
    Prodobsadr[j + NProdloc] = LOCindex
# save production observation addresses    
sp.save('Prodobsadr.npy',Prodobsadr)
NProdloc = len(Prodobsadr)

# Specify the flowing enthalpy observation GENER indices:
ObsFlowGENERSindx = sp.zeros(NProdloc, dtype=sp.int64)
ObsFlowGENERSindx[0:3] = -999 # Dummy values for production pressure observations
ObsFlowGENERSindx[3] = 2
ObsFlowGENERSindx[4] = 1
ObsFlowGENERSindx[5] = 0
sp.save('ObsFlowGENERSindx.npy',ObsFlowGENERSindx)



# ---------------------------------------------------------------------
# Specify adjustable model parametes:
# ---------------------------------------------------------------------
# The model parameters are the log10 of the horizontal, kx, and vertical,
# kz, permeabilities of every model block excluding the constant atmospheric
# boudnary blocks. In total 8,000 rock-types: giving 16,000 parameters.
Perm_directions = [1, 3]

# List of all rock-types:
RTlist = T2file.grid.rocktypelist
NR = len(RTlist) # Number of rock types
RT = ['']*NR # list of rock type names
for i in range(0,NR):
    RT[i] = RTlist[i].name
print "Created list of rock types"

# The adjustable rock types:
# Here we assume that the adjustable parameter vector is:
# log10[ kxR0, kxR1, ..., kxR7999, kzR0, kzR1, ..., kzR7999]
RTadjHalf = RT[1::] # Leave out atmospheric rock-type
NRadjHalf = len(RTadjHalf)
RTadj = ['']*(2*NRadjHalf)
NRadj = len(RTadj) # Number of adjustable permeabilities
for i in range(0,NRadjHalf):
    RTadj[i] = RTadjHalf[i]
    RTadj[i+1*NRadjHalf] = RTadjHalf[i]

# Set primary axis of each adjustable permeability parameter:
PAX = sp.zeros(NRadj, dtype=sp.int64) # List of the adjustable permeabilities' primary axis
PAX[0:NRadjHalf] = Perm_directions[0] # Horizontal perms, kx
PAX[NRadjHalf::] = Perm_directions[1] # Horizontal perms, kz

# Save PAX and RTadj indices for permeabilities:
RTadjIndex = sp.zeros(NRadj,dtype=sp.int64)
for i in range(0,NRadj):
    RTadjIndex[i] = RT.index(RTadj[i])
sp.save('PAX.npy', PAX)
sp.save('RTadjIndex.npy', RTadjIndex)






# ---------------------------------------------------------------------
# Number of parameters and observations:
# ---------------------------------------------------------------------
Nm = NRadj # Number of model parameters
Nd = NNatloc + NProdloc*Nobstimes # Number of observations


# ---------------------------------------------------------------------
# Save permeabilities of base file:
# ---------------------------------------------------------------------
# Elements of RTperms pertaining to adjustable permeabilities are 
# updated within the adjoint or direct code. However, the code also 
# needs the other permeability values that are fixed throughout.
print "Reading permeabilities from TOUGH2 input file"
# Get values of each rock type's permeabilities:
RTperms = sp.zeros((NR, 3))
for j in range(0,NR):
    rock = RTlist[j]
    rockperm = rock.permeability
    RTperms[j,0] = rockperm[0]
    RTperms[j,1] = rockperm[1]
    RTperms[j,2] = rockperm[2]
# Save the permeabilities:
sp.save('RTperms.npy',RTperms)
print "Finished reading perms"

                   
# ---------------------------------------------------------------------
# Get block and connection information:
# ---------------------------------------------------------------------
NEL = T2file.grid.num_blocks  # Number of grid elements
NCON = T2file.grid.num_connections  # Number of grid connections
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Perhaps rename NK as NK1 ?? :
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
NK = 2 # Number of components PLUS 1 for energy (NK = 2 for single water EOS)
NEQ = NK*NEL # Number of balance equations

# Get the element names, rock types, block volumes, etc.:
ELRT = ['']*NEL # List of element rock types
VOLS = sp.zeros(NEL)# Vector of block volumes
RTindices = sp.zeros(NEL, dtype=sp.int64) # RTindices for each block

blks = T2file.grid.blocklist
for j in range(0,NEL):
    blk = blks[j]
    rock = blk.rocktype.name
    ELRT[j] = rock
    RTindices[j] = RT.index(rock)
    VOLS[j] = blk.volume  

# Save block volumes:
sp.save('Volumes.npy', VOLS)
# Store vector of [NEL, NCON, NK, NEQ, Nm, Nd, Nobstimes, NRadj, permislog]:
numgridbal = sp.zeros(9, dtype=sp.int64)
numgridbal[:] = [NEL, NCON, NK, NEQ, Nm, Nd, Nobstimes, NRadj, permislog] # Should consider turning numgridbal into something more flexible, like a dictionary. !!!!!!!!!
sp.save('numgridbal.npy',numgridbal)
    

# Get connection information:
print "Generating Connection Info"
cons = T2file.grid.connectionlist

# For later use create a matrix containing:
# coninfo[i,:] = [n, m, RTn, RTm, axis, NRadjIndn, NRadjIndm]
# conDists[i,:] = [Dn, Dm]
# Where for the ith connection
# n,m are the block indices of the connected elements
# Dn,Dm are the connection distances
# RTn,RTm are the rock type indices
# axis is the number of the primary axis (1, 2 or 3) the connection is parallel to (or associated with)
# NRadjIndn is the index number of the adjustable permeability
coninfo = sp.zeros((NCON, 7), dtype=sp.int64)
coninfo[:,5:7] = -9 # Default values
conDists = sp.zeros((NCON, 2))

# Loop over grid connections:
for j in range(0, NCON):
    con = cons[j]
    axis = con.direction
    connected = con.block
    n = T2file.grid.block_index(connected[0].name)
    m = T2file.grid.block_index(connected[1].name)
    dist = con.distance
    coninfo[j,0] = n
    coninfo[j,1] = m
    conDists[j,0] = dist[0]
    conDists[j,1] = dist[1]
    coninfo[j,2] = RTindices[n]
    coninfo[j,3] = RTindices[m]
    coninfo[j,4] = axis
    
#    for k in range(0,NRadj):
#        if axis == PAX[k]:
#            if (ELRT[n]==RTadj[k]):
#                coninfo[j,5] = k   
#            if (ELRT[m]==RTadj[k]):
#                coninfo[j,6] = k     
    
    # Try to speed up:
    try:
        coninfo[j,5] = RTadj.index(ELRT[n]) + Perm_directions.index(axis)*NRadjHalf
    except ValueError:
        pass
    try:
        coninfo[j,6] = RTadj.index(ELRT[m]) + Perm_directions.index(axis)*NRadjHalf
    except ValueError:
        pass
    
sp.save('coninfo.npy', coninfo)
sp.save('conDists.npy', conDists)
print "Finished Generating Connection Info"                        




# Initialize lambda search counter:
sp.save('lamcount.npy',sp.zeros(1, dtype=sp.int64))
# Initialize LM iteration counter:
sp.save('LMiter.npy',sp.zeros(1, dtype=sp.int64))






