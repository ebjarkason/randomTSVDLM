# Script used to run forward simulations:
# Natural state simulation followed by production simulation
# Coded by: Elvar K. Bjarkason (2017)

import os
from t2data import *
import time

temptime = time.clock()
print 'Running Natural State'
# Run to a natural state:
datfile = t2data('natFWD.dat')
datfile.run(save_filename='savNat', incon_filename='incbest', simulator='autough2_5_h5all.exe',silent=True)
print 'TIME spent on natural state simulation', time.clock() - temptime

print 'Reading Natural State Outputs'
# Read natural state observations
# and udpate INCON file 'incFromNat' used to initialize the following 
# production simulation
os.system('readNatObsFromSave.py')

temptime = time.clock()
print 'Running Production'
# Run production:
datfile = t2data('prodFWD.dat')
datfile.run(save_filename='savProd', incon_filename='incFromNat', simulator='autough2_5_h5all.exe',silent=True)
print 'TIME spent on production simulation', time.clock() - temptime

print 'Reading Production Outputs'
# Read production observations:
os.system('readProdObs.py')