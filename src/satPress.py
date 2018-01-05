# Functions for two-phase pressure-temperature relationship
# Coded by: Elvar K. Bjarkason (2017)

import numpy as np

# evalSatP: 
# Evaluate saturation pressure as a function of temperature    
# ----------------------------------------------------------------------
# INPUTS:
# T: Temperature [deg. C]
# ----------------------------------------------------------------------
# RETURNS:
# P: Saturation pressure [Pa]
# ----------------------------------------------------------------------
def evalSatP(T):
    A1 = -7.691234564E0
    A2 = -2.608023696E1
    A3 = -1.681706546E2
    A4 = 6.423285504E1
    A5 = -1.189646225E2
    A6 = 4.167117320E0
    A7 = 2.097506760E1
    A8 = 1.E9
    A9 = 6. 
    
    TC=(T+273.15E0)/647.3E0
    X1=1.0E0-TC
    X2=X1*X1
    SC=A5*X1+A4
    SC=SC*X1+A3
    SC=SC*X1+A2
    SC=SC*X1+A1
    SC=SC*X1
    PC= np.exp(SC/(TC*(1.0E0+A6*X1+A7*X2))-X1/(A8*X2+A9))
    P=PC*2.212E7
    
    return P

# derivDPDT: 
# Evaluate derivative of saturation pressure w.r.t. temperature    
# ----------------------------------------------------------------------
# INPUTS:
# T: Temperature [deg. C]
# ----------------------------------------------------------------------
# RETURNS:
# DPDT: dP/dT
# ----------------------------------------------------------------------
def derivDPDT(T):
    A1 = -7.691234564E0
    A2 = -2.608023696E1
    A3 = -1.681706546E2
    A4 = 6.423285504E1
    A5 = -1.189646225E2
    A6 = 4.167117320E0
    A7 = 2.097506760E1
    A8 = 1.E9
    A9 = 6.
    
    
    TC=(T+273.15E0)/647.3E0
    DTCDT = 1.0/647.3E0
 
    X1=1.0E0-TC
    DX1DTC = -1.0
    X2=X1*X1
    DX2DX1 = 2.0*X1
    
    SC=A5*X1+A4 
    SC=SC*X1+A3
    SC=SC*X1+A2
    SC=SC*X1+A1
    SC=SC*X1
#    SCED = A1*X1 + A2*(X1**2.0) + A3*(X1**3.0) + A4*(X1**4.0) + A5*(X1**5.0)
    DSCDX1 = A1 + 2.0*A2*X1 + 3.0*A3*(X1**2.0) + 4.0*A4*(X1**3.0) + 5.0*A5*(X1**4.0)
    
    PC= np.exp(SC/(TC*(1.0E0+A6*X1+A7*X2))-X1/(A8*X2+A9))
    P=PC*2.212E7
    
    fac1 = (DSCDX1*DX1DTC*DTCDT)/(TC*(1.0E0+A6*X1+A7*X2))
    fac2a = 1.0+A6+A7
    fac2b = - 2.0*(A6+2.0*A7)*TC
    fac2c = 3.0*A7*(TC**2.0)
    fac2 = - DTCDT*(SC)*(fac2a + fac2b + fac2c)/((TC*(1.0E0+A6*X1+A7*X2))**2.0)
    fac3a = (A8*X2+A9)
    fac3 = DTCDT/fac3a - (X1/(fac3a**2.0))*(DX2DX1*DTCDT)
    
    DPDT = P*(fac1 + fac2 + fac3)
  
    return DPDT
    

#T = 1.974275987574E+02 # Temperature
#P = evalSatP(T)
    
#print 'Saturation Pressure'
#print '%.12e' %P
#print 'at'
#print 'T='
#print T

