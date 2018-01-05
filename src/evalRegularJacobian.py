# Regularization Jacobian
# Coded by: Elvar K. Bjarkason (2017)

import scipy as sp
from scipy.sparse import csr_matrix, lil_matrix

# Generates the Regularization Jacobian for the 2D slice model with 8000 rock-types and 16,000 parameters.
# Could generalize for adjsutable permeabilities in every block this 
# by using grid connection information.
def regjac(NRadj):
    NRadjHalf = NRadj/2                 
    mpr = sp.load('mprior.npy') 
    Npr = len(mpr) 
     
    Nregsmooth = 2*15820 # 2 times the number of connections between adjustable rock-types
    Nreglocalxz = 8000 # Number of adjustable rock-types
    Nreg = Nregsmooth + Nreglocalxz  + Npr
    sp.save('Nreg.npy',Nreg)
    
    rJac = lil_matrix((Nreg,NRadj))
    x = 0
    
    # Create horizontal smoothing of log10kx (perm index 1):
    for i in range(0,80):
        for j in range(0,99):
            rJac[x,j + i*100] = 1
            rJac[x,j + i*100 + 1] = -1
            x += 1
        
    # Create vertical smoothing of log10kx (perm index 1):
    for i in range(0,79):
        for j in range(0,100):
            rJac[x,j + i*100] = 1
            rJac[x,j + (i+1)*100] = -1
            x += 1
    
    # Create horizontal smoothing of log10kz (perm index 3):
    for i in range(0,80):
        for j in range(0,99):
            rJac[x,j + i*100 + NRadjHalf] = 1
            rJac[x,j + i*100 + 1 + NRadjHalf] = -1
            x += 1
        
    ## Create vertical smoothing of log10kz (perm index 3):
    for i in range(0,79):
        for j in range(0,100):
            rJac[x,j + i*100 + NRadjHalf] = 1
            rJac[x,j + (i+1)*100 + NRadjHalf] = -1
            x += 1
    
    # Add regularization to make log10kx similar to log10kz:
    for i in range(0,Nreglocalxz):
        rJac[x,i] = 1
        rJac[x,i + NRadjHalf] = -1
        x += 1 
    
    # Add prior paramater regularisation:
    for i in range(0,Npr):
        rJac[x,i] = 1*0.001
        x += 1 

    return csr_matrix(rJac)





 