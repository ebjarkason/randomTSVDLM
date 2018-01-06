# The Randomized T-SVD Levenberg-Marqardt method
# Coded by: Elvar K. Bjarkason (2017)

import scipy as sp
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_triangular
import shutil
import time
from evalTSVD import *

    
# TSVDparamUpdate: 
# Approximate LM model update using TSVD of dimensionless sensivity matrix.    
# ----------------------------------------------------------------------
# INPUTS:
# rm: weighted observation residual vector of length Nd
# Nm: number of model parameters
# ktrunc: number of retained singular values
# sk: appromximate top ktrunc singular values of A
# Uk: (Nd by ktrunc) matrix, where the ith column approximates the ith left singular vector
# VTk: (ktrunc by Nm) matrix, where the ith row approximates the ith right singular vector
# mtransformed: prior preconditioned model parameters
# beta: regularization weight
# mu: LM damping factor
# ----------------------------------------------------------------------
# RETURNS:
# dm: trial model update, in the original parameter space
# ----------------------------------------------------------------------
def TSVDparamUpdate(rm, Nm, ktrunc, sk, Uk, VTk, mtransformed, beta, mu):
    dm = sp.zeros(Nm)
    for i in range(0,ktrunc):
        alphai = 0
        ski = sk[i]
        vk = VTk[i,:]
        alphai += beta * sp.dot(vk , mtransformed)
        alphai += ski * sp.dot(Uk[:,i] , rm)
        dm += - sp.dot(alphai/(beta + mu + ski**2.) , vk)
    
    # rescale the search direction, to give update in the original parameter space
    dm = LtimesMatVec(dm, TRANS=0)
    return dm

# UpdateParamsAndTruncate2Bounds: 
# A simplistic scheme for truncating the model parameters to be within
# a specified  rectangular domain.    
# ----------------------------------------------------------------------
# INPUTS:
# mm: vector of length Nm for current model parameters
# dmm: vector of length Nm for suggested model update step
# ubounds: vector of length Nm for upper parameter bounds
# lbounds: vector of length Nm for lower parameter bounds
# ----------------------------------------------------------------------
# RETURNS:
# mmtemp: vector of length Nm for possible parameter update truncated to bounds
# dmm: vector of length Nm for model update step after truncating to bounds
# ----------------------------------------------------------------------
def UpdateParamsAndTruncate2Bounds(mm, dmm, ubounds, lbounds):
    # Update model parameters:
    mmtemp = mm + dmm
    print 'Checking paramater bounds'
    # Truncate the model parameters to be within the parameter bounds:
    # Checking upper bounds
    parcount = 0
    for par in mmtemp:
        if par > ubounds[parcount]:
            mmtemp[parcount] = ubounds[parcount]
        parcount += 1
    # Checking lower bounds
    parcount = 0
    for par in mmtemp:
        if par < lbounds[parcount]:
            mmtemp[parcount] = lbounds[parcount]
        parcount += 1
    # Edit the parameter updgrade accordingly:
    dmm = mmtemp - mm
    return mmtemp, dmm


# LMinvs: 
# Main routine for the TSVD-LM method.    
# ----------------------------------------------------------------------
# INPUTS:
# m0: vector of length Nm, prior guess for model parameters
# resfunc: function for running nonlinear forward simulation for given model parameters
# and returns the relevant weighted observation residual vector
# TSVDspecials: list of variables determining which TSVD method to use
# kmax: maximum number of LM iterations
# e2: LM convergence tolerance
# mu0: initial LM damping factor
# ubounds: vector of length Nm for upper parameter bounds
# lbounds: vector of length Nm for lower parameter bounds
# lamredfac: reduce LM damping factor to mu/lamredfac when update succeeds
# lamincrfac: increase LM damping factor to max( mu*lamincrfac , 1.e2)  when update fails
# beta: regularization weight
# ktrunc: number of retianed singular values for first LM iteration
# ktruncIncr: increase the number of retainted singular valeus (ktrunc) by ktruncIncr between LM iterations
# ktruncmax: maximum number of retained singular values
# timeClock: reference time when inversion began
# timeStart: reference time when inversion began
# Nm: number of model parameters
# Nd: number of observations
# inconName: name of TOUGH2 INCON file to be copied to most up to date INCON file
# 'incbest.incon' when a model parameter update has been accepted
# SubReuse: Used when applying randomized methods with subspace re-use. 
# Applies subspace re-use when SubRuse=True.
# ----------------------------------------------------------------------
# RETURNS:
# m: vector of length Nm for final model parameters
# fval: value of the objective function
# LMiter: number of LM iterations
# ----------------------------------------------------------------------
def LMinv(m0, resfunc, TSVDspecials, kmax, e2, mu0, ubounds, lbounds, lamredfac, lamincrfac, beta, ktrunc, ktruncIncr, ktruncmax, timeClock, timeStart, Nm, Nd, inconName, SubReuse=False):

    print '-------------------------------------------------------' 
    print 'Begin TSVD-LM Algorithm'
    print '-------------------------------------------------------'
    print 'TSVD scheme: ', TSVDspecials[0]
    if SubReuse:
        print 'WITH SUBSPACE RE-USE'
    print '-------------------------------------------------------'
  
    # Initialize parameters:
    k = 1  
    mu = mu0
    found = 0
    sp.save('LMiter.npy', k)
    m = m0
    
    temptime = time.clock()
    rm = resfunc(m) # Run nonlinear forward simulation and return weighted observation residual rm
    rcalls = 1  # Counter for number of forward simulations
    # Update best incon file:
    shutil.copy (inconName, 'incbest.incon')
    print 'TIME spent on calling resfunc : ', time.clock() - temptime 
    
    temptime = time.clock()
    print 'Estimating truncated SVD of the dimensionless sensitivity matrix S_D'
    # Evaluate truncated singular triplets:
    sk, Uk, VTk, LancIter, ktrunc = svdfunc(m, ktrunc, TSVDspecials, Nm, Nd)
    Jcalls = 1  # Counter for calls to TSVD function
    print 'TIME spent on finding a truncated SVD calling svdfunc : ', time.clock() - temptime 
    
    # Transform the model parameters:
    mtransformed = LinvTimesMatVec(m - m0)
    
    # Observation mismatch:
    Od = sp.dot(rm, rm)
    # Model mismatch:
    Om = sp.dot(mtransformed, mtransformed)
    # Current value of objective function:
    fold = Od + beta*Om

    notconverged = True
    while notconverged :
        
        temptime = time.clock()
        print 'Solving for possible parameter upgrade dm'
        dm = TSVDparamUpdate(rm, Nm, ktrunc, sk, Uk, VTk, mtransformed, beta, mu) 
        print 'TIME spent on linear update of model parameters : ', time.clock() - temptime
        
        temptime = time.clock()
        # Find possible update model parameters:
        mtemp, dm = UpdateParamsAndTruncate2Bounds(m, dm, ubounds, lbounds)   
        print 'TIME spent on checking parameter bounds and correcting accordingly : ', time.clock() - temptime
        
        if sp.linalg.norm(dm) <= e2*(sp.linalg.norm(m) + e2):
            found = 1
            notconverged = False
        else:
            print 'Finding the updated observation and regularization residuals'
            temptime = time.clock()
            rtemp = resfunc(mtemp)  # Run nonlinear forward simulation and return weighted observation residual rtemp
            print 'TIME spent running second simulation', time.clock() - temptime
            rcalls = rcalls + 1
            
            # Transformed parametes:
            mtempTransformed = mtransformed + LinvTimesMatVec(dm)
            
            Odtemp = sp.dot(rtemp, rtemp)
            Omtemp = sp.dot(mtempTransformed, mtempTransformed)
            ftemp = Odtemp + beta*Omtemp
            
            print 'Iter '+str(k)+', Od = '+str(Od)+ ', Odtemp = '+str(Odtemp)+', Objnew/Objprev= '+str(ftemp/fold)
            
            # Save some run statisics to file:
            fstat = open('runstats.txt','a')
            stuff = [k, fold, ftemp, Od, Odtemp, beta, mu, ktrunc, sk[0], sk[-1], Om, Omtemp, time.clock()-timeClock, time.time()-timeStart, LancIter]
            strstuff = ['k', 'fold', 'ftemp', 'Od', 'Odtemp', 'beta', 'mu', 'ktrunc', 'sk[0]', 'sk[-1]', 'Om', 'Omtemp', 'timeClock', 'timeStart', 'LancIter']
            itemk = 0
            for item in stuff:
                fstat.write(strstuff[itemk]+' : ')
                fstat.write(str(item))
                fstat.write('   ')
                itemk += 1
            fstat.write('\n')
            fstat.close()
            
            print 'checking upgrade'
            if (ftemp < fold) :
                
                # Update best incon file when objective function is improved:
                shutil.copy (inconName, 'incbest.incon')
                
                k = k+1
                sp.save('LMiter.npy', k)
                
                print 'accepted upgrade'
                m = sp.copy(mtemp) 
                rm = sp.copy(rtemp)
                Od = Odtemp + 0.
                Om = Omtemp + 0.
                mtransformed = sp.copy(mtempTransformed)
                
                # update the LM lambda, which is named here mu:
                mu = mu/lamredfac
                
                if TSVDspecials[0] == 'SVcutLanc':
                    # Update sv-cut truncation factor:
                    SVcut = TSVDspecials[2]
                    SVcutmin = TSVDspecials[3]
                    SVcutRedFac = TSVDspecials[4]
                    SVcut = max(SVcut*SVcutRedFac, SVcutmin)
                    TSVDspecials[2] = SVcut
                else:
                    # Update the SVD truncation factor:
                    ktrunc = min(ktrunc + ktruncIncr, ktruncmax )
                
                if (k > kmax):
                    found = 2
                    notconverged = False
                else:
                    temptime = time.clock()
                    print 'Estimating truncated SVD of the dimensionless sensitivity matrix S_D'
                    # Evaluate truncated singular triplets:
                    if SubReuse:
                        sk, Uk, VTk, LancIter = svdfuncWithSubReuse(m, ktrunc, TSVDspecials, Nm, Nd, Uk, VTk)
                    else:
                        sk, Uk, VTk, LancIter, ktrunc = svdfunc(m, ktrunc, TSVDspecials, Nm, Nd)
                    print 'TIME spent on finding a truncated SVD calling svdfunc : ', time.clock() - temptime 
                    Jcalls = Jcalls + 1 
                
                # Update fold:
                fold = Od + beta*Om
                
            else:
                print 'Upgrade not acceptable, increase mu'
                mu = max( mu*lamincrfac , 1.0e2)    

    fval = fold     # Return last function value
    LMiter = k-1

    print '-------------------------------------------------------' 
    print 'Statistics'
    if (k < kmax):
        print 'The point of minimum was found'
    else:
        print 'The point of minimum was not found'
      
    print 'Total number of iterations: ', LMiter
    print 'Total number of function evalutaions: ',rcalls
    print 'Total number of jacobian/gradient evaluations: ',Jcalls
    print 'Function value: ', fval



    return m, fval, LMiter
    
    
    
    
    
