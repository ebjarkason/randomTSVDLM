# Functions for calling TSVD routines and evaluating the dimensionless 
# sensitivity matrix to matrix products
# Coded by: Elvar K. Bjarkason (2017)

import scipy as sp
from calcAdj import STtimesMatrix as STtimesH
from calcDir import StimesMatrix as StimesH
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from functools import partial
from TSVDroutines import *

# LinvTimesMatVec: 
# Evaluates L^(-1) times a vector or matrix X, where L is a Cholesky
# matrix of the inverse of the Regularization matrix R.  
# ----------------------------------------------------------------------
# INPUTS:
# XX: (Nm by s) vector or matrix.
# ----------------------------------------------------------------------
# RETURNS: 
# L^(-1) times XX
# ----------------------------------------------------------------------
def LinvTimesMatVec(XX):
    # Load cholesky matrix of inverse of L:
    cholLinv = sp.load('cholLinv.npy')
    XX = cholLinv.dot(XX)
    return XX

# LtimesMatVec: 
# Evaluates L or L^T times a vector or matrix X, where L is a Cholesky
# matrix of the inverse of the Regularization matrix R.     
# ----------------------------------------------------------------------
# INPUTS:
# Y: (Nm by s) vector or matrix.
# TRANS: TRANS=0 or TRANS='T'
# ----------------------------------------------------------------------
# RETURNS: 
# L times YY, when TRANS=0
# L^T times YY, when TRANS='T'
# ----------------------------------------------------------------------
def LtimesMatVec(YY, TRANS=0): # Evaluates L times a vector or matrix X
    # Load cholesky matrix of inverse of L:
    cholLinv = sp.load('cholLinv.npy')    
    YY2 = solve_triangular(cholLinv, YY, trans=TRANS , lower=False , unit_diagonal=False, overwrite_b=False, debug=False, check_finite=False) 
    return YY2

# LtimesMatVec: 
# Evaluates CD^(-0.5) times a vector or matrix X,
# where CD is the observation covariance matrix.  
# ----------------------------------------------------------------------
# INPUTS:
# dd: (Nd by s) vector or matrix.
# ----------------------------------------------------------------------
# RETURNS: 
# CD^(-0.5) times dd
# ----------------------------------------------------------------------
def multCDinvSQRT(dd):
    CdInv = sp.load('CdInvSQRT.npy')
    dd = CdInv.dot(dd)
    return dd

# SDtimesVecOrMat: 
# Evaluates SD times a vector or matrix qq,
# where SD = CD^(-0.5) * S * L is the dimensionless sensitivity matrix
# ( CD is the observation covariance matrix, S is the sensitivity matrix
# and L is regularization Cholesky or SQRT matrix ).
# ----------------------------------------------------------------------
# INPUTS:
# qq: (Nm by s) vector or matrix.
# parameters: current model parameters
# MAT: MAT=False when qq is a vector and MAT=MAT when qq is a matrix.
# ----------------------------------------------------------------------
# RETURNS: 
# SD times qq
# ----------------------------------------------------------------------    
def SDtimesVecOrMat(qq, parameters, MAT=False):
    # Evaluate S_D times qq using the direct method:
    qq = LtimesMatVec(qq, TRANS=0)
    qq = StimesH(qq, parameters, Mat=MAT) # Direct solve for S times qq
    qq = multCDinvSQRT(qq)
    return qq

# SDTtimesVecOrMat: 
# Evaluates SD^T times a vector or matrix ww,
# where SD = CD^(-0.5) * S * L is the dimensionless sensitivity matrix
# ( CD is the observation covariance matrix, S is the sensitivity matrix
# and L is regularization Cholesky or SQRT matrix ).
# ----------------------------------------------------------------------
# INPUTS:
# ww: (Nd by s) vector or matrix.
# parameters: current model parameters
# MAT: MAT=False when ww is a vector and MAT=MAT when ww is a matrix.
# ----------------------------------------------------------------------
# RETURNS: 
# SD^T times ww
# ----------------------------------------------------------------------
def SDTtimesVecOrMat(ww, parameters, MAT=False):
    # Evaluate S_D transposed times ww using adjoint method:
    ww = STtimesH(multCDinvSQRT(ww), parameters, Mat=MAT)
    # Multiply with Cholesky matrix L^T:  
    ww = LtimesMatVec(ww, TRANS='T')
    return ww

# svdfunc: 
# Function for evaluating the TSVD of S_D using a specified TSVD method:    
# ----------------------------------------------------------------------
# INPUTS:
# parameters: current adjustable model parameters
# TSVDspecials: list of variables determining which TSVD method to use
# Nm: number of model parameters
# Nd: number of observations
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vhk, of A:
# sk: appromximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# VTk: (ktrunc by Nc) matrix, where the ith row approximates the ith right singular vector
# k: number of Lanczos iterations
# ----------------------------------------------------------------------
def svdfunc(parameters, ktrunc, TSVDspecials, Nm, Nd):
    TSVDmethod = TSVDspecials[0]
    LancIter = 0
    if TSVDmethod == 'Lanc':
        print 'Using LANCZOS method'
        # Set up LinearOperator for S_D and S_D^T times vectors:
        partSDmultVEC = partial(SDtimesVecOrMat, parameters=parameters, MAT=False) # For direct runs
        partSDTmultVEC = partial(SDTtimesVecOrMat, parameters=parameters, MAT=False) # For adjoint runs
        Amult = LinearOperator((Nd,Nm), matvec=partSDmultVEC, rmatvec=partSDTmultVEC, dtype = 'float64')
        # Find TSVD using Lanczos iteration:
        stol = TSVDspecials[1]
        nmax = TSVDspecials[2]
        sk, Uk, VTk, LancIter = TSVDlanczos(Amult, ktrunc, nmax, stol)
    elif TSVDmethod == 'SVcutLanc':
        print 'Using LANCZOS method with SV-CUT'
        # Set up LinearOperator for S_D and S_D^T times vectors:
        partSDmultVEC = partial(SDtimesVecOrMat, parameters=parameters, MAT=False) # For direct runs
        partSDTmultVEC = partial(SDTtimesVecOrMat, parameters=parameters, MAT=False) # For adjoint runs
        Amult = LinearOperator((Nd,Nm), matvec=partSDmultVEC, rmatvec=partSDTmultVEC, dtype = 'float64')
        # Find TSVD using Lanczos iteration:
        stol = TSVDspecials[1]
        SVcut = TSVDspecials[2]
        ktruncmax = TSVDspecials[5]
        nmax = TSVDspecials[6]
        sk, Uk, VTk, LancIter, ktrunc = TSVDlanczosSVCUT(Amult, ktrunc, nmax, stol, SVcut, ktruncmax)    
    else:
        # Set up LinearOperator for S_D and S_D^T times vectors:
        partSDTmult = partial(SDTtimesVecOrMat, parameters=parameters, MAT=True)
        partSDTmultVEC = partial(SDTtimesVecOrMat, parameters=parameters, MAT=False)
        partSDmult = partial(SDtimesVecOrMat, parameters=parameters, MAT=True)
        partSDmultVEC = partial(SDtimesVecOrMat, parameters=parameters, MAT=False)
        # Determine whether to apply the randomized 1- or 2-view methods to S_D or its transpose.
        # For simplicity this is determined in the same way for the 1-view and 2-view methods,
        # though this might not be the best strategy for the 1-view method.
        if Nd >= Nm: # Apply the view method to S_D
            Amult = LinearOperator((Nd,Nm), matvec=partSDmultVEC, matmat=partSDmult, dtype = 'float64')
            ATmult = LinearOperator((Nm,Nd), matvec=partSDTmultVEC, matmat=partSDTmult, dtype = 'float64')
        else: # Apply the randomize method to S_D Transposed
            Amult = LinearOperator((Nm,Nd), matvec=partSDTmultVEC, matmat=partSDTmult, dtype = 'float64')
            ATmult = LinearOperator((Nd,Nm), matvec=partSDmultVEC, matmat=partSDmult, dtype = 'float64')
            
        if TSVDmethod == '1view':
            print 'Using 1-VIEW method'
            ell1 = TSVDspecials[1]
            ell2 = TSVDspecials[2]
            sk, Uk, VTk = TSVDrand1view(Amult, ATmult, ktrunc, ell1, ell2)
        elif TSVDmethod == 'EVENview':
            ptrunc = TSVDspecials[1]  
            qpow = TSVDspecials[2]  
            print 'Using '+str(2 + (2*qpow))+'-VIEW method'
            sk, Uk, VTk = TSVDrandEVENview(Amult, ATmult, ktrunc, ptrunc, qpow=qpow)
        else:
            print 'No TSVD method specified > EXIT'
            raise SystemExit
            
        if Nd < Nm:
            Uk, VTk = VTk.transpose(), Uk.transpose()
    
    return sk, Uk, VTk, LancIter, ktrunc


# svdfuncWithSubReuse: 
# Function for evaluating the TSVD of S_D using a specified randomized TSVD method,
# with subspace re-use
# ----------------------------------------------------------------------
# INPUTS:
# parameters: current adjustable model parameters
# TSVDspecials: list of variables determining which TSVD method to use
# Nm: number of model parameters
# Nd: number of observations
# UkPrev: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector from previous LM iteration
# VTkPrev: (ktrunc by Nc) matrix, where the ith row approximates the ith right singular vector from previous LM iteration
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vhk, of A:
# sk: appromximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vhk: (ktrunc by Nc) matrix, where the ith row approximates the ith right singular vector
# k: number of Lanczos iterations
# ----------------------------------------------------------------------
def svdfuncWithSubReuse(parameters, ktrunc, TSVDspecials, Nm, Nd, UkPrev, VTkPrev):
    TSVDmethod = TSVDspecials[0]
    LancIter = 0
    # Set up LinearOperator for S_D and S_D^T times vectors:
    partSDTmult = partial(SDTtimesVecOrMat, parameters=parameters, MAT=True)
    partSDTmultVEC = partial(SDTtimesVecOrMat, parameters=parameters, MAT=False)
    partSDmult = partial(SDtimesVecOrMat, parameters=parameters, MAT=True)
    partSDmultVEC = partial(SDtimesVecOrMat, parameters=parameters, MAT=False)
    # Determine whether to apply the randomized 1- or 2-view methods to S_D or its transpose.
    # For simplicity this is determined in the same way for the 1-view and 2-view methods,
    # though this might not be the best strategy for the 1-view method.
    if Nd >= Nm: # Apply the view method to S_D
        Amult = LinearOperator((Nd,Nm), matvec=partSDmultVEC, matmat=partSDmult, dtype = 'float64')
        ATmult = LinearOperator((Nm,Nd), matvec=partSDTmultVEC, matmat=partSDTmult, dtype = 'float64')
    else: # Apply the randomize method to S_D Transposed
        Amult = LinearOperator((Nm,Nd), matvec=partSDTmultVEC, matmat=partSDTmult, dtype = 'float64')
        ATmult = LinearOperator((Nd,Nm), matvec=partSDmultVEC, matmat=partSDmult, dtype = 'float64')
        
    if TSVDmethod == '1view':
        print 'Using 1-VIEW method', 'WITH SUBSPACE RE-USE'
        ell1 = TSVDspecials[1]
        ell2 = TSVDspecials[2]
        if Nd >= Nm: # Apply the view method to S_D
            sk, Uk, VTk = TSVDrand1viewSubReuse(Amult, ATmult, ktrunc, ell1, ell2, VTkPrev.transpose(), UkPrev.transpose())
        else:  # Apply the randomize method to S_D Transposed
            sk, Uk, VTk = TSVDrand1viewSubReuse(Amult, ATmult, ktrunc, ell1, ell2, UkPrev, VTkPrev)
                        
    elif TSVDmethod == 'EVENview':
        ptrunc = TSVDspecials[1]  
        qpow = TSVDspecials[2]  
        print 'Using '+str(2 + (2*qpow))+'-VIEW method', 'WITH SUBSPACE RE-USE'
        if Nd >= Nm: # Apply the view method to S_D
            sk, Uk, VTk = TSVDrandEVENviewSubReuse(Amult, ATmult, ktrunc, ptrunc, VTkPrev.transpose(), qpow=qpow)
        else:  # Apply the randomize method to S_D Transposed
            sk, Uk, VTk = TSVDrandEVENviewSubReuse(Amult, ATmult, ktrunc, ptrunc, UkPrev, qpow=qpow)
    else:
        print 'No TSVD method specified > EXIT'
        raise SystemExit
        
    if Nd < Nm:
        Uk, VTk = VTk.transpose(), Uk.transpose()
    
    return sk, Uk, VTk, LancIter


