# TSVD routines
# Coded by: Elvar K. Bjarkason (2017)

import scipy as sp
import numpy as np
import multiprocessing
from multiprocessing import Pool
from scipy.sparse.linalg import LinearOperator, aslinearoperator

    
# TSVDrandEVENview: 
# A basic randomized algorithm for estimating a TSVD, see Halko et al. (2011),
# "Finding structure with randomness: Proba11bilistic algorithms for 
# constructing approximate matrix decompositions".    
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) LinearOperator (for A*matrix and AT*matrix) 
# AT: (Nc by Nr) LinearOperator (for AT*matrix and A*matrix) or matrix
# (A and AT can also be specified as a dense/sparse matrix (for test purposes),
# but the below algorithm should be adjusted for that purpose)
# ktrunc: number of retained singular values
# ell: (integer) amount of oversampling
# pow: number of optional power iterations. Default (pow=0) is using a 2-view approach.
# The number of matrix views are 2*(1+qpow).
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vhk, of A:
# sk: appromximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vhk: (ktrunc by Nc) matrix, where the ith row approximates the ith right singular vector
# ----------------------------------------------------------------------
def TSVDrandEVENview(A, AT, ktrunc, ell, qpow=0):
    A = aslinearoperator(A)
    AT = aslinearoperator(AT)
    Nr, Nc = A.shape
    
    # Generate random Gaussian sampling matrix:
    Omega = sp.random.randn(Nc, ktrunc + ell) # can also sample from uniform distribution using rand()
    
    # Approximate range Y = A*Omega:
    Y = A.matmat(Omega) 
    
    # Perform optional power iteration for improved estimates:
    for qpowit in range(0,qpow):
        # Orthonormalize:
        Y = sp.linalg.qr(Y, mode='economic')[0]
        # Find Z = A^T *Y:
        Z = AT.matmat(Y) 
        # Orthonormalize:
        Z = sp.linalg.qr(Z, mode='economic')[0]
        # Find Y = A*Z:
        Y = A.matmat(Z) 
    
    # Find orthonormal matrix using QR deomposition such that Y = QR:
    Q = sp.linalg.qr(Y, mode='economic')[0]
    
    # Find B = QT*A or BT = A^T *Q:
    BT = AT.matmat(Q) 
    
    # Find SVD of the relatively small matrix B:
    [Uhat, s, Vh] = sp.linalg.svd(BT.transpose(), full_matrices = False, compute_uv=True, overwrite_a=False, check_finite=True)
    # Form U = Q*Uhat and truncate to the desired level:
    Uk = Q.dot(Uhat[:,0:ktrunc]) # Left singular vectors for A
    Vhk = Vh[0:ktrunc,:] # Right singular vectors for A
    sk = s[0:ktrunc] # Singular values for A
        
    return sk, Uk, Vhk

# SomeMatTimesSomeMat:
# A function used within the 1-view method when using multiprocessing to 
# find A*Omega and AT*PsiT simultaneously in parallel (i.e. for parallel
# adjoint and direct solves).
# ----------------------------------------------------------------------
# INPUTS:
# inputs = [], 
# where
# Mat: (Nr by Nc) LinearOperator
# B: (Nc by s) matrix
# ----------------------------------------------------------------------
# # RETURNS:
# Mat * B
# ----------------------------------------------------------------------
def SomeMatTimesSomeMat(inputs):
    Mat, B = inputs
    return Mat.matmat(B)
    
# TSVDrand1view: 
# A randomized 1-view method for estimating a TSVD, see Tropp et al. (2016),
# "Randomized single-view algorithms for low-rank matrix approximation".  
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) LinearOperator (for A*matrix and AT*matrix) 
# AT: (Nc by Nr) LinearOperator (for AT*matrix and A*matrix) or matrix
# (A and AT can also be specified as a dense/sparse matrix (for test purposes),
# but the below algorithm should be adjusted for that purpose)
# ktrunc: number of retained singular values
# ell1: (integer) amount of oversampling for range
# ell2: (integer) amount of oversampling for co-range (ell2 >= ell1 >= 0)
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vhk, of A:
# sk: appromximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vhk: (ktrunc by Nc) matrix, where the ith row approximates the ith right singular vector
# ----------------------------------------------------------------------    
def TSVDrand1view(A, AT, ktrunc, ell1, ell2):
    A = aslinearoperator(A)
    AT = aslinearoperator(AT)
    Nr, Nc = A.shape
    
    # Generate random Gaussian sampling matrices:
    Omega = sp.random.randn(Nc, ktrunc + ell1)
    Psi = sp.random.randn(ktrunc + ell2, Nr) 
    
    # Orthogonalize:
    Omega = sp.linalg.orth(Omega)
    Psi = (sp.linalg.orth(Psi.transpose())).transpose()
    
    # Find Y = A*Omega and ZT = A^T *Psi^T simultaneously in parallel.
    # Parallel evaluation of range and co-range:
    pool = Pool(processes=2)
    SketchResults = pool.map( SomeMatTimesSomeMat , [ [A, Omega], [AT, Psi.transpose()] ])
    pool.terminate()
    [Y, ZT]  = SketchResults
    
    # Find orthonormal matrix using QR deomposition such that Y = QR:
    Q = sp.linalg.qr( Y , mode='economic')[0]
    
    [U,T] = sp.linalg.qr( Psi.dot(Q) , mode='economic')
    X = sp.linalg.solve( T , sp.dot(U.transpose() , ZT.transpose()) )
    
    # Find SVD of the relatively small matrix X:
    [Uhat, s, Vh] = sp.linalg.svd(X, full_matrices = False, compute_uv=True, overwrite_a=False, check_finite=True)
    # Form U = Q*Uhat and truncate to the desired level:
    Uk = Q.dot(Uhat[:,0:ktrunc]) # Left singular vectors for A
    Vhk = Vh[0:ktrunc,:] # Right singular vectors for A
    sk = s[0:ktrunc] # Singular values for A
        
    return sk, Uk, Vhk


# TSVDlanczos:
# A basic Lanczos algorithm for estimating a TSVD, see Vogel & Wade (1994),
# "Iterative SVD-based methods for ill-posed problems".    
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) LinearOperator (for A*vector and AT*vector) or matrix
# ktrunc: number of retained singular values
# nmax: maximum number of excess Lanczos iterations
# stol: convergence tolerance
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vhk, of A:
# sk: appromximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vhk: (ktrunc by Nc) matrix, where the ith row approximates the ith right singular vector
# k: number of Lanczos iterations
# ----------------------------------------------------------------------
def TSVDlanczos(A, ktrunc, nmax, stol):
    kmax = ktrunc + nmax
    A = aslinearoperator(A)
    Nr, Nc = A.shape
    
    Q = sp.zeros((Nc,kmax+1))
    U = sp.zeros((Nr,kmax+1))
    Tk = sp.zeros((kmax+1,kmax+1))
    # Initialize with some unit vector of length Nm:
    q = sp.random.randn(Nc)
    q = q/sp.linalg.norm(q)
    
    # Evaluate A times q (A = S_D: direct simulation):
    y = A.matvec(q)
    
    alpha = sp.linalg.norm(y)
    u = y/alpha
    Tk[0,0] = alpha
    
    notconverged = True
    k = 0
    sold = sp.zeros(ktrunc) 
    Q[:,k] = q
    U[:,k] = u
    while (k<kmax)and(notconverged):
        Qk = Q[:,0:(k+1)]
        Uk = U[:,0:(k+1)]
        
        # Evaluate A^T times u (A = S_D: adjoint simulation):
        w = A.rmatvec(u)
        w = w - alpha*q
        # Reorthogonalize:
        w = w - Qk.dot(sp.dot(Qk.transpose(),w))
        
        beta = sp.linalg.norm(w)
        q = w/beta
        
        # Evaluate A times q (A = S_D: direct simulation):
        y = A.matvec(q)
        y = y - beta*u
        # Reorthogonalize:
        y = y - Uk.dot(sp.dot(Uk.transpose(),y))
        
        alpha = sp.linalg.norm(y)
        u = y/alpha
        k += 1
        Tk[k,k] = alpha
        Tk[k-1,k] = beta
        Q[:,k] = q
        U[:,k] = u
        
        if (k>=ktrunc): # Estimate singular values
            [Usvd,s,Vhsvd] = sp.linalg.svd(Tk[0:k+1,0:k+1], full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            sk = s[0:ktrunc]
            skratios = np.absolute(sold[0:ktrunc]-sk)/sk
            # Check for convergence:
            svtolConditon = np.all( skratios <= stol )
            if svtolConditon :
                notconverged = False
            sold = s
    # Truncate and estimate singular vectors:
    Uk = U[:,0:k+1].dot(Usvd[:,0:ktrunc])
    Vk = Q[:,0:k+1].dot(Vhsvd[0:ktrunc,:].transpose())
    Vhk = Vk.transpose()
    
    return sk, Uk, Vhk, k
    
# TSVDlanczosSVCUTs:
# Using SVcut, a basic Lanczos algorithm for estimating a TSVD, see Vogel & Wade (1994),
# "Iterative SVD-based methods for ill-posed problems".    
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) LinearOperator (for A*vector and AT*vector) or matrix
# ktrunc: number of retained singular values
# nmax: maximum number of excess Lanczos iterations
# stol: convergence tolerance
# SVcut: singular value cut-off parameters
# ktruncmax: upper limit for the number of retained singular values
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vhk, of A:
# sk: appromximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vhk: (ktrunc by Nc) matrix, where the ith row approximates the ith right singular vector
# k: number of Lanczos iterations
# ktrunc: number of retained singular values
# ----------------------------------------------------------------------
def TSVDlanczosSVCUT(A, ktrunc, nmax, stol, SVcut, ktruncmax) :
    kmax = ktrunc + nmax
    A = aslinearoperator(A)
    Nr, Nc = A.shape
    
    Q = sp.zeros((Nc,kmax+1))
    U = sp.zeros((Nr,kmax+1))
    Tk = sp.zeros((kmax+1,kmax+1))
    # Initialize with some unit vector of length Nm:
    q = sp.random.randn(Nc)
    q = q/sp.linalg.norm(q)
    
    # Evaluate A times q (A = S_D: direct simulation):
    y = A.matvec(q)
    
    alpha = sp.linalg.norm(y)
    u = y/alpha
    Tk[0,0] = alpha
    
    notconverged = True
    k = 0
    sold = sp.zeros(ktruncmax) 
    Q[:,k] = q
    U[:,k] = u
    while (k<kmax)and(notconverged):
        Qk = Q[:,0:(k+1)]
        Uk = U[:,0:(k+1)]
        
        # Evaluate A^T times u (A = S_D: adjoint simulation):
        w = A.rmatvec(u)
        w = w - alpha*q
        # Reorthogonalize:
        w = w - Qk.dot(sp.dot(Qk.transpose(),w))
        
        beta = sp.linalg.norm(w)
        q = w/beta
        
        # Evaluate A times q (A = S_D: direct simulation):
        y = A.matvec(q)
        y = y - beta*u
        # Reorthogonalize:
        y = y - Uk.dot(sp.dot(Uk.transpose(),y))
        
        alpha = sp.linalg.norm(y)
        u = y/alpha
        k += 1
        Tk[k,k] = alpha
        Tk[k-1,k] = beta
        Q[:,k] = q
        U[:,k] = u
        
        [Usvd,s,Vhsvd] = sp.linalg.svd(Tk[0:k+1,0:k+1], full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
        
        # Find ktrunc, which is the smallest value such that s_ktrunc <= s_1 * SV-cut (i.e. s[ktrunc-1] <= s[0] * SV-cut )
        s1 = s[0]
        slast = s[-1]
        LowHighSVratio = slast/s1
        if (LowHighSVratio <= SVcut)or(k+1 >= ktruncmax):
            if LowHighSVratio > SVcut:
                ktrunc = ktruncmax
            else:
                LHratios = s/s1
                ktrunc = next( idx for idx, value in enumerate(LHratios) if value <= SVcut ) + 1
                if (ktrunc > ktruncmax):
                    ktrunc = ktruncmax
        
            sk = s[0:ktrunc]
            skratios = np.absolute(sold[0:ktrunc]-sk)/sk
            svtolConditon = np.all( skratios <= stol )
            if svtolConditon :
                notconverged = False
        if k+1 >= ktruncmax:
            sold = s[0:ktruncmax]
        else:
            sold[0:(k+1)] = s
    # Truncate and estimate singular vectors:
    Uk = U[:,0:k+1].dot(Usvd[:,0:ktrunc])
    Vk = Q[:,0:k+1].dot(Vhsvd[0:ktrunc,:].transpose())
    Vhk = Vk.transpose()
    
    return sk, Uk, Vhk, k, ktrunc

   
    
      
# TSVDrandEVENviewSubReuse: 
# A randomized algorithm for estimating a TSVD applying subspace re-use
# to generate the randomized sampling matrix
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) LinearOperator (for A*matrix and AT*matrix) 
# AT: (Nc by Nr) LinearOperator (for AT*matrix and A*matrix) or matrix
# (A and AT can also be specified as a dense/sparse matrix (for test purposes),
# but the below algorithm should be adjusted for that purpose)
# ktrunc: number of retained singular values
# ell: (integer) amount of oversampling
# OmegaSub: (Nc by NSub) matrix used to form part of the sampling matrix for the range finder (NSub <= ktrunc)
# pow: number of optional power iterations. Default (pow=0) is using a 2-view approach.
# The number of matrix views are 2*(1+qpow).
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vhk, of A:
# sk: appromximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vhk: (ktrunc by Nc) matrix, where the ith row approximates the ith right singular vector
# ----------------------------------------------------------------------
def TSVDrandEVENviewSubReuse(A, AT, ktrunc, ell, OmegaSub, qpow=0):
    A = aslinearoperator(A)
    AT = aslinearoperator(AT)
    Nr, Nc = A.shape
    
    # Generate random subspace re-use sampling matrix:
    Omega = sp.zeros((Nc, ktrunc + ell))
    NSub = OmegaSub.shape[1]
    Omega[:, 0:NSub] = OmegaSub
    Omega[:, NSub::] = sp.random.randn(Nc, ktrunc + ell - NSub) # can also sample from uniform distribution using rand()
    
    # Approximate range Y = A*Omega:
    Y = A.matmat(Omega) 
    
    # Perform optional power iteration for improved estimates:
    for qpowit in range(0,qpow):
        # Orthonormalize:
        Y = sp.linalg.qr(Y, mode='economic')[0]
        # Find Z = A^T *Y:
        Z = AT.matmat(Y) 
        # Orthonormalize:
        Z = sp.linalg.qr(Z, mode='economic')[0]
        # Find Y = A*Z:
        Y = A.matmat(Z) 
    
    # Find orthonormal matrix using QR deomposition such that Y = QR:
    Q = sp.linalg.qr(Y, mode='economic')[0]
    
    # Find B = QT*A or BT = A^T *Q:
    BT = AT.matmat(Q) 
    
    # Find SVD of the relatively small matrix B:
    [Uhat, s, Vh] = sp.linalg.svd(BT.transpose(), full_matrices = False, compute_uv=True, overwrite_a=False, check_finite=True)
    # Form U = Q*Uhat and truncate to the desired level:
    Uk = Q.dot(Uhat[:,0:ktrunc]) # Left singular vectors for A
    Vhk = Vh[0:ktrunc,:] # Right singular vectors for A
    sk = s[0:ktrunc] # Singular values for A
        
    return sk, Uk, Vhk
    
# TSVDrand1viewSubReuse: 
# A randomized 1-view algorithm for estimating a TSVD applying subspace re-use
# to generate the randomized sampling matrix  
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) LinearOperator (for A*matrix and AT*matrix) 
# AT: (Nc by Nr) LinearOperator (for AT*matrix and A*matrix) or matrix
# (A and AT can also be specified as a dense/sparse matrix (for test purposes),
# but the below algorithm should be adjusted for that purpose)
# ktrunc: number of retained singular values
# ell1: (integer) amount of oversampling for range
# ell2: (integer) amount of oversampling for co-range (ell2 >= ell1 >= 0)
# OmegaSub: (Nc by NSub) matrix used to form part of the sampling matrix for the range finder (NSub <= ktrunc)
# PsiSub: (NSub by Nr) matrix used to form part of the sampling matrix for the co-range finder (NSub <= ktrunc)
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vhk, of A:
# sk: appromximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vhk: (ktrunc by Nc) matrix, where the ith row approximates the ith right singular vector
# ----------------------------------------------------------------------    
def TSVDrand1viewSubReuse(A, AT, ktrunc, ell1, ell2, OmegaSub, PsiSub):
    A = aslinearoperator(A)
    AT = aslinearoperator(AT)
    Nr, Nc = A.shape
    
    # Generate random subspace re-use sampling matrices:
    Omega = sp.zeros((Nc, ktrunc + ell1))
    NSub = OmegaSub.shape[1]
    Omega[:, 0:NSub] = OmegaSub
    Omega[:, NSub::] = sp.random.randn(Nc, ktrunc + ell1 - NSub) # can also sample from uniform distribution using rand()
    Psi = sp.zeros((ktrunc + ell2, Nr))
    Psi[0:NSub, :] = PsiSub
    Psi[NSub::, ] = sp.random.randn(ktrunc + ell2 - NSub, Nr) 
    
    # Orthogonalize:
    Omega = sp.linalg.orth(Omega)
    Psi = (sp.linalg.orth(Psi.transpose())).transpose()
    
    # Find Y = A*Omega and ZT = A^T *Psi^T simultaneously in parallel.
    # Parallel evaluation of range and co-range:
    pool = Pool(processes=2)
    SketchResults = pool.map( SomeMatTimesSomeMat , [ [A, Omega], [AT, Psi.transpose()] ])
    [Y, ZT]  = SketchResults
    pool.terminate()
    
    # Find orthonormal matrix using QR deomposition such that Y = QR:
    Q = sp.linalg.qr( Y , mode='economic')[0]
    
    [U,T] = sp.linalg.qr( Psi.dot(Q) , mode='economic')
    X = sp.linalg.solve( T , sp.dot(U.transpose() , ZT.transpose()) )
    
    # Find SVD of the relatively small matrix X:
    [Uhat, s, Vh] = sp.linalg.svd(X, full_matrices = False, compute_uv=True, overwrite_a=False, check_finite=True)
    # Form U = Q*Uhat and truncate to the desired level:
    Uk = Q.dot(Uhat[:,0:ktrunc]) # Left singular vectors for A
    Vhk = Vh[0:ktrunc,:] # Right singular vectors for A
    sk = s[0:ktrunc] # Singular values for A
        
    return sk, Uk, Vhk




