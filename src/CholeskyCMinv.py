# Evaluate the Cholesky factors of the regularization matrix R = W^T W = CM^(-1):
# Coded by: Elvar K. Bjarkason (2017)

import scipy as sp
import numpy as np
import scipy.sparse as sparse
import evalRegularJacobian
import SaveLoadSparseCSRmatrix as slspCSR
from scipy.sparse import csr_matrix
from scipy.linalg import cholesky

# Evaluate the regularization Jacobian W:
print 'Evaluating regularization Jacobian'
NRadj = sp.load('numgridbal.npy')[7]
rJac = csr_matrix( evalRegularJacobian.regjac(NRadj) )
slspCSR.save_sparse_csr('regJac.npz',csr_matrix(rJac))
slspCSR.save_sparse_csr('regJacT.npz',csr_matrix(rJac.transpose()))


# Find SVD of CMinv:
print 'Finding CMinv'
CMinv = rJac.transpose().dot(rJac)
# Estimate upper triangular Cholesky matrix CMinv
# That is L^{-1}
print 'Estimating Cholesky of CMinv'
cholLinv = cholesky(CMinv.todense() , lower=False , overwrite_a=False , check_finite=True )
sp.save('cholLinv.npy',cholLinv)


