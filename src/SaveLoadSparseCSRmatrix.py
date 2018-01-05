import scipy as sp
from scipy.sparse import csr_matrix

# Functions for saving and loading a sparse CSR matrix:

def save_sparse_csr(filename,array):
    sp.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = sp.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])