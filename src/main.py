# Main script for running inversions using the Randomized T-SVD Levenberg-Marqardt method:
# Coded by: Elvar K. Bjarkason (2017)

import os
import time
import scipy as sp
import TSVDLMalgo
from evalResidualsFromSim import resFromSim as residualsFromSim
import shutil


# runinversions: 
# Runs inversion(s) using TSVD-LM method using the TSVD method specified
# by TSVDspecials and SubReuse.
# ----------------------------------------------------------------------
# INPUTS:
# nrInversion: number of inversions to run
# subfolder: subfolder for saving some run statisitics
# TSVDspecials: list specifying which TSVD method to use
# SubReuse: SubReuse=True applies subspace re-use
# ----------------------------------------------------------------------
def runinversions(nrInversions, subfolder, TSVDspecials, SubReuse):
    # Specify files to be removed before running inversion:
    remfiles = ['runstats.txt']
    for afile in remfiles:
        if os.path.isfile(afile):
            os.remove(afile)
    
    
    for i in range(0, nrInversions):
        timeClock = time.clock()
        timeStart = time.time()
        
        temptime = time.clock()
        # Initialize to generate parameter, observation and grid information for
        # the adjoint and direct solves:
        os.system('python initialize4adjdir.py')
        print 'TIME spent on initialize4adjdir.py : ', time.clock() - temptime 
        
        temptime = time.clock()
        # Initialize model parameters:
        Nm, Nd, NRadj = sp.load('numgridbal.npy')[[4,5,7]]
        sp.save('NRadj.npy', NRadj)
        x0 = sp.zeros(NRadj) -14.0 # Initial parameter guess
        ub = sp.zeros((NRadj,1)) - 13.0 # Upper parameter bounds
        lb = sp.zeros((NRadj,1)) - 16.0 # Lower parameter bounds
        sp.save('mprior.npy', x0) # Save the initial or prior parameters
        # Create SQRT of the inverse observation covariance matrix (C_D)^(-1):
        sigmaTemps = 0.5E0
        sigmaPressEnth = 1.0e4
        InvsigmaD = sp.zeros(Nd)
        NatObs = 135 # Number of natural state temperature observations
        InvsigmaD[0:NatObs] = InvsigmaD[0:NatObs] + 1.0/sigmaTemps
        InvsigmaD[NatObs::] = InvsigmaD[NatObs::] + 1.0/sigmaPressEnth
        wmatrix = sp.diag(InvsigmaD)
        sp.save('CdInvSQRT.npy', wmatrix)
        print 'TIME spent on initial parameter conditions : ', time.clock() - temptime 
        
        temptime = time.clock()
        # Evaluate the Cholesky factors of the regularization matrix R = W^T W = CM^(-1):
        os.system('python CholeskyCMinv.py')
        print 'TIME spent on SVDofCMinv.py : ', time.clock() - temptime 
        
        
        temptime = time.clock()
        # Begin inversion
        print 'Running Inversion'   
        beta = 2.5          # Regularization weight
        lambda0 = 1.e6      # Initial LM damping factor
        ktrunc0 = 1         # Initial number of retianed singular values ktrunc
        ktruncIncr = 2      # Incremental increase in ktrunc
        ktruncmax = 50      # Maximum for ktrunc
        LMiterMax = 30      # Maximum number of LM iterations
        lamredfac = 10.0    # Reduction factor for LM damping factor
        lamincrfac = 10.0   # Factor for increasing LM damping factor
        
        # Save some run statisics to file:
        fstat = open('runstats.txt','a')
        fstat.write('Initilization time : '+str(time.clock()-timeClock))
        fstat.write('\n')
        fstat.close()
        
        # Run inversion by calling TSVD-LM routine:
        res = TSVDLMalgo.LMinv(x0, residualsFromSim, TSVDspecials, LMiterMax, 0.*1e-6, lambda0, ub, lb, lamredfac, lamincrfac, beta, ktrunc0, ktruncIncr, ktruncmax, timeClock, timeStart, Nm, Nd, 'incFromNat.incon', SubReuse)
        
        # Save some run statisics to file:
        fstat = open('runstats.txt','a')
        fstat.write('Total inversion time : '+str(time.clock()-timeClock))
        fstat.write('\n')
        fstat.close()
        
        print 'TIME spent on calling LMalgoRW.LMinv : ', time.clock() - temptime 
    
        ### Save inverted parameters:
        ##sp.save(subfolder+'/ParamResult'+str(i)+'.npy', res[0])
        # copy the runstats file:
        shutil.copy ('runstats.txt', subfolder+'/runstats.txt')
        ### copy h5 output files:
        ##shutil.copy ('natPEST.h5', subfolder+'/natPEST'+str(i)+'.h5')
        ##shutil.copy ('prodPEST.h5', subfolder+'/prodPEST'+str(i)+'.h5')
        ##shutil.copy ('simulobservNat.txt', subfolder+'/simulobservNat'+str(i)+'.txt')
        ##shutil.copy ('simulobservProd.txt', subfolder+'/simulobservProd'+str(i)+'.txt')


def invLMTSVD():
    nrInversions = 20 # Number of inversion to run
    subfolder = 'dummy' # Subfolder for saving some results
    newpath = os.getcwd() + '/' + subfolder
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    # -----------------------------------------------------------------
    #Choose TSVD method:
    # -----------------------------------------------------------------
    ## When using Lanczos:
    #stol = 1.e-5 # Lanczos convergence tolerance
    #nmax = 50 # Maximum number of excess Lanczos iterations
    #TSVDspecials = ['Lanc', stol, nmax]
    #SubReuse = False
    ## When using Lanczos with sv-cut:
    #stol = 1.e-5
    #SVcut = 0.5
    #SVcutmin = 1.0e-12
    #SVcutRedFac = 0.5
    #ktruncmax = 50
    #nmax = 50
    #TSVDspecials = ['SVcutLanc', stol, SVcut, SVcutmin, SVcutRedFac, ktruncmax, nmax]
    #SubReuse = False
    # When using Random EVEN-view (with optional power iteration qpow > 0) :
    # Number of views used are 2*(1+qpow), so use qpow=0 for 2-view method.
    ell = 10
    qpow = 0
    TSVDspecials = ['EVENview', ell, qpow]
    SubReuse = False # Whether to apply subspace re-use
    ## When using Random 1-view:
    #ell1 = 10
    #ell2 = 20
    #TSVDspecials = ['1view', ell1, ell2]
    #SubReuse = False # Whether to apply subspace re-use
    
    runinversions(nrInversions, subfolder, TSVDspecials, SubReuse)

if __name__ == "__main__":
    invLMTSVD()
    
    
    



