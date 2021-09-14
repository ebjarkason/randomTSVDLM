# About randomTSVDLM
Experimental Python code applying a randomized TSVD Levenberg-Marquardt (TSVD-LM) approach for inverting
an AUTOUGH2 geothermal reservoir model. The code implements the randomized TSVD-LM approaches discussed
in Bjarkason et al. (2017). 

The Python script src/main.py runs the inversion experiments discussed in Bjarkason et al. (2017). 
The TSVD-LM model updates can be found using the randomized methods outlined in Bjarkason et al. (2017) 
to form an approximate TSVD of a dimensionless sensitivity matrix. 

Note that the TSVD routines in src/evalTSVD.py can be run independently of the inversion and forward 
simulation code. For estimating the TSVD of matrix *A*, the randomized TSVD routines just need to be supplied
with functions that evaluate *A* times a matrix and *A*<sup>T</sup> times a matrix. Therefore, the TSVD routines can also be
used for estimating the TSVD of sensitivity or Jacobian matrices pertaining to other types of simulations 
(e.g., groundwater simulations), where the simulator has adjoint and direct capabilities. That is, for models 
where there are functions for evaluating the sensitivity matrix times a matrix (direct code) and the sensitivity 
matrix transposed times a matrix (adjoint code).

# Prerequisites
[PyTOUGH](https://github.com/acroucher/PyTOUGH): The inversion code uses PyTOUGH for reading simulation input and 
output files. PyTOUGH can be downloaded at https://github.com/acroucher/PyTOUGH.

AUTOUGH2: An experimental AUTOUGH2 executable is needed to generate the binary outputs used by the inversion code.
AUTOUGH2 is the University of Auckland’s version of the [TOUGH2](http://esd1.lbl.gov/research/projects/tough/) 
subsurface simulator. The authors can make the experimental AUTOUGH2 executable available to those that have an 
appropriate [TOUGH2](http://esd1.lbl.gov/research/projects/tough/) license.

# Citation
Bjarkason, E. K., Maclaren, O. J., O'Sullivan, J. P., and O'Sullivan, M. J. (2017). *Randomized truncated SVD Levenberg-Marquardt approach to geothermal natural state and history matching*. Water Resources Research, 54, 2376–2404. [https://doi.org/10.1002/2017WR021870](https://doi.org/10.1002/2017WR021870)

# Contact Authors
Elvar K. Bjarkason: ebja558@aucklanduni.ac.nz
