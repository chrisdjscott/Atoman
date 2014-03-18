
"""
Utilities common to multiple lattice generators

@author: Chris Scott

"""
import math

################################################################################

def fixChargesOnFixedBoundaries(lattice):
    """
    Fix charges on fixed boundaries:
    
    * Eighth charges on corners
    * Quarter charges on edges
    * Half charges on faces
    
    """
    #TODO: this should be written in C
    
    tol = 0.1
    
    totalCharge = 0.0
    for i in xrange(lattice.NAtoms):
        # position
        rx, ry, rz = lattice.atomPos(i)
        
        dlx = math.fabs(lattice.cellDims[0] - rx)
        dly = math.fabs(lattice.cellDims[1] - ry)
        dlz = math.fabs(lattice.cellDims[2] - rz)
        
        # eigth * corner charges
        if rx < tol and ry < tol and rz < tol:
            lattice.charge[i] /= 8.0
        
        elif dlx < tol and ry < tol and rz < tol:
            lattice.charge[i] /= 8.0
        
        elif rx < tol and dly < tol and rz < tol:
            lattice.charge[i] /= 8.0
        
        elif rx < tol and ry < tol and dlz < tol:
            lattice.charge[i] /= 8.0
        
        elif rx < tol and dly < tol and dlz < tol:
            lattice.charge[i] /= 8.0
        
        elif dlx < tol and ry < tol and dlz < tol:
            lattice.charge[i] /= 8.0
        
        elif dlx < tol and dly < tol and rz < tol:
            lattice.charge[i] /= 8.0
        
        elif dlx < tol and dly < tol and dlz < tol:
            lattice.charge[i] /= 8.0
        
        # quarter * edge charges
        elif rx < tol and ry < tol:
            lattice.charge[i] /= 4.0
        
        elif rx < tol and rz < tol:
            lattice.charge[i] /= 4.0
        
        elif ry < tol and rz < tol:
            lattice.charge[i] /= 4.0
        
        elif rx < tol and dly < tol:
            lattice.charge[i] /= 4.0
        
        elif rx < tol and dlz < tol:
            lattice.charge[i] /= 4.0
        
        elif ry < tol and dlx < tol:
            lattice.charge[i] /= 4.0
        
        elif ry < tol and dlz < tol:
            lattice.charge[i] /= 4.0
        
        elif rz < tol and dlx < tol:
            lattice.charge[i] /= 4.0
        
        elif rz < tol and dly < tol:
            lattice.charge[i] /= 4.0
        
        elif dlx < tol and dly < tol:
            lattice.charge[i] /= 4.0
        
        elif dlx < tol and dlz < tol:
            lattice.charge[i] /= 4.0
        
        elif dly < tol and dlz < tol:
            lattice.charge[i] /= 4.0
        
        # half * face charges
        elif rx < tol:
            lattice.charge[i] /= 2.0
        
        elif ry < tol:
            lattice.charge[i] /= 2.0
        
        elif rz < tol:
            lattice.charge[i] /= 2.0
        
        elif dlx < tol:
            lattice.charge[i] /= 2.0
        
        elif dly < tol:
            lattice.charge[i] /= 2.0
        
        elif dlz < tol:
            lattice.charge[i] /= 2.0
        
        totalCharge += lattice.charge[i]
    
    return totalCharge
    
    
    





