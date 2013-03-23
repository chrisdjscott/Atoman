
"""
Vector operations.

"""
import numpy as np
from ..visclibs import vectors as c_vectors


################################################################################
# magnitude of vector
################################################################################
def magnitude(vect):
    """
    Return magnitude of vect
    
    """
    return c_vectors.magnitude(vect)


################################################################################
# return normalised vector
################################################################################
def normalise(vect):
    """
    Return vect normalised
    
    """
    mag = magnitude(vect)
    if mag == 0:
        print "WARNING: attempting to normalise zero vector"
        return vect
    return vect / mag

################################################################################
# return displacement vector between 2 position vectors
################################################################################
def separationVector(pos1, pos2, cellDimsTmp, PBC):
    """
    Return separation vector between position vectors pos1 and pos2.
    i.e. pos2 - pos1
    
    """
    sepVec = np.empty(len(pos1), np.float64)
    
    cellDims = np.zeros(9, np.float64)
    cellDims[0] = cellDimsTmp[0]
    cellDims[4] = cellDimsTmp[1]
    cellDims[8] = cellDimsTmp[2]
    
    c_vectors.separationVector(sepVec, pos1, pos2, cellDims, PBC)
        
    return sepVec
    

################################################################################
# return magnitude of separation between 2 position vectors
################################################################################
def separation(pos1, pos2, cellDimsTmp, PBC):
    """
    Return magnitude of the separation vector between pos1 and pos2.
    i.e. ||pos2 - pos1||
    
    """
    cellDims = np.zeros(9, np.float64)
    cellDims[0] = cellDimsTmp[0]
    cellDims[4] = cellDimsTmp[1]
    cellDims[8] = cellDimsTmp[2]
    
    return c_vectors.separationMagnitude(pos1, pos2, cellDims, PBC)




