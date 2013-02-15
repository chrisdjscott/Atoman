
"""
Wrapper to vectors.c

@author: Chris Scott

"""
import os
import sys
import platform

from ctypes import CDLL, c_double, POINTER, c_int

from .numpy_utils import CPtrToDouble, CPtrToInt


################################################################################

# load lib (this is messy!!)
osname = platform.system()
if osname == "Darwin":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_vectors.dylib"))
        else:
            _lib = CDLL("_vectors.dylib")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_vectors.dylib"))

elif osname == "Linux":
    _lib = CDLL("_vectors.so")

################################################################################

# separation vector prototype
_lib.separationVector.restype = c_int
_lib.separationVector.argtypes = [c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_int)]

# separation vector
def separationVector(returnVector, pos1, pos2, cellDims, PBC):
    """
    Find separation vector between to vectors.
    
    """
    return _lib.separationVector(len(pos1), CPtrToDouble(returnVector), CPtrToDouble(pos1), CPtrToDouble(pos2), CPtrToDouble(cellDims), CPtrToInt(PBC))

################################################################################

# separation magnitude prototype
_lib.separationMagnitude.restype = c_double
_lib.separationMagnitude.argtypes = [c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_int)]

# separation magnitude
def separationMagnitude(pos1, pos2, cellDims, PBC):
    """
    Find separation magnitude between to vectors.
    
    """
    return _lib.separationMagnitude(len(pos1), CPtrToDouble(pos1), CPtrToDouble(pos2), CPtrToDouble(cellDims), CPtrToInt(PBC))

################################################################################

# magnitude prototype
_lib.magnitude.restype = c_double
_lib.magnitude.argtypes = [c_int, POINTER(c_double)]

# magnitude
def magnitude(vector):
    """
    Find magnitude of a vector.
    
    """
    return _lib.magnitude(len(vector), CPtrToDouble(vector))


