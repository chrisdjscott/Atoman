
"""
Wrapper to acna.c

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
            _lib = CDLL(os.path.join(sys._MEIPASS, "_acna.dylib"))
        else:
            _lib = CDLL("_acna.dylib")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_acna.dylib"))

elif osname == "Linux":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_acna.so"))
        else:
            _lib = CDLL("_acna.so")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_acna.so"))

################################################################################

# adaptiveCommonNeighbourAnalysis filter prototype
_lib.adaptiveCommonNeighbourAnalysis.restype = c_int
_lib.adaptiveCommonNeighbourAnalysis.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_double), c_int, POINTER(c_double), 
                                                 POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_int), c_int, POINTER(c_double),  c_double]

# adaptiveCommonNeighbourAnalysis filter
def adaptiveCommonNeighbourAnalysis(visibleAtoms, pos, scalars, minPos, maxPos, cellDims, PBC, NScalars, fullScalars, maxBondDistance):
    """
    adaptiveCommonNeighbourAnalysis filter.
    
    """
    return _lib.adaptiveCommonNeighbourAnalysis(len(visibleAtoms), CPtrToInt(visibleAtoms), len(pos), CPtrToDouble(pos), 
                                                len(scalars), CPtrToDouble(scalars), CPtrToDouble(minPos), CPtrToDouble(maxPos), CPtrToDouble(cellDims), 
                                                CPtrToInt(PBC), NScalars, CPtrToDouble(fullScalars), maxBondDistance)
