
"""
Wrapper to structure_parameter.c

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
            _lib = CDLL(os.path.join(sys._MEIPASS, "_bond_order.dylib"))
        else:
            _lib = CDLL("_bond_order.dylib")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_bond_order.dylib"))

elif osname == "Linux":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_bond_order.so"))
        else:
            _lib = CDLL("_bond_order.so")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_bond_order.so"))

################################################################################

# bondOrderFilter filter prototype
_lib.bondOrderFilter.restype = c_int
_lib.bondOrderFilter.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_double), c_double, c_double, c_double, c_int, POINTER(c_double), POINTER(c_double), 
                                 POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_int), c_int, POINTER(c_double), c_int]

# bondOrderFilter filter
def bondOrderFilter(visibleAtoms, pos, minVal, maxVal, maxBondDistance, scalarsQ4, scalarsQ6, minPos, maxPos, cellDims, PBC, NScalars, fullScalars, 
             filteringEnabled):
    """
    bondOrderFilter filter.
    
    """
    return _lib.bondOrderFilter(len(visibleAtoms), CPtrToInt(visibleAtoms), len(pos), CPtrToDouble(pos), minVal, maxVal, maxBondDistance, 
                                len(scalarsQ4), CPtrToDouble(scalarsQ4), CPtrToDouble(scalarsQ6), CPtrToDouble(minPos), CPtrToDouble(maxPos), CPtrToDouble(cellDims), 
                                CPtrToInt(PBC), NScalars, CPtrToDouble(fullScalars), int(filteringEnabled))
