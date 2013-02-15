
"""
Wrapper to input.c

@author: Chris Scott

"""
import os
import sys
import platform

from ctypes import CDLL, c_double, POINTER, c_int, c_char_p, c_char

from .numpy_utils import CPtrToDouble, CPtrToInt, CPtrToChar


################################################################################

# load lib (this is messy!!)
osname = platform.system()
if osname == "Darwin":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_input.dylib"))
        else:
            _lib = CDLL("_input.dylib")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_input.dylib"))

elif osname == "Linux":
    _lib = CDLL("_input.so")

################################################################################

# read ref prototype
_lib.readRef.restype = c_int
_lib.readRef.argtypes = [c_char_p, POINTER(c_int), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), 
                         POINTER(c_double), POINTER(c_char), POINTER(c_int), POINTER(c_double), POINTER(c_double)]

# read ref
def readRef(filename, specie, pos, charge, KE, PE, force, specieList, specieCount, maxPos, minPos):
    """
    Read LBOMD ref file.
    
    """
    return _lib.readRef(filename, CPtrToInt(specie), CPtrToDouble(pos), CPtrToDouble(charge), CPtrToDouble(KE), 
                        CPtrToDouble(PE), CPtrToDouble(force), CPtrToChar(specieList), CPtrToInt(specieCount), 
                        CPtrToDouble(maxPos), CPtrToDouble(minPos))

################################################################################

# read xyz prototype
_lib.readLBOMDXYZ.restype = c_int
_lib.readLBOMDXYZ.argtypes = [c_char_p, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), 
                              POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int]

# read xyz
def readLBOMDXYZ(filename, pos, charge, KE, PE, force, maxPos, minPos, xyzFormat):
    """
    Read LBOMD xyz file.
    
    """
    return _lib.readLBOMDXYZ(filename, CPtrToDouble(pos), CPtrToDouble(charge), CPtrToDouble(KE), CPtrToDouble(PE), 
                             CPtrToDouble(force), CPtrToDouble(maxPos), CPtrToDouble(minPos), xyzFormat)

################################################################################

# read lattice prototype
_lib.readLatticeLBOMD.restype = c_int
_lib.readLatticeLBOMD.argtypes = [c_char_p, POINTER(c_int), POINTER(c_double), POINTER(c_double), POINTER(c_char), 
                                  POINTER(c_int), POINTER(c_double), POINTER(c_double)]

# read lattice
def readLatticeLBOMD(filename, specie, pos, charge, specieList, specieCount, maxPos, minPos):
    """
    Read lattice file.
    
    """
    return _lib.readLatticeLBOMD(filename, CPtrToInt(specie), CPtrToDouble(pos), CPtrToDouble(charge), CPtrToChar(specieList), 
                                 CPtrToInt(specieCount), CPtrToDouble(maxPos), CPtrToDouble(minPos))

