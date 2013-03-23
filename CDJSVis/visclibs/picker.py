
"""
Wrapper to picker.c

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
            _lib = CDLL(os.path.join(sys._MEIPASS, "_picker.dylib"))
        else:
            _lib = CDLL("_picker.dylib")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_picker.dylib"))

elif osname == "Linux":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_picker.so"))
        else:
            _lib = CDLL("_picker.so")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_picker.so"))

################################################################################

# picker prototype
_lib.pickObject.restype = c_int
_lib.pickObject.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_int), c_int, POINTER(c_int), c_int, POINTER(c_int), 
                            c_int, POINTER(c_int), c_int, POINTER(c_double), c_int, POINTER(c_double), 
                            c_int, POINTER(c_double), c_int, POINTER(c_int), c_int, POINTER(c_double), c_int, POINTER(c_double), 
                            c_int, POINTER(c_double), c_int, POINTER(c_int), c_int, POINTER(c_int), c_int, POINTER(c_double),
                            c_int, POINTER(c_double), c_int, POINTER(c_double)]

# picker
def pickObject(visibleAtoms, vacs, ints, onAnts, splits, pickPos, pos, refPos, PBC, cellDims, minPos, maxPos, specie, refSpecie,
               specieCovRad, refSpecieCovRad, result):
    """
    Picker.
    
    """
    return _lib.pickObject(len(visibleAtoms), CPtrToInt(visibleAtoms), len(vacs), CPtrToInt(vacs), len(ints), CPtrToInt(ints), len(onAnts), 
                           CPtrToInt(onAnts), len(splits), CPtrToInt(splits), len(pickPos), CPtrToDouble(pickPos), len(pos), CPtrToDouble(pos), 
                           len(refPos), CPtrToDouble(refPos), len(PBC), CPtrToInt(PBC), len(cellDims), CPtrToDouble(cellDims), len(minPos), 
                           CPtrToDouble(minPos), len(maxPos), CPtrToDouble(maxPos), len(specie), CPtrToInt(specie), len(refSpecie), 
                           CPtrToInt(refSpecie), len(specieCovRad), CPtrToDouble(specieCovRad), len(refSpecieCovRad), 
                           CPtrToDouble(refSpecieCovRad), len(result), CPtrToDouble(result))
