
"""
Wrapper to rdf.c

@author: Chris Scott

"""
import os
import sys
import platform

from ctypes import CDLL
import ctypes as C

from .numpy_utils import CPtrToDouble, CPtrToInt



# load lib (this is messy!!)
osname = platform.system()
if osname == "Darwin":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_rdf.dylib"))
        else:
            _lib = CDLL("_rdf.dylib")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_rdf.dylib"))

elif osname == "Linux":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_rdf.so"))
        else:
            _lib = CDLL("_rdf.so")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_rdf.so"))


# calculate bonds prototype
_lib.calculateRDF.restype = C.c_int
_lib.calculateRDF.argtypes = [C.c_int, C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_double), C.c_int, C.c_int, 
                              C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_int), C.c_double, 
                              C.c_double, C.c_int, C.POINTER(C.c_double)]

# calculate bonds function
def calculateRDF(visibleAtoms, specie, pos, spec1Index, spec2Index, minPos, maxPos, cellDims, PBC, binMin, binMax, NBins, rdfArray):
    """
    Calculate bonds between visible atoms.
    
    """
    return _lib.calculateRDF(len(visibleAtoms), CPtrToInt(visibleAtoms), len(specie), CPtrToInt(specie), CPtrToDouble(pos), spec1Index, 
                             spec2Index, CPtrToDouble(minPos), CPtrToDouble(maxPos), CPtrToDouble(cellDims), CPtrToInt(PBC), binMin, 
                             binMax, NBins, CPtrToDouble(rdfArray))
