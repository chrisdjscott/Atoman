
"""
Wrapper to bonds.c

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
            _bonds = CDLL(os.path.join(sys._MEIPASS, "_bonds.dylib"))
        else:
            _bonds = CDLL("_bonds.dylib")
    except OSError:
        _bonds = CDLL(os.path.join(os.path.dirname(__file__), "_bonds.dylib"))

elif osname == "Linux":
    try:
        if hasattr(sys, "_MEIPASS"):
            _bonds = CDLL(os.path.join(sys._MEIPASS, "_bonds.so"))
        else:
            _bonds = CDLL("_bonds.so")
    except OSError:
        _bonds = CDLL(os.path.join(os.path.dirname(__file__), "_bonds.so"))


# calculate bonds prototype
_bonds.calculateBonds.restype = C.c_int
_bonds.calculateBonds.argtypes = [C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_double), C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_double), 
                                  C.POINTER(C.c_double), C.c_double, C.c_int, C.POINTER(C.c_double), C.POINTER(C.c_int), C.POINTER(C.c_double), 
                                  C.POINTER(C.c_double), C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_double), C.POINTER(C.c_int)]

# calculate bonds function
def calculateBonds(NVisible, visibleAtoms, pos, specie, NSpecies, bondMinArray, bondMaxArray, approxBoxWidth, maxBondsPerAtom, cellDims, 
                   PBC, minPos, maxPos, bondArray, NBondsArray, bondVectorArray, bondSpecieCounter):
    """
    Calculate bonds between visible atoms.
    
    """
    return _bonds.calculateBonds(NVisible, CPtrToInt(visibleAtoms), CPtrToDouble(pos), CPtrToInt(specie), NSpecies, 
                                 CPtrToDouble(bondMinArray), CPtrToDouble(bondMaxArray), approxBoxWidth, maxBondsPerAtom,
                                 CPtrToDouble(cellDims), CPtrToInt(PBC), CPtrToDouble(minPos), CPtrToDouble(maxPos),
                                 CPtrToInt(bondArray), CPtrToInt(NBondsArray), CPtrToDouble(bondVectorArray), CPtrToInt(bondSpecieCounter))
