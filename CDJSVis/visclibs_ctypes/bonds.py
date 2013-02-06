
"""
Wrapper to bonds.c

@author: Chris Scott

"""
import platform

from ctypes import CDLL
import ctypes as C

from .numpy_utils import CPtrToDouble, CPtrToInt



# load lib
osname = platform.system()
if osname == "Darwin":
    _bonds = CDLL("_bonds.dylib")

elif osname == "Linux":
    _bonds = CDLL("_bonds.so")


# calculate bonds prototype
_bonds.calculateBonds.restype = C.c_int
_bonds.calculateBonds.argtypes = [C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_double), C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_double), 
                                  C.POINTER(C.c_double), C.c_double, C.c_int, C.POINTER(C.c_double), C.POINTER(C.c_int), C.POINTER(C.c_double), 
                                  C.POINTER(C.c_double), C.POINTER(C.c_int), C.POINTER(C.c_int)]

# calculate bonds function
def calculateBonds(NVisible, visibleAtoms, pos, specie, NSpecies, bondMinArray, bondMaxArray, approxBoxWidth, maxBondsPerAtom, cellDims, 
                   PBC, minPos, maxPos, bondArray, NBondsArray):
    """
    Calculate bonds between visible atoms.
    
    """
    return _bonds.calculateBonds(NVisible, CPtrToInt(visibleAtoms), CPtrToDouble(pos), CPtrToInt(specie), NSpecies, 
                                 CPtrToDouble(bondMinArray), CPtrToDouble(bondMaxArray), approxBoxWidth, maxBondsPerAtom,
                                 CPtrToDouble(cellDims), CPtrToInt(PBC), CPtrToDouble(minPos), CPtrToDouble(maxPos),
                                 CPtrToInt(bondArray), CPtrToInt(NBondsArray))
