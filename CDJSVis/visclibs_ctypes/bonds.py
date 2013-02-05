
"""
Wrapper to bonds.c

@author: Chris Scott

"""
#import platform

from ctypes import CDLL
import ctypes as C

from . import numpy_utils as nu
from .utils import libSuffix


#libdir = os.path.dirname(__file__)
#print "LIBDIR", libdir
#libpath = os.path.join(libdir, "_bonds.so")

#osname = platform.system()
#if osname == "Darwin":
#    libname = "_bonds.dylib"
#elif osname == "Linux":
#    libname = "_bonds.so"

# load lib
_bonds = CDLL("_bonds.dylib")

# calculate bonds prototype
_bonds.calculateBonds.restype = C.c_int
_bonds.calculateBonds.argtypes = [C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_double), C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_double), 
                                  C.POINTER(C.c_double), C.c_double, C.c_int, C.POINTER(C.c_double), C.POINTER(C.c_int), C.POINTER(C.c_double), 
                                  C.POINTER(C.c_double), C.POINTER(C.c_int), C.POINTER(C.c_int)]

def calculateBonds(NVisible, visibleAtoms, pos, specie, NSpecies, bondMinArray, bondMaxArray, approxBoxWidth, maxBondsPerAtom, cellDims, 
                   PBC, minPos, maxPos, bondArray, NBondsArray):
    """
    Calculate bonds between visible atoms.
    
    """
    return _bonds.calculateBonds(NVisible, nu.CPtrToInt(visibleAtoms), nu.CPtrToDouble(pos), nu.CPtrToInt(specie), NSpecies, 
                                 nu.CPtrToDouble(bondMinArray), nu.CPtrToDouble(bondMaxArray), approxBoxWidth, maxBondsPerAtom,
                                 nu.CPtrToDouble(cellDims), nu.CPtrToInt(PBC), nu.CPtrToDouble(minPos), nu.CPtrToDouble(maxPos),
                                 nu.CPtrToInt(bondArray), nu.CPtrToInt(NBondsArray))
