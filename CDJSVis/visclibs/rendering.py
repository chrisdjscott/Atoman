# -*- coding: utf-8 -*-

"""
Wrapper to rendering.c

Copyright 2014 Chris Scott

"""
import os
import sys
import platform

from ctypes import CDLL, c_double, POINTER, c_int, c_char_p, c_char

from .numpy_utils import CPtrToDouble, CPtrToInt, CPtrToChar
from .numpy_utils import Allocator as alloc


################################################################################

# load lib (this is messy!!)
osname = platform.system()
if osname == "Darwin":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_rendering.dylib"))
        else:
            _lib = CDLL("_rendering.dylib")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_rendering.dylib"))

elif osname == "Linux":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_rendering.so"))
        else:
            _lib = CDLL("_rendering.so")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_rendering.so"))

################################################################################

# splitVisAtomsBySpecie prototype
_lib.splitVisAtomsBySpecie.restype = c_int
_lib.splitVisAtomsBySpecie.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_int), POINTER(c_int), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, alloc.CFUNCTYPE]

# splitVisAtomsBySpecie
def splitVisAtomsBySpecie(visibleAtoms, NSpecies, specie, specieCount, pos, PE, KE, charge, scalars, scalarType, heightAxis, allocator):
    """
    splitVisAtomsBySpecie
    
    """
    return _lib.splitVisAtomsBySpecie(len(visibleAtoms), CPtrToInt(visibleAtoms), NSpecies, CPtrToInt(specie), CPtrToInt(specieCount), CPtrToDouble(pos), CPtrToDouble(PE), CPtrToDouble(KE), CPtrToDouble(charge), CPtrToDouble(scalars), scalarType, heightAxis, allocator)
