
"""
Module for interfacing with the C libraries.

@author: Chris Scott

"""
import os
from ctypes import CDLL, c_int, c_double, c_char, c_char_p, POINTER


################################################################################
# load the C libraries
################################################################################

# library path
LIBDIR = os.path.join(os.path.dirname(__file__), "visclibs")

################################################################################

# input module
input_c = CDLL(os.path.join(LIBDIR, "_input_c.so"))

# arg/ret types
input_c.readRef.argtypes = [c_char_p, POINTER(c_int), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), 
                            POINTER(c_double), c_int, POINTER(c_char), POINTER(c_int), POINTER(c_double), POINTER(c_double)]

input_c.readLBOMDXYZ.argtypes = [c_char_p, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), 
                                 POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int]

input_c.readLatticeLBOMD.argtypes = [c_char_p, POINTER(c_int), POINTER(c_double), POINTER(c_double), c_int, 
                                     POINTER(c_char), POINTER(c_int), POINTER(c_double), POINTER(c_double)]

input_c.writeLatticeLBOMD.argtypes = [c_char_p, c_int, POINTER(c_double), POINTER(c_char), 
                                      POINTER(c_int), POINTER(c_double), POINTER(c_double)]

################################################################################
# functions for casting pointer to numpy array to ctypes pointer
################################################################################

def CPtrToDouble(x):
    """
    Return pointer to numpy array cast to C double.
    
    """
    return x.ctypes.data_as(POINTER(c_double))

def CPtrToInt(x):
    """
    Return pointer to numpy array cast to C int.
    
    """
    return x.ctypes.data_as(POINTER(c_int))

def CPtrToChar(x):
    """
    Return pointer to numpy array cast to C char.
    
    """
    return x.ctypes.data_as(POINTER(c_char))
