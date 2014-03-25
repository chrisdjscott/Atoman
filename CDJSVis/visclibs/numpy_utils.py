# -*- coding: utf-8 -*-

"""
Utils for wrapping numpy arrays.

@author: Chris Scott

"""
import numpy as np
import ctypes as C


################################################################################

class Allocator(object):
    """
    Allocate numpy arrays from C library.
    
    """
    CFUNCTYPE = C.CFUNCTYPE(C.c_long, C.c_char_p, C.c_int, C.POINTER(C.c_int), C.c_char)
    
    def __init__(self, storeAsList=False):
        if storeAsList:
            self.allocated_arrays = []
        else:
            self.allocated_arrays = {}
        
        self.storeAsList = storeAsList
    
    def __call__(self, name, dims, shape, dtype):
        """
        Allocate array.
        
        """
        x = np.empty(shape[:dims], np.dtype(dtype))
        
        if self.storeAsList:
            self.allocated_arrays.append(x)
        else:
            self.allocated_arrays[name] = x
        
        return x.ctypes.data_as(C.c_void_p).value
    
    def getcfunc(self):
        return self.CFUNCTYPE(self)
    
    cfunc = property(getcfunc)

################################################################################

def CPtrToDouble(x):
    """
    Return pointer to numpy array cast to C double.
    
    """
    return x.ctypes.data_as(C.POINTER(C.c_double))

################################################################################

def CPtrToInt(x):
    """
    Return pointer to numpy array cast to C int.
    
    """
    return x.ctypes.data_as(C.POINTER(C.c_int))

################################################################################

def CPtrToChar(x):
    """
    Return pointer to numpy array cast to C char.
    
    """
    return x.ctypes.data_as(C.POINTER(C.c_char))
