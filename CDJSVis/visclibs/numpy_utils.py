
"""
Utils for wrapping numpy arrays.

"""
from ctypes import POINTER, c_double, c_int, c_char



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
