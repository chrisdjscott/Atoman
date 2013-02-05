
"""
Utils for wrapping to ctypes.

@author: Chris Scott

"""
import platform

_osname = platform.system()

def libSuffix():
    """
    Returns library suffix for current os.
    
    """
    if _osname == "Darwin":
        return "dylib"
    elif _osname == "Linux":
        return "so"
    
    return None
