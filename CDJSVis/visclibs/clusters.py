
"""
Wrapper to clusters.c

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
            _lib = CDLL(os.path.join(sys._MEIPASS, "_clusters.dylib"))
        else:
            _lib = CDLL("_clusters.dylib")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_clusters.dylib"))

elif osname == "Linux":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_clusters.so"))
        else:
            _lib = CDLL("_clusters.so")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_clusters.so"))


################################################################################

# find clusters prototype
_lib.findClusters.restype = c_int
_lib.findClusters.argtypes = [c_int, POINTER(c_int), POINTER(c_double), POINTER(c_int), c_double, POINTER(c_double), POINTER(c_int), 
                              POINTER(c_double), POINTER(c_double), c_int, c_int, POINTER(c_int)]

# find clusters
def findClusters(visibleAtoms, pos, clusterArray, neighbourRadius, cellDims, PBC, minPos, maxPos, minClusterSize, maxClusterSize, results):
    """
    Find clusters of atoms.
    
    """
    return _lib.findClusters(len(visibleAtoms), CPtrToInt(visibleAtoms), CPtrToDouble(pos), CPtrToInt(clusterArray), neighbourRadius,
                             CPtrToDouble(cellDims), CPtrToInt(PBC), CPtrToDouble(minPos), CPtrToDouble(maxPos), minClusterSize, 
                             maxClusterSize, CPtrToInt(results))

################################################################################

# prep draw hulls prototype
_lib.prepareClusterToDrawHulls.restype = c_int
_lib.prepareClusterToDrawHulls.argtypes = [c_int, POINTER(c_double), POINTER(c_double), POINTER(c_int), POINTER(c_int), c_double]

# fprep draw hulls
def prepareClusterToDrawHulls(N, pos, cellDims, PBC, appliedPBCs, neighbourRadius):
    """
    Prepare clusters to draw hulls.
    
    """
    return _lib.prepareClusterToDrawHulls(N, CPtrToDouble(pos), CPtrToDouble(cellDims), CPtrToInt(PBC), CPtrToInt(appliedPBCs), neighbourRadius)

