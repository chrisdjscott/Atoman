
"""
Wrapper to defects.c

@author: Chris Scott

"""
import os
import sys
import platform

from ctypes import CDLL, c_double, POINTER, c_int, c_char

from .numpy_utils import CPtrToDouble, CPtrToInt, CPtrToChar


################################################################################

# load lib (this is messy!!)
osname = platform.system()
if osname == "Darwin":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_defects.dylib"))
        else:
            _lib = CDLL("_defects.dylib")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_defects.dylib"))

elif osname == "Linux":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_defects.so"))
        else:
            _lib = CDLL("_defects.so")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_defects.so"))


################################################################################

# find defects prototype
_lib.findDefects.restype = c_int
_lib.findDefects.argtypes = [c_int, c_int, c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), c_int, 
                             POINTER(c_int), c_int, POINTER(c_int), c_int, POINTER(c_char), POINTER(c_int), POINTER(c_double), c_int, 
                             POINTER(c_char), POINTER(c_int), POINTER(c_double), POINTER(c_double), POINTER(c_int), c_double, 
                             POINTER(c_double), POINTER(c_double), c_int, c_double, POINTER(c_int), c_int, POINTER(c_int), POINTER(c_int), 
                             POINTER(c_int), POINTER(c_int), POINTER(c_int), c_int, c_int, POINTER(c_int), c_int]

# find defects
def findDefects(includeVacs, includeInts, includeAnts, NDefectsType, vacancies, interstitials, antisites, onAntisites, excludeSpecsInput, 
                excludeSpecsRef, NAtoms, specieList, specie, pos, refNAtoms, refSpecieList, refSpecie, refPos, cellDims, PBC, vacancyRadius, 
                minPos, maxPos, findClustersFlag, clusterRadius, defectCluster, vacSpecieCount, intSpecieCount, antSpecieCount, 
                onAntSpecieCount, splitIntSpecieCount, minClusterSize, maxClusterSize, splitInterstitials, identifySplits):
    """
    Find defects.
    
    """
    return _lib.findDefects(includeVacs, includeInts, includeAnts, CPtrToInt(NDefectsType), CPtrToInt(vacancies), CPtrToInt(interstitials), 
                            CPtrToInt(antisites), CPtrToInt(onAntisites), len(excludeSpecsInput), CPtrToInt(excludeSpecsInput), len(excludeSpecsRef), 
                            CPtrToInt(excludeSpecsRef), NAtoms, CPtrToChar(specieList), CPtrToInt(specie), CPtrToDouble(pos), refNAtoms, 
                            CPtrToChar(refSpecieList), CPtrToInt(refSpecie), CPtrToDouble(refPos), CPtrToDouble(cellDims), CPtrToInt(PBC), vacancyRadius, 
                            CPtrToDouble(minPos), CPtrToDouble(maxPos), findClustersFlag, clusterRadius, CPtrToInt(defectCluster), len(vacSpecieCount), 
                            CPtrToInt(vacSpecieCount), CPtrToInt(intSpecieCount), CPtrToInt(antSpecieCount), CPtrToInt(onAntSpecieCount), 
                            CPtrToInt(splitIntSpecieCount), minClusterSize, maxClusterSize, CPtrToInt(splitInterstitials), identifySplits)

