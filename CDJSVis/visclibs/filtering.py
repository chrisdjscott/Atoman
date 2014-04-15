
"""
Wrapper to filtering.c

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
            _lib = CDLL(os.path.join(sys._MEIPASS, "_filtering.dylib"))
        else:
            _lib = CDLL("_filtering.dylib")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_filtering.dylib"))

elif osname == "Linux":
    try:
        if hasattr(sys, "_MEIPASS"):
            _lib = CDLL(os.path.join(sys._MEIPASS, "_filtering.so"))
        else:
            _lib = CDLL("_filtering.so")
    except OSError:
        _lib = CDLL(os.path.join(os.path.dirname(__file__), "_filtering.so"))


################################################################################

# specie filter prototype
_lib.specieFilter.restype = c_int
_lib.specieFilter.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_int), c_int, POINTER(c_int), c_int, POINTER(c_double)]

# specie filter
def specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars):
    """
    Specie filter.
    
    """
    return _lib.specieFilter(len(visibleAtoms), CPtrToInt(visibleAtoms), len(visibleSpecieArray), CPtrToInt(visibleSpecieArray),
                             len(specieArray), CPtrToInt(specieArray), NScalars, CPtrToDouble(fullScalars))

################################################################################

# slice filter prototype
_lib.sliceFilter.restype = c_int
_lib.sliceFilter.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_double), c_double, c_double, c_double, 
                             c_double, c_double, c_double, c_int, c_int, POINTER(c_double)]

# slice filter
def sliceFilter(visibleAtoms, pos, x0, y0, z0, xn, yn, zn, invert, NScalars, fullScalars):
    """
    Slice filter.
    
    """
    return _lib.sliceFilter(len(visibleAtoms), CPtrToInt(visibleAtoms), len(pos), CPtrToDouble(pos), x0, y0, z0, 
                            xn, yn, zn, invert, NScalars, CPtrToDouble(fullScalars))

################################################################################

# crop sphere filter prototype
_lib.cropSphereFilter.restype = c_int
_lib.cropSphereFilter.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_double), c_double, c_double, c_double, c_double, 
                                  POINTER(c_double), POINTER(c_int), c_int, c_int, POINTER(c_double)]

# crop sphere filter
def cropSphereFilter(visibleAtoms, pos, xCentre, yCentre, zCentre, radius, cellDims, PBC, invertSelection, NScalars, fullScalars):
    """
    Crop sphere filter.
    
    """
    return _lib.cropSphereFilter(len(visibleAtoms), CPtrToInt(visibleAtoms), len(pos), CPtrToDouble(pos), xCentre, yCentre, zCentre, 
                                 radius, CPtrToDouble(cellDims), CPtrToInt(PBC), invertSelection, NScalars, CPtrToDouble(fullScalars))

################################################################################

# crop filter prototype
_lib.cropFilter.restype = c_int
_lib.cropFilter.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_double), c_double, c_double, c_double, c_double, 
                            c_double, c_double, c_int, c_int, c_int, c_int, c_int, POINTER(c_double)]

# crop filter
def cropFilter(visibleAtoms, pos, xmin, xmax, ymin, ymax, zmin, zmax, xEnabled, yEnabled, zEnabled, invertSelection, NScalars, fullScalars):
    """
    Crop filter.
    
    """
    return _lib.cropFilter(len(visibleAtoms), CPtrToInt(visibleAtoms), len(pos), CPtrToDouble(pos), xmin, xmax, ymin, ymax, 
                           zmin, zmax, xEnabled, yEnabled, zEnabled, invertSelection, NScalars, CPtrToDouble(fullScalars))

################################################################################

# displacement filter prototype
_lib.displacementFilter.restype = c_int
_lib.displacementFilter.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_double), c_int, POINTER(c_double), c_int, POINTER(c_double), 
                                    POINTER(c_double), POINTER(c_int), c_double, c_double, c_int, POINTER(c_double), c_int, c_int, 
                                    POINTER(c_double)]

# displacement filter
def displacementFilter(visibleAtoms, scalars, pos, refPos, cellDims, PBC, minDisp, maxDisp, NScalars, fullScalars, filteringEnabled, 
                       driftCompensation, driftVector):
    """
    Displacement filter.
    
    """
    return _lib.displacementFilter(len(visibleAtoms), CPtrToInt(visibleAtoms), len(scalars), CPtrToDouble(scalars), len(pos), CPtrToDouble(pos), 
                                   len(refPos), CPtrToDouble(refPos), CPtrToDouble(cellDims), CPtrToInt(PBC), minDisp, maxDisp, NScalars, 
                                   CPtrToDouble(fullScalars), int(filteringEnabled), int(driftCompensation), CPtrToDouble(driftVector))

################################################################################

# kinetic energy filter prototype
_lib.KEFilter.restype = c_int
_lib.KEFilter.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_double), c_double, c_double, c_int, POINTER(c_double)]

# kinetic energy filter
def KEFilter(visibleAtoms, KE, minKE, maxKE, NScalars, fullScalars):
    """
    Kinetic energy filter.
    
    """
    return _lib.KEFilter(len(visibleAtoms), CPtrToInt(visibleAtoms), len(KE), CPtrToDouble(KE), minKE, maxKE, NScalars, CPtrToDouble(fullScalars))

################################################################################

# potential energy filter prototype
_lib.PEFilter.restype = c_int
_lib.PEFilter.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_double), c_double, c_double, c_int, POINTER(c_double)]

# potential energy filter
def PEFilter(visibleAtoms, PE, minPE, maxPE, NScalars, fullScalars):
    """
    Potential energy filter.
    
    """
    return _lib.PEFilter(len(visibleAtoms), CPtrToInt(visibleAtoms), len(PE), CPtrToDouble(PE), minPE, maxPE, NScalars, CPtrToDouble(fullScalars))

################################################################################

# charge filter prototype
_lib.chargeFilter.restype = c_int
_lib.chargeFilter.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_double), c_double, c_double, c_int, POINTER(c_double)]

# charge filter
def chargeFilter(visibleAtoms, charge, minCharge, maxCharge, NScalars, fullScalars):
    """
    Charge filter.
    
    """
    return _lib.chargeFilter(len(visibleAtoms), CPtrToInt(visibleAtoms), len(charge), CPtrToDouble(charge), 
                             minCharge, maxCharge, NScalars, CPtrToDouble(fullScalars))

################################################################################

# coordination number filter prototype
_lib.coordNumFilter.restype = c_int
_lib.coordNumFilter.argtypes = [c_int, POINTER(c_int), POINTER(c_double), POINTER(c_int), c_int, POINTER(c_double), 
                                POINTER(c_double), c_double, POINTER(c_double), POINTER(c_int), POINTER(c_double), POINTER(c_double), 
                                POINTER(c_double), c_int, c_int, c_int, POINTER(c_double), c_int]

# coordination number filter
def coordNumFilter(visibleAtoms, pos, specie, NSpecies, bondMinArray, bondMaxArray, approxBoxWidth, cellDims, PBC, 
                   minPos, maxPos, coordArray, minCoordNum, maxCoordNum, NScalars, fullScalars, filteringEnabled):
    """
    Coordination number filter.
    
    """
    return _lib.coordNumFilter(len(visibleAtoms), CPtrToInt(visibleAtoms), CPtrToDouble(pos), CPtrToInt(specie), NSpecies, CPtrToDouble(bondMinArray), 
                               CPtrToDouble(bondMaxArray), approxBoxWidth, CPtrToDouble(cellDims), CPtrToInt(PBC), CPtrToDouble(minPos), 
                               CPtrToDouble(maxPos), CPtrToDouble(coordArray), minCoordNum, maxCoordNum, NScalars, CPtrToDouble(fullScalars), 
                               int(filteringEnabled))

################################################################################

# voronoi volume filter prototype
_lib.voronoiVolumeFilter.restype = c_int
_lib.voronoiVolumeFilter.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_double), c_double, c_double, c_int, 
                                     POINTER(c_double), c_int, POINTER(c_double), c_int]

# voronoi volume filter
def voronoiVolumeFilter(visibleAtoms, volume, minVolume, maxVolume, scalars, NScalars, fullScalars, filteringEnabled):
    """
    Voronoi volume filter.
    
    """
    return _lib.voronoiVolumeFilter(len(visibleAtoms), CPtrToInt(visibleAtoms), len(volume), CPtrToDouble(volume), 
                                    minVolume, maxVolume, len(scalars), CPtrToDouble(scalars), NScalars, 
                                    CPtrToDouble(fullScalars), int(filteringEnabled))

################################################################################

# voronoi neighbours filter prototype
_lib.voronoiNeighboursFilter.restype = c_int
_lib.voronoiNeighboursFilter.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_int), c_int, c_int, c_int, 
                                         POINTER(c_double), c_int, POINTER(c_double), c_int]

# voronoi neighbours filter
def voronoiNeighboursFilter(visibleAtoms, numNebsArray, minNebs, maxNebs, scalars, NScalars, fullScalars, filteringEnabled):
    """
    Voronoi neighbours filter.
    
    """
    return _lib.voronoiNeighboursFilter(len(visibleAtoms), CPtrToInt(visibleAtoms), len(numNebsArray), CPtrToInt(numNebsArray), 
                                        minNebs, maxNebs, len(scalars), CPtrToDouble(scalars), NScalars, CPtrToDouble(fullScalars), 
                                        int(filteringEnabled))

################################################################################

# Q4 filter prototype
_lib.Q4Filter.restype = c_int
_lib.Q4Filter.argtypes = [c_int, POINTER(c_int), c_int, POINTER(c_double), c_double, c_double, c_double, c_int, POINTER(c_double), 
                          POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_int), c_int, POINTER(c_double), c_int]

# Q4 filter
def Q4Filter(visibleAtoms, pos, minQ4, maxQ4, maxBondDistance, scalars, minPos, maxPos, cellDims, PBC, NScalars, fullScalars, 
             filteringEnabled):
    """
    Q4 filter.
    
    """
    return _lib.Q4Filter(len(visibleAtoms), CPtrToInt(visibleAtoms), len(pos), CPtrToDouble(pos), minQ4, maxQ4, maxBondDistance, 
                         len(scalars), CPtrToDouble(scalars), CPtrToDouble(minPos), CPtrToDouble(maxPos), CPtrToDouble(cellDims), 
                         CPtrToInt(PBC), NScalars, CPtrToDouble(fullScalars), int(filteringEnabled))

################################################################################

# calculate_drift_vector prototype
_lib.calculate_drift_vector.restype = c_int
_lib.calculate_drift_vector.argtypes = [c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_int), POINTER(c_double)]

# calculate_drift_vector
def calculate_drift_vector(NAtoms, pos, refPos, cellDims, PBC, driftVector):
    """
    calculate_drift_vector
    
    """
    return _lib.calculate_drift_vector(NAtoms, CPtrToDouble(pos), CPtrToDouble(refPos), CPtrToDouble(cellDims), CPtrToInt(PBC), CPtrToDouble(driftVector))
