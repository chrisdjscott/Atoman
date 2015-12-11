
"""
Module for computing Voronoi tesselation

@author: Chris Scott

"""
import time
import logging

import numpy as np

from . import _voronoi


class VoronoiCalculator(object):
    """Base object for Voronoi calculators."""
    def __init__(self):
        self._voronoi = None
    
    def isCalculated(self):
        """Have we already calculated the Voronoi tessellation."""
        return True if isinstance(self._voronoi, _voronoi.Voronoi) else False
    
    def getVoronoi(self, *args):
        """Return the Voronoi object."""
        if not self.isCalculated():
            self._calculate(*args)
        
        return self._voronoi
    
    def calculate(*args):
        """Calculate Voronoi: to be overridden."""
        raise NotImplementedError("VoronoiCalculator.calculate has not been implemented")


class VoronoiAtomsCalculator(VoronoiCalculator):
    """
    Object for calculating Voronoi cells for atoms.
    
    """
    def __init__(self):
        super(VoronoiAtomsCalculator, self).__init__()
        self._logger = logging.getLogger(__name__ + ".VoronoiAtomsCalculator")
    
    def _calculate(self, lattice, voronoiOptions):
        """Calculate Voronoi."""
        self._logger.info("Calculating Voronoi (Atoms)")
        self._voronoi = computeVoronoi(lattice, voronoiOptions)


class VoronoiDefectsCalculator(VoronoiCalculator):
    """
    Object for calculating Voronoi cells for defects.
    
    """
    def __init__(self):
        super(VoronoiDefectsCalculator, self).__init__()
        self._logger = logging.getLogger(__name__ + ".VoronoiDefectsCalculator")
    
    def _calculate(self, lattice, refLattice, vacancies, voronoiOptions):
        """Calculate Voronoi."""
        self._logger.info("Calculating Voronoi (Defects)")
        self._voronoi = computeVoronoiDefects(lattice, refLattice, vacancies, voronoiOptions)


def computeVoronoi(lattice, voronoiOptions):
    """
    Compute Voronoi using Voro++
    
    """
    logger = logging.getLogger(__name__)
    
    vorotime = time.time()
    
    logger.info("Computing Voronoi")
    logger.debug("  NAtoms: %d", lattice.NAtoms)
    logger.debug("  PBCs are: %s %s %s", bool(lattice.PBC[0]), bool(lattice.PBC[1]), bool(lattice.PBC[2]))
    logger.debug("  Using radii: %s", voronoiOptions.useRadii)
    
    # Voronoi object
    vor = _voronoi.Voronoi()  # TODO: store info on vor obj, eg. useRadii, etc...
    
    # call c lib
    callTime = time.time()
    vor.computeVoronoi(lattice.pos, lattice.minPos, lattice.maxPos, lattice.cellDims, lattice.PBC, lattice.specie,
                       lattice.specieCovalentRadius, voronoiOptions.useRadii, voronoiOptions.faceAreaThreshold)
    callTime = time.time() - callTime
    
    # save to file
    writeTime = 0
    if voronoiOptions.outputToFile:
        writeTime = time.time()
        
        fn = voronoiOptions.outputFilename
        
        # TODO: make this a CLIB
        
        lines = []
        nl = lines.append
        
        nl("Atom index,Voronoi volume,Voronoi neighbours (faces)")
        
        for i in xrange(lattice.NAtoms):
            line = "%d,%f,%s" % (i, vor.atomVolume(i), vor.atomNumNebs(i))
            nl(line)
        
        nl("")
        
        logger.info("Writing Voronoi data to file: %s", fn)
        
        f = open(fn, "w")
        f.write("\n".join(lines))
        f.close()
        
        writeTime = time.time() - writeTime
    
    vorotime = time.time() - vorotime
    logger.debug("  Compute Voronoi time: %f", vorotime)
    logger.debug("    Compute time: %f", callTime)
    if writeTime > 0:
        logger.debug("    Write time: %f", writeTime)
    
    return vor


def computeVoronoiDefects(lattice, refLattice, vacancies, voronoiOptions):
    """
    Compute Voronoi for system containing defects
    
    """
    logger = logging.getLogger(__name__)
    
    vorotime = time.time()
    
    logger.info("Computing Voronoi (defects)")
    logger.debug("  NAtoms: %d; NVacancies: %d", lattice.NAtoms, len(vacancies))
    logger.debug("  PBCs are: %s %s %s", bool(lattice.PBC[0]), bool(lattice.PBC[1]), bool(lattice.PBC[2]))
    logger.debug("  Using radii: %s", voronoiOptions.useRadii)
    
    # Voronoi object
    vor = _voronoi.Voronoi()  # TODO: store info on vor obj, eg. useRadii, etc...
    
    # make new pos/specie arrays, containing vacancies too
    # TODO: write in C
    preptime = time.time()
    dim = lattice.NAtoms + len(vacancies)
    pos = np.empty(3 * dim, np.float64)
    specie = np.empty(dim, np.int32)
    pos[:3 * lattice.NAtoms] = lattice.pos[:]
    specie[:lattice.NAtoms] = lattice.specie[:]
    for i in xrange(len(vacancies)):
        ind = i + lattice.NAtoms
        ind3 = 3 * ind
        vacind = vacancies[i]
        vacind3 = 3 * vacind
        
        pos[ind3] = refLattice.pos[vacind3]
        pos[ind3 + 1] = refLattice.pos[vacind3 + 1]
        pos[ind3 + 2] = refLattice.pos[vacind3 + 2]
        
        specie[ind] = refLattice.specie[vacind]
    preptime = time.time() - preptime
    
    # call c lib
    callTime = time.time()
    vor.computeVoronoi(pos, lattice.minPos, lattice.maxPos, lattice.cellDims, lattice.PBC, specie,
                       lattice.specieCovalentRadius, voronoiOptions.useRadii, voronoiOptions.faceAreaThreshold)
    callTime = time.time() - callTime
    
    vorotime = time.time() - vorotime
    logger.debug("  Compute Voronoi (defects) time: %f", vorotime)
    logger.debug("    Prep time: %f", preptime)
    logger.debug("    Compute time: %f", callTime)
    
    return vor
