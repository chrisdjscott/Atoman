
"""
Module for computing Voronoi tesselation

@author: Chris Scott

"""
import time
import logging

import numpy as np
from scipy.spatial import Voronoi
import pyvoro


################################################################################

class VoronoiResult(object):
    """
    Object to hold Voronoi result
    
    """
    def __init__(self, voroList, pbc=None):
        self.voroList = voroList
        self.pbc = pbc
    
    def atomVolume(self, atomIndex):
        """
        Return Voronoi volume of given atom
        
        """
        return self.voroList[atomIndex]["volume"]
    
    def atomNumNebs(self, atomIndex):
        """
        Return number of neighbours (faces of Voronoi cell)
        
        """
        # check non neg???
        return len([v for v in self.voroList[atomIndex]["faces"] if v["adjacent_cell"] >= 0])
    
    def atomNebList(self, atomIndex):
        """
        Return list of neighbouring atom indexes
        
        """
        # what to do about negative??
        return [v["adjacent_cell"] for v in self.voroList[atomIndex]["faces"] if v["adjacent_cell"] >= 0]
    
    def getInputAtomPos(self, atomIndex):
        """
        Returns pos of atom that was used to calculate this Voro volume
        (eg. to check ordering is as expected)
        
        """
        return self.voroList[atomIndex]["original"]
    
    def atomVertices(self, atomIndex):
        """
        Return position of Voronoi vertices for given atoms Voronoi cell
        
        """
        return self.voroList[atomIndex]["vertices"]
    
    def atomFaces(self, atomIndex):
        """
        Return faces of Voronoi cell as list of list of indexes of vertices
        
        """
        negFound = False
        for v in self.voroList[atomIndex]["faces"]:
            if v["adjacent_cell"] < 0:
                negFound = True
                break
        
        if negFound:
            retval = None
        
        else:
            retval =  [v["vertices"] for v in self.voroList[atomIndex]["faces"]]
        
        return retval
        

################################################################################

def computeVoronoi(lattice, voronoiOptions, PBC):
    """
    Compute Voronoi
    
    """
    res = computeVoronoiPyvoro(lattice, voronoiOptions, PBC)
    
    computeVoronoiScipy(lattice, PBC)
    
    return res

################################################################################

def computeVoronoiPyvoro(lattice, voronoiOptions, PBC):
    """
    Compute Voronoi
    
    """
    logger = logging.getLogger(__name__+".computeVoronoiPyvoro")
    
    vorotime = time.time()
    
    logger.info("Computing Voronoi (pyvoro)")
    logger.debug("  NAtoms: %d", lattice.NAtoms)
    logger.info("  Dispersion is: %f", voronoiOptions.dispersion)
    logger.info("  PBCs are: %s %s %s", bool(PBC[0]), bool(PBC[1]), bool(PBC[2]))
    logger.info("  Using radii: %s", voronoiOptions.useRadii)
    
    # make points
    ptsTime = time.time()
    pts = np.empty((lattice.NAtoms, 3), dtype=np.float64)
    for i in xrange(lattice.NAtoms):
        pts[i][0] = lattice.atomPos(i)[0]
        pts[i][1] = lattice.atomPos(i)[1]
        pts[i][2] = lattice.atomPos(i)[2]
    ptsTime = time.time() - ptsTime
    
    # boundary
    upper = np.empty(3, np.float64)
    lower = np.empty(3, np.float64)
    for i in xrange(3):
        if PBC[i]:
            lower[i] = 0.0
            upper[i] = lattice.cellDims[i]
        
        else:
            lower[i] = lattice.minPos[i] - voronoiOptions.dispersion
            upper[i] = lattice.maxPos[i] + voronoiOptions.dispersion
    
    logger.info("  Limits: [[%f, %f], [%f, %f], [%f, %f]]", lower[0], upper[0], lower[1], upper[1], lower[2], upper[2])
    
    # radii
    if voronoiOptions.useRadii:
        radii = [lattice.specieCovalentRadius[specInd] for specInd in lattice.specie]
    
    else:
        radii = []
    
    # call pyvoro
    callTime = time.time()
    pyvoro_result = pyvoro.compute_voronoi(pts,
                                           [[lower[0], upper[0]], [lower[1], upper[1]], [lower[2], upper[2]]],
                                           voronoiOptions.dispersion,
                                           radii=radii,
                                           periodic=[bool(PBC[0]), bool(PBC[1]), bool(PBC[2])])
    callTime = time.time() - callTime
    
    # create result object
    resTime = time.time()
    vor = VoronoiResult(pyvoro_result, PBC)
    resTime = time.time() - resTime
    
    # save to file
    if voronoiOptions.outputToFile:
        fn = voronoiOptions.outputFilename
        
        #TODO: make this a CLIB
        
        lines = []
        nl = lines.append
        
        nl("Atom index,Voronoi volume,Voronoi neighbours (faces)")
        
        for i in xrange(lattice.NAtoms):
            line = "%d,%f,%s" % (i, vor.atomVolume(i), vor.atomNumNebs(i))
            nl(line)
        
        nl("")
        
        logger.info("  Writing Voronoi data to file: %s", fn)
        
        f = open(fn, "w")
        f.write("\n".join(lines))
        f.close()
    
#     print "CHECKING NEBS"
#     for i in xrange(lattice.NAtoms):
#         nebs = vor.atomNebList(i)
#         for neb in nebs:
#             if neb < 0:
#                 print "******* neg neb for %d" % i
    
    vorotime = time.time() - vorotime
    logger.debug("  Compute Voronoi time: %f", vorotime)
    logger.debug("    Make points time: %f", ptsTime)
    logger.debug("    Call time: %f", callTime)
    logger.debug("    Make result time: %f", resTime)
    
    return vor

################################################################################

def computeVoronoiScipy(lattice, PBC):
    """
    Compute Voronoi
    
    """
    logger = logging.getLogger(__name__+".computeVoronoiSciPy")
    
    vorotime = time.time()
    logger.info("Computing Voronoi (SciPy)")
    logger.debug("NAtoms = %d", lattice.NAtoms)
    logger.debug("PBC = %r", PBC)
     
    # make points
    pts = np.empty((lattice.NAtoms, 3), dtype=np.float64)
    for i in xrange(lattice.NAtoms):
        pts[i][0] = lattice.atomPos(i)[0]
        pts[i][1] = lattice.atomPos(i)[1]
        pts[i][2] = lattice.atomPos(i)[2]
    
    from . import _voronoi
    skin = 6.0
    pts2 = _voronoi.makeVoronoiPoints(lattice.pos, lattice.cellDims, PBC, skin)
    print "LEN PTS2", len(pts2)
    
    # compute
    vor = Voronoi(pts2)
     
    print vor
     
    assert len(vor.point_region) == lattice.NAtoms
      
    # atom near middle (3768)
    index = 3768
    assert index < lattice.NAtoms
     
    assert vor.point_region[index] != -1
     
    # point_region is index of that atoms region (in regions array)
    print "REGION OF %d: %d" % (index, vor.point_region[index])
    regionIndex = vor.point_region[index]
     
    # each element of region array holds array of indexes of vertices that make this region
    region = vor.regions[regionIndex]
    print "REGION:", len(region), region
     
    ok = True
    for vertid in region:
        if vertid == -1:
            ok = False
            break
     
    if not ok:
        print "ERROR: UNBOUNDED REGION"
    else:
        pass
    
    vorotime = time.time() - vorotime
    logger.debug("  Compute Voronoi time: %f", vorotime)
    
    return vor
