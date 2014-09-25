
"""
Module for computing Voronoi tesselation

@author: Chris Scott

"""
import time
import logging

import numpy as np
from scipy.spatial import Voronoi
import pyvoro

from . import _voronoi
from . import clusters


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
    
    vor, vols = computeVoronoiScipy(lattice, PBC)
    
#     import math
#     for i in xrange(lattice.NAtoms):
#         if math.fabs(vols[i] - res.atomVolume(i)) > 1e-5:
#             print "VOLDIFF(%d): %.10f <-> %.10f" % (i, vols[i], res.atomVolume(i))
    
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
    
    skin = 5.0 # should be passed in
    
    vorotime = time.time()
    logger.info("Computing Voronoi (SciPy)")
    logger.debug("NAtoms = %d", lattice.NAtoms)
    logger.debug("PBC = %r", PBC)
    logger.debug("Skin = %f", skin)
     
    ptsTime = time.time()
    pts = _voronoi.makeVoronoiPoints(lattice.pos, lattice.cellDims, PBC, skin)
    ptsTime = time.time() - ptsTime
    
#     f = open("vorolattice.dat", "w")
#     f.write("%d\n" % len(pts))
#     f.write("%f %f %f\n" % tuple(lattice.cellDims))
#     for i in xrange(len(pts)):
#         f.write("Au %f %f %f 0.0\n" % (pts[i][0], pts[i][1], pts[i][2]))
#     f.close()
    
    # compute
    compTime = time.time()
    vor = Voronoi(pts)
    compTime = time.time() - compTime
    assert len(vor.point_region) == len(pts)
    
    # calculate volumes (this is probably slow!; just testing at the moment)
    volsTime = time.time()
    volumes = np.empty(lattice.NAtoms, np.float64)
    for i in xrange(lattice.NAtoms):
        # point region is index of this atoms region (in regions array)
        regionIndex = vor.point_region[i]
        
        # each element of region array holds array of indices of vertices that make this region
        region = vor.regions[regionIndex]
        
        # check if Voronoi region is unbounded (shouldn't happen!)
        # and construct pts array (should be in C!)
        pts = []
        unbounded = False
        for vertid in region:
            if vertid == -1:
                unbounded = True
                break
            
            pts.append(vor.vertices[vertid])
        
        if unbounded:
            logger.warning("Atom %d Voronoi region is unbounded", i)
            volume = 0.0
        
        else:
            # calculate volume of convex hull of region vertices
            volume, _ = clusters.findConvexHullVolume(len(region), pts, posIsPts=True)
        
        # store volume
        volumes[i] = volume
    volsTime = time.time() - volsTime
    
    vorotime = time.time() - vorotime
    logger.debug("  Compute Voronoi time: %f", vorotime)
    logger.debug("    Make pts time: %f", ptsTime)
    logger.debug("    Compute time: %f", compTime)
    logger.debug("    Volumes time: %f", volsTime)
    
    return vor, volumes
