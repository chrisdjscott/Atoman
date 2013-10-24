
"""
Module for computing Voronoi tesselation

@author: Chris Scott

"""
import time

import numpy as np
from scipy.spatial import Voronoi
import pyvoro


################################################################################

class VoronoiResult(object):
    """
    Object to hold Voronoi result
    
    """
    def __init__(self, voroList):
        self.voroList = voroList
    
    def atomVolume(self, atomIndex):
        """
        Return Voronoi volume of given atom
        
        """
        return self.voroList[atomIndex]["volume"]
    
    def atomNumNebs(self, atomIndex):
        """
        Return number of neighbours (faces of Voronoi cell)
        
        """
        return len(self.voroList[atomIndex]["faces"])
    
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
        return [v["vertices"] for v in self.voroList[atomIndex]["faces"]]

################################################################################

def computeVoronoi(lattice, log=None):
    """
    Compute Voronoi
    
    """
    return computeVoronoiPyvoro(lattice, log)

################################################################################

def computeVoronoiPyvoro(lattice, log=None):
    """
    Compute Voronoi
    
    """
    vorotime = time.time()
    print "COMPUTING VORONOI"
    print "NATOMS", lattice.NAtoms
     
    # make points
    pts = np.empty((lattice.NAtoms, 3), dtype=np.float64)
    for i in xrange(lattice.NAtoms):
        pts[i][0] = lattice.atomPos(i)[0]
        pts[i][1] = lattice.atomPos(i)[1]
        pts[i][2] = lattice.atomPos(i)[2]
    
    pyvoro_vor = pyvoro.compute_voronoi(pts,
                                        [[0.0, lattice.cellDims[0]], [0.0, lattice.cellDims[1]], [0.0, lattice.cellDims[2]]],
                                        10.0,
                                        periodic=[True, True, True])
    
    print "PYVORO TYPE", type(pyvoro_vor)
    print "PYVORO LEN", len(pyvoro_vor)
    
    # create result object
    vor = VoronoiResult(pyvoro_vor)
    
    vorotime = time.time() - vorotime
    print "PYVORO VORO TIME", vorotime
    log("PYVORO VORO TIME: %f" % vorotime)
    
    return vor

################################################################################

def computeVoronoiScipy(lattice, log=None):
    """
    Compute Voronoi
    
    """
    vorotime = time.time()
    print "COMPUTING VORONOI"
    print "NATOMS", lattice.NAtoms
     
    # make points
    pts = np.empty((lattice.NAtoms, 3), dtype=np.float64)
    for i in xrange(lattice.NAtoms):
        pts[i][0] = lattice.atomPos(i)[0]
        pts[i][1] = lattice.atomPos(i)[1]
        pts[i][2] = lattice.atomPos(i)[2]
    
    # compute
    vor = Voronoi(pts)
     
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
    print "SCIPY VORO TIME", vorotime
      
    log("SCIPY VORO TIME: %f" % vorotime)
    
    return vor
