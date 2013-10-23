
"""
Module for computing Voronoi tesselation

@author: Chris Scott

"""
import time

import numpy as np
from scipy.spatial import Voronoi
import pyvoro



def computeVoronoi(lattice, log=None):
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
    
    vorotime = time.time()
    
    pyvoro_vor = pyvoro.compute_voronoi(pts,
                                        [[0.0, lattice.cellDims[0]], [0.0, lattice.cellDims[1]], [0.0, lattice.cellDims[2]]],
                                        10.0,
                                        periodic=[True, True, True])
    
    print "PYVORO TYPE", type(pyvoro_vor)
    print "PYVORO LEN", len(pyvoro_vor)
    
    index = 3768
    voro_cell = pyvoro_vor[index]
    for k in voro_cell.keys():
        print "%s:" % k, voro_cell[k]
    
    
    
    vorotime = time.time() - vorotime
    print "PYVORO VORO TIME", vorotime
    log("PYVORO VORO TIME: %f" % vorotime)
    
    return pyvoro_vor
    
    
