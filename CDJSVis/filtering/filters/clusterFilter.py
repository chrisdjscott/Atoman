
"""
Cluster
=======

This filter will identify clusters of atoms in the system...

"""
import numpy as np

from . import base
from .. import clusters
from .. import _clusters


class ClusterFilterSettings(base.BaseSettings):
    """
    Settings for the cluster filter
    
    """
    def __init__(self):
        super(ClusterFilterSettings, self).__init__()
        
        self.registerSetting("calculateVolumes", default=False)
        self.registerSetting("calculateVolumesVoro", default=True)
        self.registerSetting("calculateVolumesHull", default=False)
        self.registerSetting("hideAtoms", default=False)
        self.registerSetting("neighbourRadius", default=5.0)
        self.registerSetting("hullCol", default=[0,0,1])
        self.registerSetting("hullOpacity", default=0.5)
        self.registerSetting("minClusterSize", default=8)
        self.registerSetting("maxClusterSize", default=-1)
        self.registerSetting("drawConvexHulls", default=False)


class ClusterFilter(base.BaseFilter):
    """
    Cluster filter.
    
    """
    def apply(self, filterInput, settings):
        """Apply the filter."""
        # unpack inputs
        lattice = filterInput.inputState
        NScalars = filterInput.NScalars
        fullScalars = filterInput.fullScalars
        NVectors = filterInput.NVectors
        fullVectors = filterInput.fullVectors
        visibleAtoms = filterInput.visibleAtoms
        PBC = lattice.PBC
        
        # settings
        minSize = settings.getSetting("minClusterSize")
        maxSize = settings.getSetting("maxClusterSize")
        nebRad = settings.getSetting("neighbourRadius")
        
        # arrays for the cluster calculation
        atomCluster = np.empty(len(visibleAtoms), np.int32)
        result = np.empty(2, np.int32)
        
        # call C lib
        _clusters.findClusters(visibleAtoms, lattice.pos, atomCluster, nebRad, lattice.cellDims, PBC, 
                               minSize, maxSize, result, NScalars, fullScalars, NVectors, fullVectors)
        
        NVisible = result[0]
        NClusters = result[1]
        
        # resize arrays
        visibleAtoms.resize(NVisible, refcheck=False)
        atomCluster.resize(NVisible, refcheck=False)
        
        # build cluster lists
        clusterList = []
        for i in xrange(NClusters):
            clusterList.append(clusters.AtomCluster())
        
        # add atoms to cluster lists
        clusterIndexMapper = {}
        count = 0
        for i in xrange(NVisible):
            atomIndex = visibleAtoms[i]
            clusterIndex = atomCluster[i]
            
            if clusterIndex not in clusterIndexMapper:
                clusterIndexMapper[clusterIndex] = count
                count += 1
            
            clusterListIndex = clusterIndexMapper[clusterIndex]
            clusterList[clusterListIndex].indexes.append(atomIndex)
        
        # result
        result = base.FilterResult()
        result.setClusterList(clusterList)
        
        return result
