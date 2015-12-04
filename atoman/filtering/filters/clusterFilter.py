
"""
Cluster
=======

This filter will identify clusters of atoms in the system. It uses a recursive 
algorithm to build the clusters using a fixed cut-off. There are options to
calculate the volumes of the clusters and also to draw convex hulls around the
clusters to highlight them.

Parameters are:

.. glossary::

    Neighbour radius
        When constructing clusters two atoms are said to belong to the same
        cluster if their separation is less than this value.
    
    Minimum cluster size
        Clusters are only visible if they contain at least this number of atoms.
    
    Maximum cluster size
        Clusters are only visible if they contain less than this number of atoms.
        Set this parameter to `-1` if you do not want an upper limit on the
        cluster size.
    
    Draw convex hulls
        Compute and draw a convex hull around each cluster to highlight it.
    
    Hull colour
        The colour of the convex hulls, if `Draw convex hulls` is selected.
    
    Hull opacity
        The opacity of the convex hulls, if `Draw convex hulls` is selected.
    
    Hide atoms
        If `Draw convex hulls` is selected this will make the atoms invisible,
        so just the hulls are shown.
    
    Calculate volumes
        Calculate the volumes of the clusters of atoms.
    
    Calculate volumes Voronoi
        Sum the Voronoi volumes of the atoms in the cluster in order to calculate
        the cluster volume. The Voronoi volumes are computed using the *Voronoi
        settings* on the :ref:`voronoi_options_label` page.
    
    Calculate volumes hull
        The cluster volume is calculated from the volume of convex hulls of the
        set of points in the cluster.

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
        self.registerSetting("hullCol", default=[0, 0, 1])
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
        
        # hide atoms if required
        if settings.getSetting("hideAtoms"):
            visibleAtoms.resize(0, refcheck=False)
        
        # result
        result = base.FilterResult()
        result.setClusterList(clusterList)
        
        return result
