
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
        so just the hulls are shown. Cannot be selected at the same time as
        `Show atoms inside hulls` or `Show atoms outside hulls`.
    
    Show atoms inside hulls
        If `Draw convex hulls` is selected this will make all atoms that fall
        within a convex hull visible, regardless of previous filters (i.e. it
        acts on the original input lattice). Cannot be selected at the same
        time as `Hide atoms` or `Show atoms outside hulls`.
    
    Show atoms outside hulls
        If `Draw convex hulls` is selected this will make all atoms that fall
        outside the convex hulls visible, regardless of previous filters (i.e. it
        acts on the original input lattice). Cannot be selected at the same
        time as `Hide atoms` or `Show atoms inside hulls`.
    
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
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
import copy

import numpy as np
from scipy.spatial import Delaunay

from . import base
from .. import clusters
from .. import _clusters
from six.moves import range


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
        self.registerSetting("showAtomsInHulls", default=False)
        self.registerSetting("showAtomsOutHulls", default=False)
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
        voronoiCalculator = filterInput.voronoiAtoms
        
        # settings
        minSize = settings.getSetting("minClusterSize")
        maxSize = settings.getSetting("maxClusterSize")
        nebRad = settings.getSetting("neighbourRadius")
        calcVols = settings.getSetting("calculateVolumes")
        self.logger.debug("Cluster size: %d -> %d", minSize, maxSize)
        self.logger.debug("Neighbour radius: %f", nebRad)
        self.logger.debug("Calculating volumes: %r", calcVols)
        
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
        for i in range(NClusters):
            clusterList.append(clusters.AtomCluster(lattice))
        
        # add atoms to cluster lists
        clusterIndexMapper = {}
        count = 0
        for i in range(NVisible):
            atomIndex = visibleAtoms[i]
            clusterIndex = atomCluster[i]
            
            if clusterIndex not in clusterIndexMapper:
                clusterIndexMapper[clusterIndex] = count
                count += 1
            
            clusterListIndex = clusterIndexMapper[clusterIndex]
            clusterList[clusterListIndex].addAtom(atomIndex)
        
        # show all atoms inside or outside the clusters
        drawHulls = settings.getSetting("drawConvexHulls")
        showInHulls = settings.getSetting("showAtomsInHulls")
        showOutHulls = settings.getSetting("showAtomsOutHulls")
        if drawHulls and (showInHulls or showOutHulls):
            # first we calculate the Delaunay triangulation of each cluster (including periodic images)
            self.logger.debug("Calculating Delaunay triangulations for showing atoms")
            hulls, hullsMap = self.computeDelaunayForClusters(lattice, clusterList, nebRad)
            
            # for each atom determine whether it lies within a cluster or not
            self.logger.debug("Determining location of atoms (inside or outside clusters)")
            # TODO: write in C
            inClusterMask = np.zeros(lattice.NAtoms, np.int32)
            pos = np.empty((1, 3), np.float64)
            for i in range(lattice.NAtoms):
                pos[0][:] = lattice.atomPos(i)[:]
                for hull, hullMap in zip(hulls, hullsMap):
                    res = hull.find_simplex(pos) >= 0
                    if res[0]:
                        # set the mask to in cluster
                        inClusterMask[i] = 1
                        
                        # add to the cluster if doesn't already belong
                        cluster = clusterList[hullMap]
                        if i not in cluster:
                            cluster.addAtom(i)
                        
                        break
            
            # make the new visible atoms array, starting with full system
            self.logger.info("Overriding visible atoms based on cluster occupancy")
            visibleAtoms.resize(lattice.NAtoms, refcheck=False)
            visibleMask = 1 if showInHulls else 0
            # TODO: write in C
            numVisible = 0
            for i in range(lattice.NAtoms):
                if inClusterMask[i] == visibleMask:
                    visibleAtoms[numVisible] = i
                    numVisible += 1
            visibleAtoms.resize(numVisible, refcheck=False)
            
            # TODO: set cluster list to be empty on case of show out
            if showOutHulls:
                clusterList = []
        
        # calculate volumes
        if calcVols:
            self.logger.debug("Calculating cluster volumes")
            for i, cluster in enumerate(clusterList):
                cluster.calculateVolume(voronoiCalculator, settings)
                volume = cluster.getVolume()
                if volume is not None:
                    self.logger.debug("Cluster %d: volume is %f", i, volume)
                area = cluster.getFacetArea()
                if area is not None:
                    self.logger.debug("Cluster %d: facet area is %f", i, area)
        
        # hide atoms if required
        if drawHulls and settings.getSetting("hideAtoms"):
            visibleAtoms.resize(0, refcheck=False)
        
        # result
        result = base.FilterResult()
        result.setClusterList(clusterList)
        
        return result
    
    def computeDelaunayForClusters(self, lattice, clusterList, neighbourRadius):
        """Compute Delaunay triangulation for each cluster, including periodic images."""
        # build and store convex hulls for each cluster (unapply PBCs!?)
        self.logger.debug("Computing Delaunay for each hull (unapplying PBCs)")
        cellDims = lattice.cellDims
        hulls = []
        hullClusterMap = []
        for clusterIndex, cluster in enumerate(clusterList):
            appliedPBCs = np.zeros(7, np.int32)
            clusterPos = cluster.makeClusterPos()
            
            _clusters.prepareClusterToDrawHulls(len(cluster), clusterPos, cellDims, np.ones(3, np.int32), appliedPBCs,
                                                neighbourRadius)
            
            hulls.append(self.makeDelaunay(clusterPos))
            hullClusterMap.append(clusterIndex)
            
            # handle PBCs here
            while max(appliedPBCs) > 0:
                tmpClusterPos = copy.deepcopy(clusterPos)
                clusters.applyPBCsToCluster(tmpClusterPos, cellDims, appliedPBCs)
                hulls.append(self.makeDelaunay(clusterPos))
                hullClusterMap.append(clusterIndex)
        
        return hulls, hullClusterMap
    
    def makeDelaunay(self, clusterPos):
        """Calculate Delaunay for the given position."""
        # make pts
        # TODO: C or view
        num = len(clusterPos) // 3
        pts = np.empty((num, 3), np.float64)
        for i in range(num):
            i3 = 3 * i
            pts[i][0] = clusterPos[i3]
            pts[i][1] = clusterPos[i3 + 1]
            pts[i][2] = clusterPos[i3 + 2]
        
        # make hull
        hull = Delaunay(pts)
        
        return hull
