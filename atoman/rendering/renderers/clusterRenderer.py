
"""
Module for rendering clusters

"""
import time
import logging
import functools

import vtk
import numpy as np

from . import baseRenderer
from .. import utils
from ...filtering import _clusters
from ...filtering import clusters


class ClusterRenderer(baseRenderer.BaseRenderer):
    """
    Render clusters.
    
    """
    def __init__(self):
        super(ClusterRenderer, self).__init__()
        self._logger = logging.getLogger(__name__)
    
    def render(self, lattice, clusterList, settings):
        """
        Render the given clusters.
        
        """
        
        
        # loop over clusters
        for clusterIndex, cluster in enumerate(clusterList):
            appliedPBCs = np.zeros(7, np.int32)
            clusterPos = np.empty(3 * len(cluster), np.float64)
            for i in xrange(len(cluster)):
                index = cluster[i]
                clusterPos[3*i] = lattice.pos[3*index]
                clusterPos[3*i+1] = lattice.pos[3*index+1]
                clusterPos[3*i+2] = lattice.pos[3*index+2]
            
            neighbourRadius = settings.getSetting("neighbourRadius")
            _clusters.prepareClusterToDrawHulls(len(cluster), clusterPos, lattice.cellDims, 
                                                lattice.PBC, appliedPBCs, neighbourRadius)
            
            # get facets
            facets = None
            if len(cluster) > 3:
                facets = clusters.findConvexHullFacets(len(cluster), clusterPos)
            elif len(cluster) == 3:
                facets = []
                facets.append([0, 1, 2])
            
            # now render
            if facets is not None:
                #TODO: make sure not facets more than neighbour rad from cell
                facets = clusters.checkFacetsPBCs(facets, clusterPos, 2.0 * neighbourRadius,
                                                  lattice.PBC, lattice.cellDims)
                
                renderer.getActorsForHullFacets(facets, clusterPos, self.mainWindow,
                                                actorsDictLocal, settings,
                                                "Cluster {0}".format(clusterIndex))
            
            # handle PBCs
            if len(cluster) > 1:
                count = 0
                while max(appliedPBCs) > 0:
                    tmpClusterPos = copy.deepcopy(clusterPos)
                    clusters.applyPBCsToCluster(tmpClusterPos, lattice.cellDims, appliedPBCs)
                    
                    # get facets
                    facets = None
                    if len(cluster) > 3:
                        facets = clusters.findConvexHullFacets(len(cluster), tmpClusterPos)
                    elif len(cluster) == 3:
                        facets = []
                        facets.append([0, 1, 2])
                    
                    # render
                    if facets is not None:
                        #TODO: make sure not facets more than neighbour rad from cell
                        facets = clusters.checkFacetsPBCs(facets, tmpClusterPos, 
                                                          2.0 * neighbourRadius, lattice.PBC,
                                                          lattice.cellDims)
                        
                        renderer.getActorsForHullFacets(facets, tmpClusterPos, self.mainWindow,
                                                        actorsDictLocal, settings,
                                                        "Cluster {0} (PBC {1})".format(clusterIndex,
                                                        count))
                         
                        count += 1
        
