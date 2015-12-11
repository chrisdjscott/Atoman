
"""
Module for rendering clusters

"""
import copy
import logging

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
        self._logger.debug("Rendering %d clusters", len(clusterList))
        
        # append poly data
        appendPolyData = vtk.vtkAppendPolyData()
        
        # loop over clusters making poly data
        for clusterIndex, cluster in enumerate(clusterList):
            appliedPBCs = np.zeros(7, np.int32)
            clusterPos = np.empty(3 * len(cluster), np.float64)
            for i in xrange(len(cluster)):
                index = cluster[i]
                clusterPos[3 * i] = lattice.pos[3 * index]
                clusterPos[3 * i + 1] = lattice.pos[3 * index + 1]
                clusterPos[3 * i + 2] = lattice.pos[3 * index + 2]
            
            neighbourRadius = settings.getSetting("neighbourRadius")
            _clusters.prepareClusterToDrawHulls(len(cluster), clusterPos, lattice.cellDims, lattice.PBC, appliedPBCs,
                                                neighbourRadius)
            
            # get facets
            facets = None
            if len(cluster) > 3:
                facets = clusters.findConvexHullFacets(len(cluster), clusterPos)
            elif len(cluster) == 3:
                facets = []
                facets.append([0, 1, 2])
            
            # now render
            if facets is not None:
                # TODO: make sure not facets more than neighbour rad from cell
                facets = clusters.checkFacetsPBCs(facets, clusterPos, 2.0 * neighbourRadius, lattice.PBC,
                                                  lattice.cellDims)
                
                # make the poly data for this facet
                self.makeClusterPolyData(clusterPos, facets, appendPolyData)
            
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
                        # TODO: make sure not facets more than neighbour rad from cell
                        facets = clusters.checkFacetsPBCs(facets, tmpClusterPos, 2.0 * neighbourRadius, lattice.PBC,
                                                          lattice.cellDims)
                        
                        # make the poly data for this facet
                        self.makeClusterPolyData(tmpClusterPos, facets, appendPolyData)
                        
                        count += 1
        appendPolyData.Update()
        
        # remove any duplicate points
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(appendPolyData.GetOutputPort())
        cleanFilter.Update()
        
        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cleanFilter.GetOutputPort())
        
        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(settings.getSetting("hullOpacity"))
        hullCol = settings.getSetting("hullCol")
        actor.GetProperty().SetColor(hullCol[0], hullCol[1], hullCol[2])
        
        # store attributes
        self._actor = utils.ActorObject(actor)
    
    def makeClusterPolyData(self, pos, facets, appendPolyData):
        """
        Create polydata for a cluster.
        
        """
        points = vtk.vtkPoints()
        for i in xrange(len(pos) / 3):
            points.InsertNextPoint(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2])
        
        # create triangles
        triangles = vtk.vtkCellArray()
        for i in xrange(len(facets)):
            facet = facets[i]
            triangle = vtk.vtkTriangle()
            for j in xrange(3):
                triangle.GetPointIds().SetId(j, facet[j])
            triangles.InsertNextCell(triangle)
        
        # polydata object
        trianglePolyData = vtk.vtkPolyData()
        trianglePolyData.SetPoints(points)
        trianglePolyData.SetPolys(triangles)
        
        # add polydata
        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            appendPolyData.addInputConnection(trianglePolyData.GetProducerPort())
        else:
            appendPolyData.AddInputData(trianglePolyData)
