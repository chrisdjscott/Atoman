
"""
Module for rendering clusters

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import copy
import logging

import vtk
import numpy as np

from . import baseRenderer
from . import povrayWriters
from .. import utils
from ...filtering import _clusters
from ...filtering import clusters
from six.moves import range


class ClusterRenderer(baseRenderer.BaseRenderer):
    """
    Render clusters.
    
    """
    def __init__(self):
        super(ClusterRenderer, self).__init__()
        self._logger = logging.getLogger(__name__)
    
    def render(self, clusterList, settings, refState=None):
        """
        Render the given clusters.
        
        """
        self._logger.debug("Rendering %d clusters", len(clusterList))
        
        # object for combining poly datas
        appendPolyData = vtk.vtkAppendPolyData()
        
        # neighbour radius used for constructing clusters
        neighbourRadius = settings.getSetting("neighbourRadius")
        
        # loop over clusters making poly data
        for clusterIndex, cluster in enumerate(clusterList):
            # get the positions for this cluster
            clusterPos = cluster.makeClusterPos()
            
            # lattice
            lattice = cluster.getLattice()
            
            # get settings and prepare to render (unapply PBCs)
            appliedPBCs = np.zeros(7, np.int32)
            _clusters.prepareClusterToDrawHulls(len(cluster), clusterPos, lattice.cellDims, lattice.PBC, appliedPBCs,
                                                neighbourRadius)
            
            # render this clusters facets
            self.renderClusterFacets(len(cluster), clusterPos, lattice, neighbourRadius, appendPolyData)
            
            # handle PBCs
            if len(cluster) > 1:
                # move the cluster across each PBC that it overlaps
                while max(appliedPBCs) > 0:
                    # send the cluster across PBCs
                    tmpClusterPos = copy.deepcopy(clusterPos)
                    clusters.applyPBCsToCluster(tmpClusterPos, lattice.cellDims, appliedPBCs)
                    
                    # render the modified clusters facets
                    self.renderClusterFacets(len(cluster), tmpClusterPos, lattice, neighbourRadius, appendPolyData)
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
        self._data["Hull colour"] = hullCol
        self._data["Hull opacity"] = settings.getSetting("hullOpacity")
        self._data["Neighbour radius"] = neighbourRadius
        self._data["Cluster list"] = clusterList
    
    def renderClusterFacets(self, clusterSize, clusterPos, lattice, neighbourRadius, appendPolyData):
        """
        Render facets of a cluster.
        
        """
        # get facets
        facets = clusters.findConvexHullFacets(clusterSize, clusterPos)
        
        # now render
        if facets is not None:
            # TODO: make sure not facets more than neighbour rad from cell
            facets = clusters.checkFacetsPBCs(facets, clusterPos, 2.0 * neighbourRadius, lattice.PBC, lattice.cellDims)
            
            # create vtk points from cluster positions
            points = vtk.vtkPoints()
            for i in range(clusterSize):
                points.InsertNextPoint(clusterPos[3 * i], clusterPos[3 * i + 1], clusterPos[3 * i + 2])
            
            # create triangles
            triangles = vtk.vtkCellArray()
            for facet in facets:
                triangle = vtk.vtkTriangle()
                for j in range(3):
                    triangle.GetPointIds().SetId(j, facet[j])
                triangles.InsertNextCell(triangle)
            
            # polydata object
            trianglePolyData = vtk.vtkPolyData()
            trianglePolyData.SetPoints(points)
            trianglePolyData.SetPolys(triangles)
            
            # add polydata
            if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
                appendPolyData.AddInputConnection(trianglePolyData.GetProducerPort())
            else:
                appendPolyData.AddInputData(trianglePolyData)
    
    def writePovray(self, filename):
        """Write atoms to POV-Ray file."""
        self._logger.debug("Writing atoms POV-Ray file")
        
        # povray writer
        writer = povrayWriters.PovrayClustersWriter()
        writer.write(filename, self._data["Cluster list"], self._data["Neighbour radius"], self._data["Hull opacity"],
                     self._data["Hull colour"])
